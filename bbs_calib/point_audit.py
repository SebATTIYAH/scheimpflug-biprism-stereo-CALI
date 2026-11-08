"""Read-only audit of high-residual points from rectangular paired-crop grids.

Ellipse edges are recomputed, NOT historical stored edges. Never refits calibration.
"""
from pathlib import Path
from collections import Counter
import hashlib
import cv2
import numpy as np
from .io import read_json, read_csv, write_json, write_csv, read_gray, save_image, new_output
from .image_inputs import discover_frames
from .detection import refine_ellipse

EDGE_SETTINGS = dict(edge_search_px=10, boundary_samples=96, profile_step_px=.2,
                     min_profile_contrast=.05, profile_smoothing_px=1.5)


def key(row):
    return row['image_id'], row['view'], str(row['point_id'])


def ranked_points(errors, observations, origins, top):
    if top < 1:
        raise ValueError('--top must be positive')
    lookup = {key(r): r for r in observations}
    if len(lookup) != len(observations) or len({key(r) for r in errors}) != len(errors):
        raise ValueError('Duplicate observation keys')
    result = []
    for row in errors:
        obs = lookup[key(row)]
        measured = np.array([float(row[k]) for k in ('u_px', 'v_px')])
        stored = np.array([float(obs[k]) for k in ('u_px', 'v_px')])
        source = np.array([float(obs[k]) for k in ('source_u_px', 'source_v_px')])
        origin = np.array(origins[row['image_id']][row['view']], float)
        delta = np.array([float(row[k]) for k in ('du_px', 'dv_px')])
        error = float(row['error_px'])
        if not np.isfinite(np.r_[measured, stored, source, origin, delta, error]).all():
            raise ValueError('Nonfinite observation')
        if not np.allclose(measured, stored, atol=1e-7, rtol=0) or not np.allclose(source+origin, measured, atol=1e-7, rtol=0):
            raise ValueError('Inconsistent source/sensor coordinates: '+str(key(row)))
        if not np.isclose(np.linalg.norm(delta), error, atol=1e-7, rtol=0):
            raise ValueError('Residual norm mismatch')
        result.append(dict(image_id=row['image_id'], view=row['view'], point_id=int(row['point_id']),
                           u_px=measured[0], v_px=measured[1], source_u_px=source[0], source_v_px=source[1],
                           predicted_source_u_px=source[0]+delta[0], predicted_source_v_px=source[1]+delta[1],
                           du_px=delta[0], dv_px=delta[1], error_px=error,
                           stored_edge_rms_px=float(obs['fit_error_px'])))
    result.sort(key=lambda r: (-r['error_px'], r['image_id'], r['view'], r['point_id']))
    return [dict(rank=i+1, **r) for i, r in enumerate(result[:top])]


def reproduce(gray, contour, settings=None):
    x,y,w,h = cv2.boundingRect(contour)
    x0,y0 = max(0,x-22), max(0,y-22)
    patch = gray[y0:min(gray.shape[0],y+h+22), x0:min(gray.shape[1],x+w+22)]
    fit = refine_ellipse(patch, contour-[x0,y0], EDGE_SETTINGS if settings is None else settings)
    fit['patch_origin_source_px'] = [x0,y0]
    return patch, fit


def ellipse_curve(fit):
    t = np.linspace(0,2*np.pi,361)
    a,b = fit['semi_axes']; theta = fit['theta_rad']
    rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.column_stack([a*np.cos(t), b*np.sin(t)])@rotation.T+fit['center']


def draw_panel(ax, patch, fit, row, raw=False, zoom=False):
    ax.imshow(patch, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax.set_aspect('equal')
    if raw:
        ax.set_title('Raw pixels (no contrast stretch)')
        return
    offset = np.array(fit['patch_origin_source_px'])
    measured = np.array([row['source_u_px'],row['source_v_px']])-offset
    predicted = np.array([row['predicted_source_u_px'],row['predicted_source_v_px']])-offset
    boundary = np.array(fit['boundary']); curve = ellipse_curve(fit)
    if not zoom:
        ax.scatter(*boundary.T,s=5,c='#00ffff',label='Recomputed edges',zorder=3)
        ax.plot(*curve.T,c='#ffdd00',lw=.8,label='Recomputed ellipse',zorder=4)
    ax.plot(*measured,'+',c='#ff3344',ms=12,mew=1.5,label='Stored measured center',zorder=7)
    ax.plot(*predicted,'x',c='#33aaff',ms=9,mew=1.5,label='Model prediction',zorder=8)
    ax.plot(*fit['center'],'o',mfc='none',mec='#ffdd00',ms=7,label='Recomputed center',zorder=6)
    if zoom:
        radius = max(1., row['error_px']*1.35)
        ax.set_xlim(measured[0]-radius,measured[0]+radius)
        ax.set_ylim(measured[1]+radius,measured[1]-radius)
        ax.plot([measured[0],predicted[0]],[measured[1],predicted[1]],c='#33aaff',lw=.8)
        ax.set_title('Center zoom: true pixel coordinates')
        ax.grid(alpha=.2)
    else:
        ax.set_title('Edges / ellipse / centers')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(calibration, output, top=30):
    from bbs_calib.prism_workflow import find_grid,refinement_settings
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(calibration).resolve()
    path = path/'calibration.json' if path.is_dir() else path
    cal = read_json(path)
    config = cal.get('config_snapshot', {})
    board = config.get('board', {})
    if cal.get('model_version') != 'bbs-shared-ray-radial-v1' or cal.get('input_mode') != 'paired_crops' or not board.get('rows') or not board.get('cols'):
        raise ValueError('audit-points requires rectangular full-grid biprism paired crops')
    profile=config.get('full_grid_extraction',{}).get('dot_profile','legacy')
    error_path, obs_path = path.parent/'per_point_errors.csv', path.parent/'observations.csv'
    errors, obs = read_csv(error_path), read_csv(obs_path)
    if not errors:
        raise ValueError('No training residuals to audit')
    if not all(r['image_id'] in cal['training_frames'] for r in errors):
        raise ValueError('Residual file contains non-training frames')
    selected = ranked_points(errors,obs,cal['crop_origins_by_image'],top)
    frames = {r['image_id']:r for r in discover_frames(config)}
    tracked = {str(p):digest(p) for p in (path,error_path,obs_path)}
    out = new_output(Path(output))
    cache, cards, rows = {}, [], []
    for row in selected:
        frame_key = (row['image_id'], row['view'])
        source = frames[row['image_id']][row['view']]
        if frame_key not in cache:
            tracked[str(source)] = digest(source)
            gray = read_gray(source)
            if gray.dtype != np.uint8:
                raise ValueError('This full-grid extractor requires uint8 original images')
            if profile=='standard':
                from .standard_grid import find_standard_grid
                blobs,indices,attempts=find_standard_grid(gray,board['rows'],board['cols'])
            else:
                blobs, indices, attempts = find_grid(gray,rows=board['rows'],cols=board['cols'],min_area_px=8 if profile=='small' else 150)
            cache[frame_key] = (gray,blobs,indices,attempts)
        gray,blobs,indices,attempts = cache[frame_key]
        row['source_image'] = str(source)
        stem = '%02d_%s_%s_p%03d'%(row['rank'],row['image_id'],row['view'],row['point_id'])
        try:
            if indices is None:
                raise ValueError('Full-grid detection failed during reproduction')
            contour=blobs[indices[row['point_id']]][1]
            settings=refinement_settings(contour,profile)
            patch, fit = reproduce(gray,contour,settings)
            center = np.array(fit['center'])+fit['patch_origin_source_px']
            difference = np.linalg.norm(center-[row['source_u_px'],row['source_v_px']])
            row.update(recomputed_center_difference_px=float(difference),
                       recomputed_edge_rms_px=fit['fit_error'], angular_coverage=fit['angular_coverage'],
                       axis_ratio=min(fit['semi_axes'])/max(fit['semi_axes']),
                       edge_contrast=fit['contrast'], fit_converged=fit['fit_converged'],
                       reproduction_status='matched' if difference<=1e-5 and fit['fit_converged'] else 'REVIEW_MISMATCH')
            save_image(out/'raw_crops'/(stem+'.png'),patch)
            write_json(out/'edge_data'/(stem+'.json'),dict(point=row,fit=fit,settings=settings,
                       attempts=attempts,coordinate_space='patch-local pixel centers',
                       note='Recomputed edges; stored observations and calibration are unchanged.'))
            fig,axes = plt.subplots(1,3,figsize=(13,5.4),dpi=120)
            draw_panel(axes[0],patch,fit,row,raw=True)
            draw_panel(axes[1],patch,fit,row)
            draw_panel(axes[2],patch,fit,row,zoom=True)
            for ax in axes: ax.set_xlabel('Patch x [px]'); ax.set_ylabel('Patch y [px]')
            handles,labels = axes[1].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
            fig.suptitle('#%d  %s / %s / point %d\nReprojection %.4f px (du=%+.4f, dv=%+.4f); edge RMS %.4f px; coverage %.3f\nCenter reproduction difference %.3g px [%s]'
                         %(row['rank'],row['image_id'],row['view'],row['point_id'],row['error_px'],row['du_px'],row['dv_px'],fit['fit_error'],fit['angular_coverage'],difference,row['reproduction_status']),fontsize=10)
            fig.subplots_adjust(left=.06,right=.985,bottom=.23,top=.73,wspace=.35)
            card = out/'cards'/(stem+'.png');card.parent.mkdir(exist_ok=True)
            fig.savefig(str(card));plt.close(fig)
            row['card'] = card.relative_to(out).as_posix()
            if len(cards)<6: cards.append((patch,fit,dict(row)))
        except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:
            row.update(reproduction_status='FAILED',reason=str(exc))
        rows.append(row)
        print('Audit %d/%d: %s, error %.4f px'%(row['rank'],len(selected),row['reproduction_status'],row['error_px']),flush=True)
    if cards:
        fig,axes = plt.subplots(2,3,figsize=(12,10),dpi=110)
        for ax in axes.flat:ax.set_visible(False)
        for ax,(patch,fit,row) in zip(axes.flat,cards):
            ax.set_visible(True);draw_panel(ax,patch,fit,row)
            ax.set_title('#%d %s / p%d\n%s\nReproj %.3f px; edge %.3f px'%(row['rank'],row['view'],row['point_id'],row['image_id'],row['error_px'],row['recomputed_edge_rms_px']),fontsize=9)
        handles,labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
        fig.suptitle('Highest training reprojection errors: recomputed edge audit',fontsize=12)
        fig.subplots_adjust(left=.055,right=.98,bottom=.105,top=.865,wspace=.27,hspace=.43)
        fig.savefig(str(out/'overview.png'));plt.close(fig)
    unchanged = all(digest(p)==sha for p,sha in tracked.items())
    write_csv(out/'points.csv',rows)
    summary = dict(calibration=str(path),selection='Largest TRAINING residuals; not a random or held-out sample',
                   requested_top=top,selected_count=len(rows),training_count=len(errors),
                   training_rms_px=float(np.sqrt(np.mean([float(r['error_px'])**2 for r in errors]))),
                   matched_count=sum(r['reproduction_status']=='matched' for r in rows),
                   selected_views=dict(Counter(r['view'] for r in rows)),
                   selected_board_rows_1based=dict(Counter(r['point_id']//board['cols']+1 for r in rows)),
                   selected_positive_dv_count=sum(r['dv_px']>0 for r in rows),
                   inputs_unchanged=unchanged,input_sha256=tracked,settings=EDGE_SETTINGS,
                   residual_convention='predicted minus measured; vectors are not amplified',
                   no_observations_removed=True,no_calibration_refit=True)
    write_json(out/'summary.json',summary)
    lines=['# High-reprojection-error point audit','',
           'Selected the %d points with the largest training-set reprojection errors. The training set contains %d points, with an original RMS of %.9f px.'%(len(rows),len(errors),summary['training_rms_px']),
           '', 'Diagnostic only: no points were removed, no centers were changed, and no recalibration was performed. SHA256 checks of the input files: '+('all unchanged.' if unchanged else 'changes detected; verify the inputs!'),
           '', 'Selected points by view: '+str(summary['selected_views'])+'; by calibration-grid row (1-based): '+str(summary['selected_board_rows_1based'])+'.',
           '%d selected points have dv > 0, meaning the predicted position is below the measured center. This describes only the selected high-error points, not all points.'%summary['selected_positive_dv_count'],
           '', '## Reading the images','',
           '- cards contains the original crop, an edge and ellipse overlay, and a magnified view of center pixel coordinates. raw_crops preserves the original uint8 grayscale values.',
           '- Red + marks the saved measured center; blue x marks the model prediction; cyan dots mark newly extracted edges; the yellow ellipse and open circle show the refitted results.',
           '- du/dv = predicted − measured. The right panel magnifies the coordinate range only; it does not exaggerate the displacement vector.',
           '- Edge RMS is the ellipse Sampson residual, not center accuracy. Reprojection error is the discrepancy between measurement and model. They are not interchangeable.',
           '- The historical files did not preserve edges, so they are regenerated with the original algorithm and parameters. A point is marked matched only when the center difference is ≤ 1e-5 px and the fit converges; all other points require review.',
           '- High-error points were selected with bias. They cannot characterize every image or establish center extraction as the main error source. Do not replace measured centers with model predictions.',
           '', '## Point-by-point results','', '|Rank|Frame / view / point ID|Reprojection error (px)|Edge RMS (px)|Center reproduction difference (px)|Status|', '|---|---|---:|---:|---:|---|']
    for r in rows:
        name='%s / %s / %s'%(r['image_id'],r['view'],r['point_id'])
        if 'card' in r:name='[%s](%s)'%(name,r['card'])
        lines.append('|%d|%s|%.4f|%s|%s|%s|'%(r['rank'],name,r['error_px'],
                     str(r.get('recomputed_edge_rms_px',''))[:8],str(r.get('recomputed_center_difference_px',''))[:10],r['reproduction_status']))
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not unchanged:raise ValueError('Input files changed during audit; see summary.json')
    return out
