from __future__ import annotations
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import MODEL_VERSION,__version__
from .io import write_json,write_csv
from .geometry import optical_system
from .validation import error_metrics,validate


def environment()->dict:
    return dict(python_executable=sys.executable,python=sys.version,platform=platform.platform(),
                numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,matplotlib=matplotlib.__version__,
                package=__version__)


def calibration_document(result:dict,c:dict)->dict:
    from .image_inputs import input_mode,crop_origins
    state=result['state']; system=optical_system(state['prism'],state['omega'])
    opt=c.get('optimization',{})
    status={}
    for names,key in [(('j','k','l'),'release_K'),(('omega_x','omega_y','omega_z'),'release_omega'),(('kappa1','kappa2'),'release_radial')]:
        for n,release in zip(names,opt.get(key,[True]*len(names))):
            status[n]=dict(status='estimated' if release else 'fixed',
                           source=c['camera'].get('omega_source','configured') if key=='release_omega' else 'configured/optimized',
                           trustworthy_mechanical_value=None,
                           local_data_identifiable=bool(result['status']['identifiability_passed']) if release else None)
    status['c']=dict(status=c['camera']['c_source'],value=state['c'],optimized=False)
    for k in state['prism']:
        status['prism.'+k]=dict(status='fixed',source=c['prism'].get('source','configured'))
    hardware=result['diagnostics'].get('hardware_refinement')
    if hardware:
        for name,item in hardware['parameters'].items():
            status['prism.'+name]=dict(status='estimated_with_nominal_prior',
                source='bounded joint fit',trustworthy_mechanical_value=None,
                local_data_identifiable=bool(result['status']['identifiability_passed']),**item)
    return dict(model_version=MODEL_VERSION,state=state,K_eff=state['K'],c=state['c'],omega_p=state['omega'],Rp=system['Rp'],
                optimizer_search=result['diagnostics'].get('optimizer_search'),
                hardware_refinement=hardware,
                distortion=dict(kappa=state['kappa'],q_scale_mm=state['q_scale'],
                                normalization=c['camera'].get('q_scale_resolution'),
                                model='normalized reference-plane radial r2,r4'),
                radius_limit_mm=c['camera'].get('radius_limit_mm') or 2*state['q_scale'],
                prism=state['prism'],G_L=system['L']['G'],G_R=system['R']['G'],
                H=np.vstack([system['L']['M'],system['R']['M']]),
                H_row_order='physical L.x,L.y,R.x,R.y',
                view_branches={v:c['views'][v]['branch'] for v in ('left','right')},views=c['views'],
                input_mode=input_mode(c),crop_origins_by_image=crop_origins(c),
                default_crop_origins={v:c['views'][v].get('crop_origin_px') for v in ('left','right')},
                extrinsics=result['poses'],parameter_status=status,mechanical_angles=result['mechanical_angles'],
                units=dict(length='mm',pixels='px',angles='rad'),
                coordinates=dict(pixel_origin='full frame top-left pixel center (0,0)',u='right',v='down',
                                 lens_z='towards object',handedness='right-handed prism and lens frames',
                                 prism_back='z=0',prism_front='tau_s*x+z=thickness',
                                 prism_to_lens='Rp; no transverse translation assumed',
                                 image_axis_transform='identity; common optical inversion carried by signed j,k',
                                 board_to_prism='R_i @ P + t_i; no first pose artificially fixed'),
                status=result['status'],training_frames=result['training_frames'],validation_frames=result['validation_frames'],
                config_snapshot=c)


def save_results(result:dict,c:dict,obs:dict,output:Path,command:str)->dict:
    cal=calibration_document(result,c)
    val=validate(cal,obs['used'],result['validation_frames'],c)
    cal['status']['external_validation_available']=val['external_validation_available']
    write_json(output/'calibration.json',cal)
    write_json(output/'diagnostics.json',result['diagnostics'])
    if cal.get('hardware_refinement'):
        write_json(output/'hardware_refinement.json',cal['hardware_refinement'])
    if result['diagnostics'].get('optimizer_search'):
        from .solver_search import save_report
        save_report(result['diagnostics']['optimizer_search'],output)
    write_json(output/'validation.json',val)
    write_json(output/'config_snapshot.json',c)
    write_json(output/'environment.json',environment())
    write_csv(output/'observations.csv',[v['row'] for v in obs['used']])
    validation_rows=[v['row'] for v in obs['used'] if v['row']['image_id'] in result['validation_frames']]
    write_csv(output/'validation_observations.csv',validation_rows,
              list(obs['used'][0]['row']) if not validation_rows else None)
    write_csv(output/'extrinsics.csv',[dict(image_id=f,**dict(zip(('rx_rad','ry_rad','rz_rad','tx_mm','ty_mm','tz_mm'),p))) for f,p in result['poses'].items()])
    perpoint=[]
    threshold=c.get('optimization',{}).get('outlier_review_px',2.)
    for v,e in zip(result['data']['records'],result['raw_errors']):
        perpoint.append(dict(image_id=v['row']['image_id'],view=v['row']['view'],point_id=v['row']['point_id'],
                             u_px=v['p'][0],v_px=v['p'][1],du_px=e[0],dv_px=e[1],error_px=np.linalg.norm(e),
                             needs_outlier_review=np.linalg.norm(e)>threshold,
                             handling='retained with configured robust loss; no dynamic deletion'))
    write_csv(output/'per_point_errors.csv',perpoint)
    frame_rows=[]
    for f in result['training_frames']:
        e=np.array([[r['du_px'],r['dv_px']] for r in perpoint if r['image_id']==f])
        metrics=error_metrics(e)
        frame_rows.append(dict(image_id=f,euclidean_rms_px=metrics['euclidean_rms_px'],p95_px=metrics['p95_px'],count=len(e)))
    write_csv(output/'per_frame_errors.csv',frame_rows)
    plot_diagnostics(perpoint,val,output)
    metrics=error_metrics(result['raw_errors'])
    write_json(output/'metrics.json',dict(training=metrics,held_out=val['metrics'],
                                        whitened_sum_squares=result['whitened_sum_squares'],robust_objective=result['robust_objective']))
    report=[f'# BBS calibration report ({c.get("dataset_kind","real")})','',
            'Experimental joint model; implementation, feature quality, identifiability and external precision are separate claims.','',
            f'- Command: `{command}`',f'- Python: `{sys.executable}`',f'- Seed: {c.get("seed",1234)}',
            f'- Optimization + diagnostics time: {result["elapsed_seconds"]:.3f} s',
            f'- Training frames: {result["training_frames"]}',f'- Whole-frame validation: {result["validation_frames"]}',
            f'- Pixel Euclidean RMS = sqrt(mean(du^2+dv^2)): {metrics["euclidean_rms_px"]:.8g} px',
            f'- Coordinate component RMS: {metrics["component_rms_px"]:.8g} px',
            f'- Per-axis RMS: {metrics["axis_rms_px"]}; means: {metrics["axis_mean_px"]}',
            f'- P95 point error: {metrics["p95_px"]:.8g} px',
            f'- Whitened unrobust sum squares: {result["whitened_sum_squares"]:.8g}',
            f'- Robust objective (including any priors): {result["robust_objective"]:.8g}',
            f'- Held-out pixel RMS: {val["metrics"].get("euclidean_rms_px")}',
            f'- Excluded/unconfirmed observations: {len(obs["excluded"])}',
            f'- Outlier-review observations retained: {sum(r["needs_outlier_review"] for r in perpoint)}',
            '', '## Evidence status','']
    report += ['- '+k+': '+str(v) for k,v in cal['status'].items()]
    report += ['', 'Implementation test status is false until a matching test run is linked; optimizer status does not set it.',
               'Feature contours and detection audit remain in the detection run directory.',
               'Held-out poses use a subset of point IDs; disjoint IDs evaluate the fixed global model.',
               'Board distance validation uses no free-scale alignment. External translation truth: '+str(val['external_validation_available']),
               '', '## Warnings','']+[ '- '+w for w in result['diagnostics']['warnings']]
    report += ['', 'Mechanical angles: '+str(result['mechanical_angles']),
               'See diagnostics.json for scaled DATA and prior-augmented Jacobians and multistart spread.',
               'See docs/limitations.md. No claim of actual-system precision follows from this report.']
    (output/'report.md').write_text('\n'.join(report),encoding='utf-8')
    return cal


def plot_diagnostics(rows:list,val:dict,output:Path)->None:
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,view in zip(axes,('left','right')):
        r=[x for x in rows if x['view']==view]
        ax.quiver([x['u_px'] for x in r],[x['v_px'] for x in r],
                  [x['du_px'] for x in r],[x['dv_px'] for x in r],angles='xy')
        ax.invert_yaxis(); ax.set_title(view+' reprojection residual vectors (autoscaled)'); ax.set_xlabel('u / px');ax.set_ylabel('v / px')
    fig.tight_layout();fig.savefig(output/'reprojection_vectors.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for view in ('left','right'):
        ax.hist([x['error_px'] for x in rows if x['view']==view],bins=25,alpha=.5,label=view)
    ax.legend();ax.set_xlabel('Euclidean pixel residual / px');ax.set_ylabel('Count')
    fig.tight_layout();fig.savefig(output/'error_distribution.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(7,5));ax=fig.add_subplot(111,projection='3d')
    for r in val['per_frame']:
        X=r.get('reconstruction',{}).get('X_mm')
        if X is not None:
            ax.scatter(*np.asarray(X).T,s=10,label=r['image_id'])
    ax.set_xlabel('prism X / mm');ax.set_ylabel('prism Y / mm');ax.set_zlabel('prism Z / mm')
    if 0<len(ax.collections)<=12: ax.legend()
    elif ax.collections: ax.set_title('%d validation frames (legend omitted for readability)'%len(ax.collections))
    fig.tight_layout();fig.savefig(output/'validation_3d.png',dpi=150);plt.close(fig)
