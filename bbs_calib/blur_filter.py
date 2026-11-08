"""Image-only edge-width filtering; no residuals or model predictions are read."""
from pathlib import Path
from collections import Counter
import hashlib
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
from .io import read_csv, read_json, write_csv, write_json, read_gray, save_image, new_output
from .config import load_config
from .image_inputs import discover_frames
from .review_images import resize_points, draw_center


def profile_width(profile, t):
    """10-90% transition width, normalized by local inside/outside plateaus."""
    p = gaussian_filter1d(np.asarray(profile,float),.5/(t[1]-t[0]))
    low,high = np.median(p[t<=-7]),np.median(p[t>=7])
    if high-low < 20:
        return None
    q=(p-low)/(high-low)
    crossing=np.flatnonzero((q[:-1]<=.5)&(q[1:]>.5))
    if not len(crossing):return None
    mid=crossing[np.argmin(np.abs(t[crossing]))]
    if abs(t[mid])>4:return None
    before=np.flatnonzero((q[:-1]<=.1)&(q[1:]>.1)&(np.arange(len(q)-1)<=mid))
    after=np.flatnonzero((q[:-1]<=.9)&(q[1:]>.9)&(np.arange(len(q)-1)>=mid))
    if not len(before) or not len(after):return None
    def interpolate(i,level):return t[i]+(t[i+1]-t[i])*(level-q[i])/(q[i+1]-q[i])
    w=interpolate(after[0],.9)-interpolate(before[-1],.1)
    return float(w) if w>0 else None


def edge_width(gray, contour, measured):
    (cx,cy),(wa,wb),angle=cv2.fitEllipseAMS(np.asarray(contour,np.float32))
    if np.linalg.norm(np.array([cx,cy])-measured)>3:
        raise ValueError('Contour/point correspondence check failed (>3 px)')
    a,b=wa/2,wb/2;theta=np.radians(angle)
    if min(a,b)<=10:
        raise ValueError('Dot too small for the legacy +/-10 px edge-width metric; requires manual review')
    angles=np.arange(96)*2*np.pi/96
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    perimeter=np.column_stack([a*np.cos(angles),b*np.sin(angles)])@rotation.T+[cx,cy]
    normals=np.column_stack([np.cos(angles)/a,np.sin(angles)/b])@rotation.T
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    t=np.linspace(-10,10,101)
    locations=perimeter[:,None,:]+normals[:,None,:]*t[None,:,None]
    valid=np.all((locations[:,:,0]>=0)&(locations[:,:,0]<=gray.shape[1]-1)&
                 (locations[:,:,1]>=0)&(locations[:,:,1]<=gray.shape[0]-1),axis=1)
    # Only convert the small sampled ROI, not the entire sensor, to float per dot.
    x0,y0=np.maximum(np.floor(locations.min(axis=(0,1)))-1,0).astype(int)
    x1,y1=np.minimum(np.ceil(locations.max(axis=(0,1)))+2,[gray.shape[1],gray.shape[0]]).astype(int)
    patch=gray[y0:y1,x0:x1].astype(float)
    profiles=map_coordinates(patch,[locations[:,:,1]-y0,locations[:,:,0]-x0],order=1,mode='nearest')
    widths=[profile_width(p,t) if v else None for p,v in zip(profiles,valid)]
    good=[w for w in widths if w is not None]
    if len(good)<72:
        raise ValueError('Insufficient reliable edge profiles (<72/96)')
    return dict(edge_width_median_px=float(np.median(good)),edge_width_p75_px=float(np.percentile(good,75)),
                edge_width_p90_px=float(np.percentile(good,90)),valid_profiles=len(good))


def keep_decision(metric, maximum):
    if not np.isfinite(maximum) or maximum<=0:raise ValueError('Edge-width threshold must be positive and finite')
    return 'blurred' if metric['edge_width_p75_px']>maximum else 'kept'


def examples(scores,frames,out):
    """Raw pixel crops spanning the observed sharpness distribution, not residuals."""
    ranked=sorted((s for s in scores if 'edge_width_p75_px' in s),key=lambda s:s['edge_width_p75_px'])
    if not ranked:return
    indices=sorted(set(int(round(q*(len(ranked)-1))) for q in (.1,.5,.8,.9,.95,1.)))
    canvas=np.full((760,1020,3),235,np.uint8);selected=[]
    for n,i in enumerate(indices):
        s=ranked[i];raw=read_gray(frames[s['image_id']][s['view']])
        x,y=int(round(s['source_u_px'])),int(round(s['source_v_px']))
        crop=raw[max(0,y-40):min(raw.shape[0],y+41),max(0,x-40):min(raw.shape[1],x+41)]
        patch=cv2.cvtColor(cv2.resize(crop,(320,320),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
        ox,oy=(n%3)*340,(n//3)*380
        canvas[oy+55:oy+375,ox+10:ox+330]=patch
        label=s['view']+' p'+s['point_id']+' width=%.2f'%s['edge_width_p75_px']
        cv2.putText(canvas,label,(ox+10,oy+20),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(canvas,s['blur_decision'],(ox+10,oy+42),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,220) if s['blur_decision']=='blurred' else (0,130,0),1,cv2.LINE_AA)
        selected.append(dict(panel=n+1,**s))
    save_image(out/'blur_examples.png',canvas);write_json(out/'blur_examples.json',selected)


def filter_dataset(source, output, max_width=8.):
    from bbs_calib.prism_workflow import find_grid
    keep_decision({'edge_width_p75_px':0},max_width)
    source=Path(source).resolve();c=load_config(source/'config.json')
    if c['images'].get('mode')!='paired_crops':
        raise ValueError('filter-blur requires rectangular paired-crop extraction')
    profile=c.get('full_grid_extraction',{}).get('dot_profile','legacy')
    if profile=='small':
        raise ValueError('Small-dot blur filtering is not validated: the +/-10 px profile crosses the opposite edge. No data filtered; use review/audit-points.')
    records=read_csv(source/'observations.csv')
    groups={};seen=set()
    for row in records:
        k=(row['image_id'],row['view'],row['point_id'])
        if k in seen:raise ValueError('Duplicate point: '+str(k))
        seen.add(k)
        if row.get('accepted','').lower() in ('true','1') and row.get('review_status')=='confirmed':
            groups.setdefault(k[:2],[]).append(row)
    if not groups:raise ValueError('No accepted confirmed observations')
    frames={f['image_id']:f for f in discover_frames(c)}
    if any(fid not in frames or view not in ('left','right') for fid,view in groups):
        raise ValueError('Missing original image or invalid view')
    tracked={}
    def track(p):tracked[str(p)]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name in ('config.json','observations.csv','audit.json','cli_input.json'):
        if (source/name).is_file():track(source/name)
    out=new_output(Path(output));scores=[];lookup={};audit=[]
    for fid in sorted({k[0] for k in groups}):
        previews=[]
        for view in ('left','right'):
            path=frames[fid][view];track(path);gray=read_gray(path)
            if gray.dtype!=np.uint8:raise ValueError('Only uint8 full-grid images supported')
            if profile=='standard':
                from .standard_grid import find_standard_grid
                blobs,selected,_=find_standard_grid(gray,c['board']['rows'],c['board']['cols'])
            else:
                blobs,selected,_=find_grid(gray,rows=c['board']['rows'],cols=c['board']['cols'])
            preview=cv2.cvtColor(cv2.resize(gray,(750,1200)),cv2.COLOR_GRAY2BGR)
            for row in groups.get((fid,view),[]):
                pid=int(row['point_id']);p=np.array([float(row['source_u_px']),float(row['source_v_px'])])
                s=dict(image_id=fid,view=view,point_id=str(pid),source_u_px=p[0],source_v_px=p[1])
                try:
                    if selected is None or not 0<=pid<len(selected):raise ValueError('Cannot reproduce full-grid ordering')
                    s.update(edge_width(gray,blobs[selected[pid]][1],p))
                    s['blur_decision']=keep_decision(s,max_width)
                except (ValueError,cv2.error) as exc:
                    # Do not silently remove an unmeasurable point as "blurred".
                    s.update(blur_decision='unmeasured_kept_for_review',reason=str(exc))
                scores.append(s);lookup[(fid,view,str(pid))]=s
                color=(0,0,255) if s['blur_decision']=='blurred' else ((0,180,0) if s['blur_decision']=='kept' else (0,180,255))
                draw_center(preview,resize_points(p,gray.shape,preview.shape),color=color,radius=3)
            cv2.putText(preview,view+'  RED: removed  GREEN: kept  ORANGE: review',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            previews.append(preview)
        sets=[{s['point_id'] for s in scores if s['image_id']==fid and s['view']==v and s['blur_decision']!='blurred'} for v in ('left','right')]
        matched=len(sets[0]&sets[1]);minimum=c.get('detection',{}).get('min_points_per_view',6)
        audit.append(dict(image_id=fid,left_kept=len(sets[0]),right_kept=len(sets[1]),matched_good_points=matched,
                          retained_pair=matched>=minimum,below_original_60_point_gate=matched<60))
        save_image(out/'review'/(fid+'_blur_review.png'),np.hstack(previews))
        print('Blur filter: %s left=%d right=%d matched=%d'%(fid,len(sets[0]),len(sets[1]),matched),flush=True)
    filtered=[]
    for original in records:
        row=dict(original);s=lookup.get((row['image_id'],row['view'],row['point_id']))
        if s:
            row.update({k:v for k,v in s.items() if k not in ('image_id','view','point_id','source_u_px','source_v_px')})
            if s['blur_decision']=='blurred':row.update(accepted=False,review_status='blur_removed',reject_reason='edge_width_p75_above_threshold')
        filtered.append(row)
    # Preserve all rejected rows for recovery; the existing solver skips accepted=False.
    write_csv(out/'observations.csv',filtered);write_csv(out/'blur_scores.csv',scores)
    write_csv(out/'removed_points.csv',[s for s in scores if s['blur_decision']=='blurred'])
    write_json(out/'audit.json',audit)
    if (source/'audit.json').exists():write_json(out/'source_audit.json',read_json(source/'audit.json'))
    c['output']=str(out/'calibration');c.pop('_config_path',None);write_json(out/'config.json',c)
    meta=read_json(source/'cli_input.json') if (source/'cli_input.json').exists() else {'workflow':'prism'}
    meta.update(blur_filter_source=str(source),blur_filter_threshold_px=max_width);write_json(out/'cli_input.json',meta)
    examples(scores,frames,out)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in tracked.items())
    counts=dict(Counter(s['blur_decision'] for s in scores))
    summary=dict(source=str(source),threshold_px=max_width,metric='75th percentile of 96 normal-profile 10-90% edge widths',
                 threshold_status='User-adjustable heuristic, not a calibrated depth-of-field limit',
                 counts=counts,attempted_observations=len(scores),
                 measured_observations=sum('edge_width_p75_px' in s for s in scores),source_rows=len(records),
                 active_observations=sum(str(r.get('accepted','')).lower() in ('true','1') and r.get('review_status')=='confirmed' for r in filtered),
                 removed_by_view=dict(Counter(s['view'] for s in scores if s['blur_decision']=='blurred')),
                 minimum_matched_points=min(a['matched_good_points'] for a in audit),
                 frames_below_60=[a['image_id'] for a in audit if a['below_original_60_point_gate']],
                 point_count_ready=all(a['retained_pair'] for a in audit),
                 quality_review_required=bool(counts.get('unmeasured_kept_for_review',0)),
                 fit_ready=all(a['retained_pair'] for a in audit) and not counts.get('unmeasured_kept_for_review',0),
                 inputs_unchanged=unchanged,input_sha256=tracked,
                 residuals_used=False,calibration_refit=False,clear_counterpart_removed=False)
    write_json(out/'blur_summary.json',summary)
    (out/'README.md').write_text('# Blurred-circle screening\n\n'
        'Screening uses the 10%%–90%% edge-transition width in the original grayscale images, independently of reprojection error. For each point, 96 edge normals are sampled and at least 72 must be valid. A point is marked as blurred when the 75th percentile of the widths exceeds %.2f px. This is an empirical screening threshold, not a measured depth of field.\n\n'%max_width+
        'Classification counts: '+str(counts)+'. Minimum matched points across the two views: '+str(summary['minimum_matched_points'])+'.\n\n'
        'observations.csv retains every record. Blurred observations have accepted=False and are skipped by the existing fit; the corresponding clear observation in the other view is retained. Points whose sharpness could not be measured reliably are retained and marked orange; they have not passed a sharpness check.\n\n'
        'In review, red indicates removal, green indicates retention, and orange requires manual inspection. Check point IDs, edges, and calibration-grid coverage. removed_points.csv lists all removed observations.\n\n'
        'The original images, detected center coordinates, and previous calibration results are unchanged. This command does not recalibrate, so it cannot establish a lower RMS. After refitting, do not treat an RMS comparison across different point sets as evidence of improved accuracy.\n',encoding='utf-8')
    if not unchanged:raise ValueError('Inputs changed during blur filtering')
    return out,summary
