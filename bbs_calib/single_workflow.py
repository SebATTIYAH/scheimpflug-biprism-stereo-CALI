"""Single-camera full rectangular dark-dot workflow."""
from pathlib import Path
import sys,argparse
import cv2
import numpy as np
from .prism_workflow import find_grid, validate_grid
from .angles import effective_angles
from bbs_calib.detection import refine_ellipse
from bbs_calib.review_images import resize_points,draw_center
from bbs_calib.io import read_gray,save_image,write_json,read_json,write_csv,new_output
from bbs_calib.single_camera import calibrate_single,calibrate_single_search,validate_single

def extract(root, directory=None, pitch_mm=3, rows=12, cols=9):
    validate_grid(rows,cols)
    count=rows*cols
    minimum=min(60,max(6,int(np.ceil(count*.55))))
    if directory is None:raise ValueError('An explicit image directory is required')
    groups=[('custom',Path(directory))]
    for group,folder in groups:
        dest=new_output(root if directory is not None else root/group)
        frames=[];audit=[];rows=[]
        files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.bmp','.tif','.tiff','.png','.jpg'))
        if not files:raise ValueError('No supported images in '+str(folder))
        if len({p.stem for p in files})!=len(files):raise ValueError('Image stems must be unique')
        image_shape=None
        for index,path in enumerate(files):
            gray=read_gray(path);h,w=gray.shape
            if gray.dtype!=np.uint8:raise ValueError('Full-grid extractor currently requires uint8 images')
            if image_shape is not None and image_shape!=(w,h):raise ValueError('All images must have the same dimensions')
            image_shape=(w,h)
            blobs,selected,attempts=find_grid(gray,max_grid_error=6,rows=rows,cols=cols);points=[];pixels=[];ids=[];reject=[]
            preview=cv2.cvtColor(cv2.resize(gray,(750,600)),cv2.COLOR_GRAY2BGR)
            if selected is not None:
                for pid,bi in enumerate(selected):
                    _,contour=blobs[bi];x,y,cw,ch=cv2.boundingRect(contour);x0=max(0,x-22);y0=max(0,y-22)
                    try:
                        f=refine_ellipse(gray[y0:min(h,y+ch+22),x0:min(w,x+cw+22)],contour-[x0,y0],
                            dict(edge_search_px=10,boundary_samples=96,profile_step_px=.2,min_profile_contrast=.05,profile_smoothing_px=1.5))
                        if f['fit_error']>.8 or f['angular_coverage']<.8 or not f['fit_converged']:raise ValueError('edge_quality')
                        p=np.array(f['center'])+[x0,y0];points.append([pitch_mm*(pid%cols),pitch_mm*(pid//cols),0]);pixels.append(p);ids.append(pid)
                        rows.append(dict(image_id=path.stem,point_id=pid,u_px=p[0],v_px=p[1],fit_error_px=f['fit_error']))
                        xy=resize_points(p,gray.shape,preview.shape)
                        draw_center(preview,xy,label=pid if pid in (0,cols-1,(rows-1)*cols,count-1) else None)
                    except (ValueError,cv2.error) as exc:reject.append(dict(point_id=pid,reason=str(exc)))
            if len(ids)>=10:
                A,inliers=cv2.estimateAffine2D(np.ascontiguousarray(np.array(points)[:,:2],dtype=np.float32),
                    np.ascontiguousarray(pixels,dtype=np.float32),method=cv2.RANSAC,ransacReprojThreshold=1.5)
                if A is not None:
                    e=np.linalg.norm(np.column_stack([np.array(points)[:,:2],np.ones(len(ids))])@A.T-pixels,axis=1)
                    keep=np.flatnonzero(e<=2)
                    reject += [dict(point_id=ids[i],reason='refined_grid_residual_above_2px') for i in np.flatnonzero(e>2)]
                    points=[points[i] for i in keep];pixels=[pixels[i] for i in keep];ids=[ids[i] for i in keep]
            retained=len(ids)>=minimum
            if retained:frames.append(dict(id=path.stem,P=points,p=pixels,ids=ids))
            for row in rows:
                if row['image_id']==path.stem:row['used_in_frames_json']=retained and row['point_id'] in ids
            audit.append(dict(file=str(path),shape=[w,h],attempts=attempts,retained=retained,points=len(ids),rejected=reject))
            save_image(dest/('review_%02d.png'%index),preview)
            write_json(dest/'frames.json',frames);write_json(dest/'audit.json',audit);write_csv(dest/'centers.csv',rows)
            print(group,index+1,'/',len(files),'points',len(ids),'retained',retained,flush=True)

def fit(source,out,variant,solver_options=None):
    frames=read_json(source/'frames.json')
    for f in frames:f['P']=np.array(f['P']);f['p']=np.array(f['p'])
    if len(frames)<8:raise ValueError('Need at least 8 usable frames for training and validation')
    rng=np.random.default_rng(271828);held=set(rng.choice(len(frames),max(2,round(len(frames)*.25)),replace=False).tolist())
    train=[f for i,f in enumerate(frames) if i not in held];test=[f for i,f in enumerate(frames) if i in held]
    shape=read_json(source/'audit.json')[0]['shape']
    if solver_options is None:
        r=calibrate_single(train,shape,release_center=variant!='fixed_center',zero_tilt=variant=='zero_tilt')
    else:
        r=calibrate_single_search(train,shape,solver_options,release_center=variant!='fixed_center',zero_tilt=variant=='zero_tilt')
    r['validation']=validate_single(r,test);r['training_frames']=[f['id'] for f in train];r['heldout_frames']=[f['id'] for f in test]
    r['angles']=effective_angles(r['K']);r['source_directory']=str(source);r['variant']=variant
    r['model']='Hu 2022 Eqs.13-29, single camera without biprism; three radial coefficients'
    write_json(out/'calibration.json',r)
    if r.get('optimizer_search'):
        from bbs_calib.solver_search import save_report
        save_report(r['optimizer_search'],out)
    write_csv(out/'residuals.csv',[dict(du_px=e[0],dv_px=e[1]) for e in r['residuals']])
    print('RESULT',source.name,variant,r['angles'],'train',r['stages'][-1]['rms_px'],'heldout',r['validation']['rms_px'],
          'rank',r['global_diagnostics'].get('rank'),'/',r['global_diagnostics'].get('columns'),'bounds',r['at_bounds'],flush=True)
