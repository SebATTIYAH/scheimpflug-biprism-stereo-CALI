"""Reproducible real-data extraction and joint-model calibration driver."""
from pathlib import Path
import sys, argparse
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .templates import template_config
from bbs_calib.io import read_gray,save_image,write_json,read_json,write_csv,read_csv,new_output
from bbs_calib.image_inputs import discover_frames
from bbs_calib.detection import refine_ellipse
from bbs_calib.review_images import resize_points,draw_center
from bbs_calib.config import load_config
from bbs_calib.optimization import observations,calibrate
from bbs_calib.reporting import save_results

def validate_grid(rows,cols):
    if not all(isinstance(x,int) and not isinstance(x,bool) and x>=2 for x in (rows,cols)) or rows*cols<6:
        raise ValueError("Grid requires integer rows/cols >=2 and at least 6 dots")


def find_grid(gray,max_grid_error=3,min_area_px=150,rows=12,cols=9):
    validate_grid(rows,cols)
    count=rows*cols
    attempts=[]
    for block,C in [(101,10),(151,8),(81,5),(151,15)]:
        binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,C)
        contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        blobs=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if not min_area_px<area<15000 or len(contour)<12:continue
            (x,y),(a,b),angle=cv2.fitEllipse(contour)
            if min(a,b)/max(a,b)<.4 or max(a,b)>100:continue
            if 4*np.pi*area/cv2.arcLength(contour,True)**2<.5:continue
            if any(np.linalg.norm(np.array([x,y])-v[0])<8 for v in blobs):continue
            blobs.append((np.array([x,y]),contour))
        attempts.append(dict(block=block,C=C,candidates=len(blobs)))
        if not count<=len(blobs)<=max(2*count+4,30):continue
        median=np.median([cv2.contourArea(v[1]) for v in blobs])
        blobs=[v for v in blobs if .6*median<cv2.contourArea(v[1])<1.65*median]
        pts=np.array([v[0] for v in blobs],np.float32)
        attempts[-1]['size_filtered']=len(blobs)
        ok,ordered=cv2.findCirclesGrid((pts*.25).reshape(-1,1,2),(cols,rows),
                    flags=cv2.CALIB_CB_SYMMETRIC_GRID|cv2.CALIB_CB_CLUSTERING,
                    blobDetector=None,parameters=cv2.CirclesGridFinderParameters())
        if not ok:continue
        grid=ordered.reshape(rows,cols,2)/.25
        if np.mean(grid[-1,:,1])<np.mean(grid[0,:,1]):grid=grid[::-1]
        if np.mean(grid[:,-1,0])<np.mean(grid[:,0,0]):grid=grid[:,::-1]
        selected=[int(np.argmin(np.linalg.norm(pts-p,axis=1))) for p in grid.reshape(-1,2)]
        lattice=np.array([[j,i] for i in range(rows) for j in range(cols)],np.float32)
        H,inliers=cv2.estimateAffine2D(lattice,pts[selected],method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=5000)
        if H is None:continue
        design=np.column_stack([lattice,np.ones(count)])
        for _ in range(3):
            predicted=design@H.T
            cost=np.linalg.norm(predicted[:,None,:]-pts[None,:,:],axis=2)
            a,b=linear_sum_assignment(cost)
            if len(a)!=count:break
            selected=b.tolist()
            H=np.linalg.lstsq(design,pts[selected],rcond=None)[0].T
        errors=np.linalg.norm(design@H.T-pts[selected],axis=1)
        attempts[-1]['max_grid_affine_error_px']=float(errors.max())
        if len(set(selected))!=count or errors.max()>max_grid_error:continue
        return blobs,selected,attempts
    return None,None,attempts

def pair_complete(pair,rows=12,cols=9):
    count=rows*cols
    expected={str(i) for i in range(count)}
    return all(len([r for r in pair if r['view']==v])==count and
               {r['point_id'] for r in pair if r['view']==v}==expected for v in ('left','right'))


def refinement_settings(contour,dot_profile):
    if dot_profile=='standard':
        (_, _),(wa,wb),_=cv2.fitEllipseAMS(np.asarray(contour,np.float32))
        radius=min(wa,wb)/2
        # Stay near the OUTER edge; a wide search on a small annulus can jump
        # to its inner rim. Quality gates below remain identical to legacy.
        return dict(edge_search_px=min(10.,.3*radius),boundary_samples=96,profile_step_px=.2,
                    min_profile_contrast=.05,profile_smoothing_px=min(1.5,.08*radius))
    if dot_profile=='legacy':
        return dict(edge_search_px=10,boundary_samples=96,profile_step_px=.2,
                    min_profile_contrast=.05,profile_smoothing_px=1.5)
    (_, _),(wa,wb),_=cv2.fitEllipseAMS(np.asarray(contour,np.float32))
    radius=min(wa,wb)/2
    return dict(edge_search_px=min(2.,.45*radius),boundary_samples=96,profile_step_px=.1,
                min_profile_contrast=.05,profile_smoothing_px=.4)


def extract(out,limit,directory=None,pitch_mm=3,pattern='*.tif',dot_profile='legacy',require_complete_pair=False,rows=12,cols=9,config=None,left_suffix='_0',right_suffix='_1'):
    validate_grid(rows,cols)
    count=rows*cols
    minimum=min(60,max(6,int(np.ceil(count*.55))))
    if directory is None:raise ValueError('An explicit image directory is required')
    if dot_profile not in ('legacy','small','standard'):raise ValueError('Unknown dot profile')
    custom=directory is not None
    groups=[('custom',Path(directory))]
    for group,directory in groups:
        frames=discover_frames(dict(images=dict(mode='paired_crops',directory=str(directory),pattern=pattern,suffixes=dict(left=left_suffix,right=right_suffix))))
        dest=new_output(out if custom else out/group);records=[];audits=[];sensor_shape=None
        for i,frame in enumerate(frames[:limit] if limit else frames):
            left=read_gray(frame['left']);right=read_gray(frame['right'])
            hl,wl=left.shape;hr,wr=right.shape
            if hl!=hr or wr not in (wl,wl+1):raise ValueError('Invalid MATLAB midpoint crop dimensions')
            if sensor_shape is not None and sensor_shape!=(wl+wr,hl):raise ValueError('Sensor dimensions changed between frames')
            sensor_shape=(wl+wr,hl)
            pair=[];imgs=[];audit=dict(image_id=frame['image_id'],views={})
            for view in ('left','right'):
                gray=left if view=='left' else right
                if gray.dtype!=np.uint8:raise ValueError('Full-grid extractor currently requires uint8 images')
                if dot_profile=='standard':
                    from .standard_grid import find_standard_grid
                    blobs,selected,attempts=find_standard_grid(gray,rows,cols)
                else:
                    blobs,selected,attempts=find_grid(gray,min_area_px=8 if dot_profile=='small' else 150,rows=rows,cols=cols)
                va=dict(attempts=attempts,grid_found=selected is not None);audit['views'][view]=va
                preview=cv2.cvtColor(cv2.resize(gray,(375,600)),cv2.COLOR_GRAY2BGR)
                if selected is not None:
                    errors=[];rejected=[]
                    for pid,index in enumerate(selected):
                        initial,contour=blobs[index];x,y,w,h=cv2.boundingRect(contour)
                        pad=22;x0=max(0,x-pad);y0=max(0,y-pad)
                        patch=gray[y0:min(gray.shape[0],y+h+pad),x0:min(gray.shape[1],x+w+pad)]
                        try:
                            settings=refinement_settings(contour,dot_profile)
                            if dot_profile in ('small','standard'):
                                margin=settings['edge_search_px']+1
                                if x<margin or y<margin or x+w>gray.shape[1]-margin or y+h>gray.shape[0]-margin:
                                    raise ValueError('circle/edge search touches crop boundary')
                            f=refine_ellipse(patch,contour-[x0,y0],settings)
                            center=np.array(f['center'])+[x0,y0]
                            if f['fit_error']>(.35 if dot_profile=='small' else .8) or f['angular_coverage']<.8 or not f['fit_converged']:
                                raise ValueError('edge quality: rms=%.3f coverage=%.3f'%(f['fit_error'],f['angular_coverage']))
                            if dot_profile=='small' and np.linalg.norm(center-initial)>1.:
                                raise ValueError('subpixel refinement shift exceeds 1 px')
                            errors.append(f['fit_error'])
                            pair.append(dict(image_id=frame['image_id'],view=view,point_id=str(pid),
                                u_px=center[0]+(wl if view=='right' else 0),v_px=center[1],
                                source_u_px=center[0],source_v_px=center[1],coordinate_space='full_frame',
                                accepted=True,review_status='confirmed',fit_error_px=f['fit_error'],
                                correspondence_method='complete grid; image-top row and image-left column; paired geometry audit'))
                            xy=resize_points(center,gray.shape,preview.shape)
                            draw_center(preview,xy,label=pid if pid in (0,cols-1,(rows-1)*cols,count-1) else None)
                        except (ValueError,cv2.error) as exc:rejected.append(dict(point_id=pid,reason=str(exc)))
                    va.update(refined=len(errors),median_fit_error_px=np.median(errors) if errors else None,rejected=rejected)
                imgs.append(preview)
            matched=set(v['point_id'] for v in pair if v['view']=='left') & set(v['point_id'] for v in pair if v['view']=='right')
            good=all(v['grid_found'] for v in audit['views'].values()) and len(matched)>=minimum
            if require_complete_pair:good=good and pair_complete(pair,rows,cols)
            audit['matched_good_points']=len(matched)
            audit['retained_pair']=good
            audit['require_complete_pair']=require_complete_pair
            audit['dot_profile']=dot_profile
            audit['discard_reasons']=([] if good else [view+(':grid_incomplete_or_not_found' if not va['grid_found'] else ':subpixel_points_incomplete')
                for view,va in audit['views'].items() if not va['grid_found'] or (require_complete_pair and va.get('refined',0)!=count)])
            if good:records.extend(pair)
            audits.append(audit)
            save_image(dest/(frame['image_id']+'_review.png'),np.hstack(imgs))
            # Checkpoints keep long sequences recoverable without rewriting all rows per frame.
            if (i+1)%25==0:write_json(dest/'audit.json',audits);write_csv(dest/'observations.csv',records)
            print(group,i+1,[(v.get('grid_found'),v.get('refined')) for v in audit['views'].values()],good,flush=True)
        write_json(dest/'audit.json',audits);write_csv(dest/'observations.csv',records)
        write_csv(dest/'discarded_pairs.csv',[dict(image_id=a['image_id'],reason=';'.join(a['discard_reasons']),matched_points=a['matched_good_points']) for a in audits if not a['retained_pair']],['image_id','reason','matched_points'])
        write_json(dest/'extraction_summary.json',dict(total_pairs=len(audits),retained_pairs=sum(a['retained_pair'] for a in audits),
                   discarded_pairs=sum(not a['retained_pair'] for a in audits),observations=len(records),dot_profile=dot_profile,
                   require_complete_pair=require_complete_pair,source_images_deleted=False,calibration_run=False))
        c=load_config(config) if config is not None else template_config('cropped')
        c['images'].update(directory=str(directory),pattern=pattern,suffixes=dict(left=left_suffix,right=right_suffix))
        c['board'].update(rows=rows,cols=cols,pitch_mm=pitch_mm,diameter_mm=None)
        if c['camera'].get('c_px') is None:
            c['camera'].update(c_px=[(sensor_shape[0]-1)/2,(sensor_shape[1]-1)/2],c_source='assumed_full_sensor_geometric_center')
        c.pop('_config_path',None)
        # Unknown hardware is not filled from a previous experiment.
        c['output']=str(dest/'calibration');write_json(dest/'config.json',c)
        c['full_grid_extraction']=dict(rows=rows,cols=cols,dot_profile=dot_profile,require_complete_pair=require_complete_pair,
            candidate_min_area_px=8 if dot_profile=='small' else 150,
            candidate_max_area_px=15000,
            boundary_policy='outer_boundary_solid_and_annular' if dot_profile=='standard' else 'legacy_contour_order',
            note='Dedicated full-grid settings; detection section is not the full-grid extractor configuration')
        write_json(dest/'config.json',c)

def fit(source,out,variant,solver_options=None):
    c=load_config(source/'config.json');obs=observations(source/'observations.csv',c)
    if variant in ('center_plus','center_minus'):
        shift=50 if variant=='center_plus' else -50
        c['camera']['c_px']=(np.array(c['camera']['c_px'])+[shift,shift]).tolist()
        c['camera']['c_source']='sensitivity_geometric_center_diagonal_shift_%d_px'%shift
    if variant=='initial_alternative':
        c['camera'].update(m_px_per_mm=c['camera']['m_px_per_mm']*.9,omega_p_deg=[.5,-.5,.5])
    if solver_options is not None:
        c['optimization'].update(solver_profile=solver_options['profile'],multistart=solver_options['starts'],
            search_seed=solver_options['seed'],refine_nfev=solver_options['refine_nfev'])
        for key in ('backend','workers'):
            if key in solver_options:c['optimization'][key]=solver_options[key]
        from .hardware_refinement import apply_mode
        apply_mode(c,solver_options.get('refine_hardware'))
    c['output']=str(out)
    result=calibrate(c,obs);save_results(result,c,obs,out,'bbs_calib prism fit')
    print(result['status'],result['mechanical_angles'],flush=True)
