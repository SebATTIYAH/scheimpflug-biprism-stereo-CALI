from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.optimize import least_squares
from .io import display_u8, read_gray, save_image, write_json, write_csv
from .review_images import resize_points,draw_center


def ellipse_distance(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Signed Sampson approximation in pixels (not exact Euclidean distance)."""
    cx, cy, la, lb, theta = params
    a, b = np.exp(la), np.exp(lb)
    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = (points-np.array([cx, cy])).T
    x, y = ct*dx+st*dy, -st*dx+ct*dy
    F = x*x/a**2+y*y/b**2-1
    return F/np.maximum(2*np.sqrt(x*x/a**4+y*y/b**4), 1e-9)


def refine_ellipse(gray: np.ndarray, contour: np.ndarray, settings: dict) -> dict:
    method = getattr(cv2, settings.get('ellipse_method', 'fitEllipseAMS'), cv2.fitEllipse)
    (cx, cy), (wa, wb), angle = method(np.asarray(contour, np.float32).reshape(-1,1,2))
    a, b, theta = wa/2, wb/2, np.radians(angle)
    if min(a,b) < 1 or max(a,b) > max(gray.shape):
        raise ValueError('Unreasonable initial ellipse')
    init = np.array([cx, cy, np.log(a), np.log(b), theta])
    n = settings.get('boundary_samples', 96)
    angles = np.arange(n)*2*np.pi/n
    rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    perimeter = np.column_stack([a*np.cos(angles), b*np.sin(angles)]) @ rotation.T+[cx,cy]
    normals = np.column_stack([np.cos(angles)/a, np.sin(angles)/b]) @ rotation.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    search = settings.get('edge_search_px', 3.)
    step = settings.get('profile_step_px', .2)
    t = np.arange(-search, search+step/2, step)
    locations = perimeter[:, None, :]+normals[:, None, :]*t[None, :, None]
    profiles = map_coordinates(gray.astype(float), [locations[:,:,1],locations[:,:,0]], order=1, mode='nearest')
    # Morphology/thresholds are NOT used here: sample original sensor values.
    profile_sigma=settings.get('profile_smoothing_px',.7*step)/step
    gradient = np.abs(np.gradient(gaussian_filter1d(profiles, profile_sigma, axis=1), step, axis=1))
    peak = np.argmax(gradient, axis=1)
    contrast = np.abs(np.mean(profiles[:,-4:],axis=1)-np.mean(profiles[:,:4],axis=1))
    local_range = max(float(np.percentile(gray,99)-np.percentile(gray,1)), 1.)
    usable = (peak > 1)&(peak < len(t)-2)&(contrast/local_range > settings.get('min_profile_contrast', .15))
    refined, sample_angles = [], []
    for i in np.flatnonzero(usable):
        j = peak[i]
        g1,g2,g3 = gradient[i,j-1:j+2]
        denominator = g1-2*g2+g3
        offset = .5*(g1-g3)/denominator if abs(denominator)>1e-14 else 0.
        if abs(offset)>1:
            continue
        refined.append(perimeter[i]+normals[i]*(t[j]+step*offset))
        sample_angles.append(angles[i])
    points = np.asarray(refined)
    coverage = len(set((np.array(sample_angles)/(2*np.pi)*24).astype(int)))/24
    if len(points) < 12:
        raise ValueError('Too few valid original-grayscale subpixel edges')
    fit = least_squares(lambda x: ellipse_distance(x,points), init,
                        bounds=(init+[-search,-search,-.6,-.6,-np.pi], init+[search,search,.6,.6,np.pi]),
                        loss='soft_l1', f_scale=.2, max_nfev=100)
    e = ellipse_distance(fit.x, points)
    # theta remains recorded but is not a quality gate for near-circular ellipses.
    return dict(initial_center=[cx,cy], center=fit.x[:2].tolist(),
                semi_axes=np.exp(fit.x[2:4]).tolist(), theta_rad=float(fit.x[4]),
                boundary=points.tolist(), contour=np.asarray(contour).reshape(-1,2).tolist(),
                fit_error=float(np.sqrt(np.mean(e*e))), angular_coverage=coverage,
                contrast=float(np.median(contrast)/local_range), fit_converged=bool(fit.success),
                center_covariance=None, covariance_note='Not estimated; fit residual is not center accuracy',
                edge_method='original-grayscale normal profiles; quadratic interpolation of gradient peak; robust Sampson ellipse')


def detect_roi(gray: np.ndarray, roi: list, settings: dict, masks: list = None) -> list:
    x0,y0,w,h = [int(v) for v in roi]
    if x0<0 or y0<0 or x0+w>gray.shape[1] or y0+h>gray.shape[0]:
        raise ValueError('ROI outside input image')
    raw = gray[y0:y0+h,x0:x0+w]
    image = display_u8(raw)
    work = image.copy()
    bg = settings.get('background_sigma_px', 0)
    if bg:
        work = np.clip(work.astype(float)-cv2.GaussianBlur(work.astype(float),(0,0),bg)+128,0,255).astype(np.uint8)
    dark = settings.get('polarity','dark') == 'dark'
    flag = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
    threshold = settings.get('threshold')
    if threshold is None:
        _,binary = cv2.threshold(work,0,255,flag|cv2.THRESH_OTSU)
    else:
        _,binary = cv2.threshold(work,threshold,255,flag)
    kernel = int(settings.get('candidate_open_px',0))
    if kernel>1:
        binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((kernel,kernel),np.uint8))
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    candidates = []
    for idx,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # Retain a lightweight record even for unusable candidates.
        bx,by,bw,bh = cv2.boundingRect(contour)
        item = dict(candidate_id=idx, accepted=False, reject_reason='', contour=contour.reshape(-1,2).tolist(),
                    area=area, center=[bx+bw/2,by+bh/2], initial_center=[bx+bw/2,by+bh/2],
                    boundary=[], fit_error=None, angular_coverage=0.,contrast=0., quality=0.)
        reasons=[]
        if not settings.get('min_area_px',20)<=area<=settings.get('max_area_px',10000):
            reasons.append('area')
        margin = settings.get('edge_search_px',3.)+1
        if bx<margin or by<margin or bx+bw>w-margin or by+bh>h-margin:
            reasons.append('roi_boundary')
        for mx,my,mw,mh in masks or []:
            if bx+x0<mx+mw and bx+x0+bw>mx and by+y0<my+mh and by+y0+bh>my:
                reasons.append('ridge_mask')
        if len(contour)<5:
            reasons.append('short_contour')
        if not reasons:
            try:
                item.update(refine_ellipse(raw,contour,settings))
                a,b=item['semi_axes']
                if min(a,b)/max(a,b)<settings.get('min_axis_ratio',.25):
                    reasons.append('axis_ratio')
                if min(a,b)<settings.get('min_semiaxis_px',2) or max(a,b)>settings.get('max_semiaxis_px',100):
                    reasons.append('axis_length')
                if item['fit_error']>settings.get('max_fit_error_px',.6):
                    reasons.append('fit_residual')
                if item['angular_coverage']<settings.get('min_coverage',.8):
                    reasons.append('incomplete_edge')
                if item['contrast']<settings.get('min_contrast',.15):
                    reasons.append('contrast')
                if not item['fit_converged']:
                    reasons.append('fit_not_converged')
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                reasons.append('refine_failed:'+str(exc))
        for key in ('center','initial_center'):
            item[key]=(np.array(item[key])+[x0,y0]).tolist()
        for key in ('boundary','contour'):
            if len(item[key]):
                item[key]=(np.array(item[key])+[x0,y0]).tolist()
        item['accepted']=not reasons
        item['reject_reason']=';'.join(reasons)
        item['quality']=item['angular_coverage']/(1+(item['fit_error'] or 0.)) if not reasons else 0.
        candidates.append(item)
    return candidates


def detect_dataset(c: dict, output: Path) -> dict:
    from .config import validate_config
    from .grid_ordering import order_grid
    from .io import read_csv
    from .image_inputs import input_mode,discover_frames,crop_origins,origin_for
    validate_config(c,'detect')
    frames=discover_frames(c)
    mode=input_mode(c);origins=crop_origins(c)
    anchors=read_csv(Path(c['review']['anchors_csv'])) if c.get('review',{}).get('anchors_csv') else []
    records, audit=[],[]
    fields=['image_id','view','point_id','u_px','v_px','accepted','review_status','fit_error','quality',
            'source_image','candidate_id','initial_u_px','initial_v_px','reject_reason',
            'coordinate_space','source_u_px','source_v_px','crop_origin_x_px','crop_origin_y_px']
    for frame in frames:
        image_id=frame['image_id']
        shared_overlay=None
        for view in ('left','right'):
            path=frame[view];gray=read_gray(path)
            if mode=='full_frame' and shared_overlay is not None:
                overlay=shared_overlay
            else:
                overlay=cv2.cvtColor(display_u8(gray),cv2.COLOR_GRAY2BGR)
            roi=c['views'][view].get('roi') or [0,0,gray.shape[1],gray.shape[0]]
            origin=origin_for(c,image_id,view,origins)
            offset=np.zeros(2) if origin is None else origin
            masks=list(c.get('ridge_masks',[]))
            if mode=='paired_crops':
                masks=list(c['views'][view].get('masks',[]))
                if c.get('ridge_masks'):
                    if origin is None:
                        raise ValueError('Global ridge_masks need known crop origins; use views.<view>.masks for crop-local masks')
                    masks += [[x-offset[0],y-offset[1],w,h] for x,y,w,h in c['ridge_masks']]
            candidates=detect_roi(gray,roi,c.get('detection',{}),masks)
            good=[v for v in candidates if v['accepted']]
            anch=[dict(r) for r in anchors if r['image_id']==image_id and r['view']==view]
            anchor_space=c.get('review',{}).get('anchor_coordinates','crop_local' if mode=='paired_crops' else 'full_frame')
            if anchor_space not in ('full_frame','crop_local'):
                raise ValueError('review.anchor_coordinates must be full_frame or crop_local')
            if mode=='paired_crops' and anchor_space=='full_frame' and anch:
                if origin is None:
                    raise ValueError('Full-frame anchors require crop origins')
                for a in anch:
                    a['u_px']=float(a['u_px'])-offset[0];a['v_px']=float(a['v_px'])-offset[1]
            if c['board'].get('points_csv'):
                ordering=dict(status='needs_review',mapping={},reason='Explicit board CSV requires manual point IDs',candidates=[])
            else:
                ordering=order_grid(np.array([v['center'] for v in good]).reshape(-1,2),
                                    c['board']['rows'],c['board']['cols'],anch,
                                    c.get('review',{}).get('anchor_tolerance_px',5.))
            coordinate_space='full_frame' if origin is not None else 'crop_local'
            audit.append(dict(image_id=image_id,view=view,ordering=ordering,source_image=str(path),
                              accepted_candidates=len(good),rejected_candidates=len(candidates)-len(good),
                              input_mode=mode,coordinate_space=coordinate_space,
                              source_to_sensor=None if origin is None else [[1,0,offset[0]],[0,1,offset[1]],[0,0,1]],
                              source_to_detection_roi=[[1,0,-roi[0]],[0,1,-roi[1]],[0,0,1]],
                              warning='Crop origin missing: detection/review allowed, physical calibration blocked' if origin is None else None,
                              display_transform='identity; zoom crops saved separately'))
            lookup={v['candidate_id']:i for i,v in enumerate(good)}
            for item in candidates:
                good_index=lookup.get(item['candidate_id'],-1)
                pid=ordering['mapping'].get(good_index,'')
                status=ordering['status'] if item['accepted'] else 'rejected'
                u,v=item['center']
                records.append(dict(image_id=image_id,view=view,point_id=pid,u_px=u+offset[0],v_px=v+offset[1],
                                    accepted=item['accepted'],review_status=status,fit_error=item['fit_error'],quality=item['quality'],
                                    source_image=str(path),candidate_id=item['candidate_id'],
                                    initial_u_px=item['initial_center'][0]+offset[0],initial_v_px=item['initial_center'][1]+offset[1],
                                    reject_reason=item['reject_reason'],coordinate_space=coordinate_space,
                                    source_u_px=u,source_v_px=v,
                                    crop_origin_x_px='' if origin is None else offset[0],
                                    crop_origin_y_px='' if origin is None else offset[1]))
                item['source_center']=[u,v]
                item['sensor_center']=None if origin is None else [u+offset[0],v+offset[1]]
                item['boundary_coordinate_space']='source_image'
                item['crop_origin_px']=None if origin is None else origin.tolist()
                color=(0,200,0) if status=='confirmed' else (0,100,255)
                draw_center(overlay,[u,v],color=color,radius=3)
                cv2.putText(overlay,str(pid) if pid!='' else '?'+str(item['candidate_id']),
                            (round(u)+3,round(v)-3),cv2.FONT_HERSHEY_SIMPLEX,.3,color,1)
                if item['boundary']:
                    boundary=np.array(item['boundary'])
                    ix,iy=np.floor(boundary.min(axis=0)-4).astype(int)
                    ax,ay=np.ceil(boundary.max(axis=0)+4).astype(int)
                    ix,iy=max(ix,0),max(iy,0)
                    zoom=cv2.cvtColor(display_u8(gray)[iy:ay+1,ix:ax+1],cv2.COLOR_GRAY2BGR)
                    zoom_source_shape=zoom.shape
                    zoom=cv2.resize(zoom,None,fx=5,fy=5,interpolation=cv2.INTER_NEAREST)
                    for p in boundary:
                        q=resize_points(p-[ix,iy],zoom_source_shape,zoom.shape)
                        cv2.circle(zoom,tuple(np.rint(q*256).astype(int)),256,(0,0,255),-1,cv2.LINE_AA,8)
                    save_image(output/'edges'/(image_id+'_'+view+'_'+str(item['candidate_id'])+'.png'),zoom)
            write_json(output/'candidates'/(image_id+'_'+view+'.json'),candidates)
            if mode=='paired_crops':
                save_image(output/'overlays'/(image_id+'_'+view+'.png'),overlay)
            else:
                shared_overlay=overlay
        if mode=='full_frame':
            save_image(output/'overlays'/(image_id+'.png'),shared_overlay)
    write_csv(output/'observations.csv',records,fields)
    write_json(output/'detection_audit.json',audit)
    write_json(output/'config_snapshot.json',c)
    return dict(images=len(frames),input_files=len(frames)*(2 if mode=='paired_crops' else 1),
                observations=len(records),confirmed=sum(r['review_status']=='confirmed' for r in records),
                needs_review=sum(r['review_status']=='needs_review' for r in records),audit=audit)
