from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from .config import board_points,initial_state
from .geometry import scheimpflug,optical_system,path_validity
from .projection import projected_circle,object_points,project
from .io import read_json,write_json,write_csv,save_image


def fixture_config()->dict:
    from .templates import template_config
    return template_config('synthetic')


def true_scene(c:dict)->tuple:
    state=initial_state(c)
    spec=c.get('synthetic',{})
    state['K']=spec.get('true_m_px_per_mm',-100)*scheimpflug(*np.radians(spec.get('true_tilt_deg',[8,5])))
    state['omega']=np.radians(spec.get('true_omega_deg',[.7,-.5,.8]))
    state['kappa']=np.array(spec.get('true_kappa',[.004,-.001]))
    _,P=board_points(c)
    poses=[]
    # Multiple tilt directions; all object points remain beyond the front faces.
    angles=[[-8,-10,4],[10,-7,-5],[-12,9,8],[6,13,-9],[-4,-15,12],[14,3,-12],[5,8,3],[-9,-5,-7]]
    for i,a in enumerate(angles[:spec.get('frames',8)]):
        R=Rotation.from_euler('xyz',a,degrees=True)
        center=np.array([.05*np.sin(i),.08*np.cos(i),20.+.3*np.sin(i*.8)])
        t=center-R.apply(P.mean(axis=0))
        poses.append(np.r_[R.as_rotvec(),t])
    return state,np.asarray(poses)


def render_frame(c:dict,state:dict,pose:np.ndarray,rng:np.random.Generator,occlude:bool=False)->tuple:
    ids,P=board_points(c); s=c.get('synthetic',{})
    width,height=s.get('width',1000),s.get('height',500)
    ss=s.get('supersampling',4)
    # Render geometry on a supersampled pixel lattice centered on full-frame integer pixel centers.
    canvas=np.full((height*ss,width*ss),210,np.uint8)
    truth=[]
    system=optical_system(state['prism'],state['omega'])
    for view in ('left','right'):
        branch=c['views'][view]['branch']
        X=object_points(P,pose)
        valid,_,_=path_validity(X,branch,system,state['prism'])
        if not valid.all():
            raise ValueError('Synthetic fixture contains invalid optical paths')
        uv=project(X,branch,state,system)
        for pid,point,center in zip(ids,P,uv):
            ring=projected_circle(point,c['board']['diameter_mm'],pose,branch,state,160)
            polygon=np.round((ring+.5)*ss-.5).astype(np.int32)
            cv2.fillPoly(canvas,[polygon],40,lineType=cv2.LINE_8)
            truth.append(dict(view=view,point_id=pid,u_px=center[0],v_px=center[1]))
    canvas=cv2.GaussianBlur(canvas.astype(np.float32),(0,0),s.get('blur_sigma_px',.8)*ss)
    image=cv2.resize(canvas,(width,height),interpolation=cv2.INTER_AREA)
    xx,yy=np.meshgrid(np.linspace(-1,1,width),np.linspace(-1,1,height))
    image=image*(1+.08*xx+.035*yy)
    # Blurry ridge band, marked explicitly in configured masks.
    ridge=np.exp(-((np.arange(width)-(width-1)/2)/6.)**2)
    image-=70*ridge[None,:]
    if occlude:
        target=truth[len(truth)//3]
        x,y=round(target['u_px']),round(target['v_px'])
        image[max(y-12,0):y+12,max(x-2,0):x+15]=210
    image+=rng.normal(0,s.get('noise_sigma_gray',.6),image.shape)
    return np.clip(image,0,255).astype(np.uint8),truth


def generate(output:Path)->dict:
    c=fixture_config(); state,poses=true_scene(c)
    rng=np.random.default_rng(c['seed'])
    c['images']['directory']=str((output/'images').resolve())
    c['output']=str((output/'calibration').resolve())
    c['review']['anchors_csv']=str((output/'anchors.csv').resolve())
    truth,anchors=[],[]
    # Fixture-provided ID anchors simulate a human identifying 3 known corners.
    # These coordinates are never used as measured centers in optimization.
    anchor_ids={'0',str(c['board']['cols']-1),str((c['board']['rows']-1)*c['board']['cols'])}
    for i,pose in enumerate(poses):
        frame='frame_%02d'%i
        image,centers=render_frame(c,state,pose,rng)
        save_image(output/'images'/(frame+'.png'),image)
        for center in centers:
            row=dict(image_id=frame,**center); truth.append(row)
            if row['point_id'] in anchor_ids:
                a=dict(row); a['u_px']+=float(rng.normal(0,.4)); a['v_px']+=float(rng.normal(0,.4))
                anchors.append(a)
    stress,_=render_frame(c,state,poses[0],rng,True)
    save_image(output/'challenges'/'occluded.png',stress)
    write_json(output/'config.json',c)
    write_json(output/'truth.json',dict(state=state,poses=poses,description='Synthetic fixture, not measured hardware',
                                      anchor_protocol='Three approximate known corners per view supply orientation only'))
    write_csv(output/'true_centers.csv',truth)
    write_csv(output/'anchors.csv',anchors)
    return c


def evaluate_centers(detected:list,truth:list)->dict:
    truth_map={(r['image_id'],r['view'],str(r['point_id'])):np.array([float(r['u_px']),float(r['v_px'])]) for r in truth}
    errors=[]
    for r in detected:
        if r.get('review_status')=='confirmed' and str(r.get('accepted')).lower() in ('true','1'):
            key=(r['image_id'],r['view'],str(r['point_id']))
            if key in truth_map:
                errors.append(np.linalg.norm(np.array([float(r['u_px']),float(r['v_px'])])-truth_map[key]))
    return dict(kind='synthetic_true_center_error',matched_count=len(errors),truth_count=len(truth),
                median_px=float(np.median(errors)) if errors else None,
                p95_px=float(np.percentile(errors,95)) if errors else None,
                rms_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
