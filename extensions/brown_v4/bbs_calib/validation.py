from __future__ import annotations
from pathlib import Path
import numpy as np
from .initialization import initialize_pose
from .projection import object_points, project
from .reconstruction import reconstruct_points
from .geometry import optical_system
from .io import read_csv


def error_metrics(errors:np.ndarray)->dict:
    errors=np.asarray(errors,float).reshape(-1,2)
    if not len(errors):
        return dict(count=0,euclidean_rms_px=None)
    lengths=np.linalg.norm(errors,axis=1)
    return dict(count=len(errors),euclidean_rms_px=float(np.sqrt(np.mean(lengths**2))),
                component_rms_px=float(np.sqrt(np.mean(errors**2))),
                axis_rms_px=np.sqrt(np.mean(errors**2,axis=0)),axis_mean_px=np.mean(errors,axis=0),
                median_px=float(np.median(lengths)),p95_px=float(np.percentile(lengths,95)),max_px=float(lengths.max()))


def validate(calibration:dict,used:list,frames:list,c:dict)->dict:
    state=calibration['state']
    state={k:np.array(v) if k in ('K','c','omega','kappa') else v for k,v in state.items()}
    rng=np.random.default_rng(c.get('seed',1234)+29)
    reports,errors,reconstructions=[],[],{}
    mapping=calibration['view_branches']
    for frame in frames:
        records=[v for v in used if v['row']['image_id']==frame]
        ids=sorted(set(v['row']['point_id'] for v in records))
        fit_count=max(4,len(ids)//2)
        if len(ids)<=fit_count+1:
            reports.append(dict(image_id=frame,status='insufficient_held_out_points')); continue
        fitting=set(rng.choice(ids,fit_count,replace=False).tolist())
        fit=[v for v in records if v['row']['point_id'] in fitting]
        test=[v for v in records if v['row']['point_id'] not in fitting]
        try:
            pose,diag=initialize_pose(np.array([v['P'] for v in fit]),np.array([v['p'] for v in fit]),[v['branch'] for v in fit],state)
            residual=[]
            for v in test:
                residual.append(project(object_points(v['P'][None,:],pose),v['branch'],state)[0]-v['p'])
            errors.extend(residual)
            views={v:{o['row']['point_id']:o for o in records if o['row']['view']==v} for v in ('left','right')}
            matched=sorted(set(views['left'])&set(views['right']))
            rec=reconstruct_points(np.array([views['left'][i]['p'] for i in matched]),
                                   np.array([views['right'][i]['p'] for i in matched]),calibration)
            report=dict(image_id=frame,status='evaluated',pose_fit_point_ids=sorted(fitting),
                        reprojection_test_point_ids=sorted(set(ids)-fitting),pose_fit_diagnostics=diag,
                        global_parameters_fixed=True,errors=error_metrics(residual),geometry_valid=rec['geometry_valid'],
                        reconstruction_condition=rec['condition'],pose=pose)
            if rec['X_mm'] is not None:
                X=rec['X_mm']; truth=np.array([views['left'][i]['P'] for i in matched])
                if rec['geometry_valid']:
                    reconstructions[frame]=dict(zip(matched,X))
                # Only held-out IDs contribute to reported metric distances and flatness.
                selected=np.array([i not in fitting for i in matched])
                Xt,Pt=X[selected],truth[selected]
                good=np.asarray(rec['valid_mask'])[selected]
                if not good.all():
                    report['metric_geometry_status']='invalid_optical_paths; metrics withheld'
                elif len(Xt)>=3:
                    target_dist=np.linalg.norm(Pt[:,None,:]-Pt[None,:,:],axis=2)
                    actual_dist=np.linalg.norm(Xt[:,None,:]-Xt[None,:,:],axis=2)
                    idx=np.triu_indices(len(Xt),1)
                    difference=actual_dist[idx]-target_dist[idx]
                    report['pair_distance_rms_mm']=float(np.sqrt(np.mean(difference**2)))
                    report['pair_distance_p95_mm']=float(np.percentile(abs(difference),95))
                    if np.linalg.matrix_rank(Pt-Pt.mean(axis=0),tol=1e-8)==2:
                        _,_,Vt=np.linalg.svd(Xt-Xt.mean(axis=0))
                        heights=(Xt-Xt.mean(axis=0))@Vt[-1]
                        report['plane_rms_mm']=float(np.sqrt(np.mean(heights**2)))
                    else:
                        report['plane_rms_mm']=None
                    report['metric_note']='Held-out IDs; known board distances; no free-scale alignment. Plane fit estimates only a plane.'
                report['reconstruction']=rec
                report['matched_point_ids']=matched
            reports.append(report)
        except ValueError as exc:
            reports.append(dict(image_id=frame,status='failed',reason=str(exc)))
    external=[]
    translation_file=c.get('validation',{}).get('translations_csv')
    if translation_file:
        for r in read_csv(Path(translation_file)):
            a,b=r['from_image'],r['to_image']
            if a not in reconstructions or b not in reconstructions:
                external.append(dict(from_image=a,to_image=b,status='frames_not_available')); continue
            shared=sorted(set(reconstructions[a])&set(reconstructions[b]))
            gt=np.array([float(r[k]) for k in ('dx_mm','dy_mm','dz_mm')])
            delta=np.array([reconstructions[b][i]-reconstructions[a][i] for i in shared])
            external.append(dict(from_image=a,to_image=b,status='evaluated',truth_mm=gt,
                                 mean_displacement_mm=delta.mean(axis=0),rms_error_mm=float(np.sqrt(np.mean(np.sum((delta-gt)**2,axis=1)))),
                                 convention='External displacement must be expressed in fixed prism axes'))
    return dict(whole_frames=frames,global_parameters_fixed=True,point_split='Half IDs fit held-out pose, remaining IDs evaluate',
                metrics=error_metrics(errors),per_frame=reports,external_translation=external or None,
                external_validation_available=any(r.get('status')=='evaluated' for r in external),
                limitation='Board self-consistency is not independent metrological truth; external displacements require supplied CSV')
