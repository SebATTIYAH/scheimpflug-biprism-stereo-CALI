from __future__ import annotations
from .runtime import configure
configure()
import argparse
import json
from pathlib import Path
import sys
import traceback


def main(argv:list=None)->int:
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv==['--version']:
        from . import __version__
        print(__version__)
        return 0
    if not argv or argv[0] in ('extract','fit','inspect','review','audit-points','filter-blur','init','-h','--help'):
        from .step_cli import main as step_main
        return step_main(argv)
    if argv and argv[0] in ('single','prism'):
        from .workflow_cli import main as workflow_main
        return workflow_main(argv)
    from .io import new_output,read_json,read_csv,write_json,write_csv,serializable
    parser=argparse.ArgumentParser(description='Experimental shared Scheimpflug/biprism calibration',
        epilog='Real-data workflows: single --help | prism --help | inspect --help')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    demo=sub.add_parser('demo');demo.add_argument('--output',required=True)
    for name in ('detect','calibrate'):
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--output')
        if name=='calibrate':p.add_argument('--observations',required=True)
    p=sub.add_parser('validate');p.add_argument('--calibration',required=True);p.add_argument('--observations',required=True);p.add_argument('--output')
    p=sub.add_parser('reconstruct');p.add_argument('--calibration',required=True);p.add_argument('--pairs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--coordinates',choices=['full_frame','crop_local'],default='full_frame');p.add_argument('--image-id')
    args=parser.parse_args(argv)
    try:
        if args.command=='doctor':
            from .reporting import environment
            e=environment()
            e['dependencies_imported']=True
            print(json.dumps(e,ensure_ascii=False,indent=2))
            return 0
        if args.command=='demo':
            from .synthetic import generate,evaluate_centers
            from .detection import detect_dataset
            from .optimization import observations,calibrate
            from .reporting import save_results
            out=new_output(Path(args.output))
            c=generate(out)
            print('Synthetic images generated',flush=True)
            det=detect_dataset(c,new_output(out/'detection'))
            centers=evaluate_centers(read_csv(out/'detection'/'observations.csv'),read_csv(out/'true_centers.csv'))
            write_json(out/'center_accuracy.json',centers)
            print(json.dumps(centers),flush=True)
            obs=observations(out/'detection'/'observations.csv',c)
            result=calibrate(c,obs)
            cal=save_results(result,c,obs,new_output(out/'calibration'),' '.join(sys.argv))
            # Truth-based synthetic reconstruction accuracy without rotation/scale alignment.
            from .reconstruction import reconstruct_points
            from .projection import object_points
            from .config import board_points
            import numpy as np
            truth=read_json(out/'truth.json');ids,P=board_points(c)
            reconstruction=[]
            for index,frame in enumerate(obs['frames']):
                byview={v:{r['row']['point_id']:r['p'] for r in obs['used'] if r['row']['image_id']==frame and r['row']['view']==v} for v in ('left','right')}
                matched=[i for i in ids if i in byview['left'] and i in byview['right']]
                rec=reconstruct_points([byview['left'][i] for i in matched],[byview['right'][i] for i in matched],cal)
                if rec['X_mm'] is not None:
                    Xtruth=object_points(P[[ids.index(i) for i in matched]],np.array(truth['poses'][index]))
                    err=np.linalg.norm(rec['X_mm']-Xtruth,axis=1)
                    reconstruction.append(dict(image_id=frame,rms_mm=float(np.sqrt(np.mean(err**2))),p95_mm=float(np.percentile(err,95)),
                                               geometry_valid=rec['geometry_valid'],alignment='none; coordinate gauge can affect absolute error'))
            write_json(out/'synthetic_reconstruction_accuracy.json',reconstruction)
            print(json.dumps(serializable(dict(output=out,center_accuracy=centers,status=cal['status'])),ensure_ascii=False,indent=2))
            return 0
        if args.command in ('detect','calibrate'):
            from .config import load_config,validate_config
            c=load_config(Path(args.config));validate_config(c,args.command)
            if not args.output and not c.get('output'):
                raise ValueError('Configure output or pass --output')
            if args.command=='detect':
                from .detection import detect_dataset
                out=new_output(Path(args.output or c['output']))
                result=detect_dataset(c,out)
                print(json.dumps({k:v for k,v in result.items() if k!='audit'},ensure_ascii=False))
            else:
                from .optimization import observations,calibrate
                from .reporting import save_results
                obs=observations(Path(args.observations),c)
                out=new_output(Path(args.output or c['output']))
                result=calibrate(c,obs)
                cal=save_results(result,c,obs,out,' '.join(sys.argv))
                print(json.dumps(serializable(cal['status']),ensure_ascii=False,indent=2))
            return 0
        if args.command=='validate':
            from .optimization import observations
            from .validation import validate
            cal=read_json(Path(args.calibration));c=cal['config_snapshot']
            obs=observations(Path(args.observations),c)
            overlap=set(obs['frames'])&set(cal.get('training_frames',[]))
            if overlap:
                raise ValueError('Validation input overlaps training frames: '+str(sorted(overlap)))
            out=new_output(Path(args.output) if args.output else Path(args.calibration).parent/'validation_run')
            result=validate(cal,obs['used'],obs['frames'],c)
            write_json(out/'validation.json',result)
            print(json.dumps(serializable(result['metrics']),ensure_ascii=False,indent=2))
            return 0
        if args.command=='reconstruct':
            from .reconstruction import reconstruct_points
            import numpy as np
            cal=read_json(Path(args.calibration));pairs=read_csv(Path(args.pairs))
            ids=[r['point_id'] for r in pairs]
            if len(set(ids))!=len(ids): raise ValueError('Duplicate point IDs')
            rec=reconstruct_points([[float(r[k]) for k in ('u_left_px','v_left_px')] for r in pairs],
                                   [[float(r[k]) for k in ('u_right_px','v_right_px')] for r in pairs],cal,
                                   pixel_coordinates=args.coordinates,image_id=args.image_id)
            out=new_output(Path(args.output));write_json(out/'reconstruction.json',rec)
            if rec['X_mm'] is not None:
                write_csv(out/'points3d.csv',[dict(point_id=i,X_mm=x[0],Y_mm=x[1],Z_mm=x[2],valid=bool(v))
                                             for i,x,v in zip(ids,rec['X_mm'],rec['valid_mask'])])
            return 0 if rec['geometry_valid'] else 3
    except (ValueError,KeyError,OSError,ImportError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    sys.exit(main())
