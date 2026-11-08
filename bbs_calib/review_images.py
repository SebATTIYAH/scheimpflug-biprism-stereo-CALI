"""Pixel-center-correct review rendering. Never modifies measured coordinates."""
from pathlib import Path
import cv2
import numpy as np
from .io import read_json,read_csv,read_gray,save_image,write_json,new_output


def resize_points(points,source_shape,target_shape):
    """OpenCV resize pixel-center convention, with shapes in (height, width)."""
    scale=np.array([target_shape[1]/source_shape[1],target_shape[0]/source_shape[0]])
    return (np.asarray(points,dtype=float)+.5)*scale-.5


def draw_center(image,point,label=None,color=(0,0,255),radius=2.5):
    """Draw a cross at floating-point coordinates using OpenCV fixed-point shift."""
    p=np.asarray(point,float);shift=8;factor=1<<shift
    def fixed(q):return tuple(np.rint(np.asarray(q)*factor).astype(int))
    for d in ([radius,0],[0,radius]):
        cv2.line(image,fixed(p-d),fixed(p+d),color,1,cv2.LINE_AA,shift)
    if label is not None:
        cv2.putText(image,str(label),tuple(np.rint(p+[4,-4]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,.4,(0,200,0),1,cv2.LINE_AA)


def render(gray,points,size=None,corner_ids=None):
    base=gray if size is None else cv2.resize(gray,size)
    from .io import display_u8
    image=cv2.cvtColor(display_u8(base),cv2.COLOR_GRAY2BGR)
    for item in points:
        pid=int(item['point_id'])
        p=resize_points(item['center'],gray.shape,image.shape)
        draw_center(image,p,label=pid if corner_ids is None or pid in corner_ids else None,radius=5 if size is None else 2.5)
    return image


def regenerate(source,output,mode,image_id=None,full_resolution=False):
    """Re-render accepted saved centers; neither extraction nor fitting is run."""
    source=Path(source).resolve()
    metadata=read_json(source/'cli_input.json') if (source/'cli_input.json').is_file() else {}
    corner_ids=None
    jobs=[]
    if mode=='single':
        frames=read_json(source/'frames.json');audit=read_json(source/'audit.json')
        paths={Path(a['file']).stem:Path(a['file']) for a in audit}
        for frame in frames:
            if image_id is not None and frame['id']!=image_id:continue
            if frame['id'] not in paths:raise ValueError('Missing original image path: '+frame['id'])
            points=[dict(point_id=pid,center=p) for pid,p in zip(frame['ids'],frame['p'])]
            jobs.append((frame['id'],[('single',paths[frame['id']],points)]))
    else:
        from .config import load_config
        from .image_inputs import discover_frames
        c=load_config(source/'config.json')
        metadata=c['board']
        grouped={}
        for row in read_csv(source/'observations.csv'):
            if row.get('accepted','').lower() not in ('true','1') or row.get('review_status')!='confirmed':continue
            key=(row['image_id'],row['view'])
            if not row.get('source_u_px') or not row.get('source_v_px'):
                raise ValueError('Review needs saved crop-local source_u_px/source_v_px')
            grouped.setdefault(key,[]).append(dict(point_id=int(row['point_id']),center=[float(row['source_u_px']),float(row['source_v_px'])]))
        for frame in discover_frames(c):
            fid=frame['image_id']
            if image_id is not None and fid!=image_id:continue
            views=[(view,frame[view],grouped.get((fid,view),[])) for view in ('left','right')]
            if any(points for _,_,points in views):jobs.append((fid,views))
    if metadata.get('rows') and metadata.get('cols'):
        nr,nc=int(metadata['rows']),int(metadata['cols'])
        corner_ids={0,nc-1,(nr-1)*nc,nr*nc-1}
    if not jobs:raise ValueError('No retained frame matches --image-id / source data')
    for _,views in jobs:
        for _,path,_ in views:
            if not path.is_file():raise ValueError('Original image missing: '+str(path))
    out=new_output(Path(output));manifest=[]
    for fid,views in jobs:
        # Input file stems are treated as labels, never arbitrary output paths.
        if Path(fid).name!=fid or '/' in fid or '\\' in fid or fid in ('.','..'):
            raise ValueError('Invalid image ID for review output')
        previews=[]
        for view,path,points in views:
            gray=read_gray(path)
            preview=render(gray,points,(750,600) if mode=='single' else (375,600),corner_ids)
            previews.append(preview)
            if full_resolution:save_image(out/'full_resolution'/(fid+'_'+view+'.png'),render(gray,points,corner_ids=corner_ids))
            manifest.append(dict(image_id=fid,view=view,source_image=str(path),points=len(points),
                source_shape=list(gray.shape),preview_shape=list(preview.shape[:2])))
        save_image(out/(fid+'_review.png'),np.hstack(previews))
        print('Review saved: '+fid,flush=True)
    write_json(out/'review_manifest.json',dict(source=str(source),mode=mode,full_resolution=full_resolution,
        center_data_changed=False,transform='p_preview=(p_source+0.5)*scale-0.5; fixed-point cross shift=8',frames=manifest))
    return out
