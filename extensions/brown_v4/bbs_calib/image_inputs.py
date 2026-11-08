"""Pair pre-cropped images and preserve the common sensor coordinate system."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .io import read_csv, read_gray


def input_mode(c: dict) -> str:
    mode=c.get('images',{}).get('mode','full_frame')
    if mode not in ('full_frame','paired_crops'):
        raise ValueError('images.mode must be full_frame or paired_crops')
    return mode


def crop_origins(c: dict) -> dict:
    """Explicit offsets, or dimensions under the verified MATLAB midpoint rule."""
    path=c.get('images',{}).get('crop_origins_csv')
    result={}
    if path:
        for row in read_csv(Path(path)):
            image_id,view=row['image_id'],row['view']
            if view not in ('left','right') or view in result.get(image_id,{}):
                raise ValueError('Invalid/duplicate crop origin row: '+str((image_id,view)))
            origin=validate_origin([float(row['x0_px']),float(row['y0_px'])])
            result.setdefault(image_id,{})[view]=origin.tolist()
    policy=c.get('images',{}).get('crop_layout','explicit')
    if policy not in ('explicit','matlab_midpoint_vertical'):
        raise ValueError('Unknown images.crop_layout: '+str(policy))
    if policy=='matlab_midpoint_vertical':
        if input_mode(c)!='paired_crops':
            raise ValueError('Midpoint layout requires paired_crops')
        for frame in discover_frames(c):
            hl,wl=read_gray(frame['left']).shape
            hr,wr=read_gray(frame['right']).shape
            if hl!=hr or wl<1 or wr not in (wl,wl+1):
                raise ValueError('Invalid MATLAB midpoint crop dimensions: '+frame['image_id'])
            origins={'left':[0,0],'right':[wl,0]}
            for view,origin in origins.items():
                supplied=[result.get(frame['image_id'],{}).get(view),
                          c['views'][view].get('crop_origin_px')]
                for value in supplied:
                    if value is not None and not np.array_equal(validate_origin(value),origin):
                        raise ValueError('Explicit crop origin conflicts with midpoint layout: '+frame['image_id'])
            result[frame['image_id']]=origins
    return result


def validate_origin(value: list) -> np.ndarray:
    a=np.asarray(value,float)
    if a.shape!=(2,) or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('Crop origin must be finite nonnegative [x0_px,y0_px]')
    return a


def origin_for(c:dict,image_id:str,view:str,table:dict=None):
    if input_mode(c)=='full_frame':
        return np.zeros(2)
    table=crop_origins(c) if table is None else table
    origin=table.get(image_id,{}).get(view,c['views'][view].get('crop_origin_px'))
    return None if origin is None else validate_origin(origin)


def discover_frames(c:dict) -> list:
    files=sorted(p for p in Path(c['images']['directory']).glob(c['images']['pattern']) if p.is_file())
    if not files:
        raise ValueError('No images match configured directory/pattern')
    if input_mode(c)=='full_frame':
        if len({p.stem for p in files})!=len(files):
            raise ValueError('Image stems must be unique')
        return [dict(image_id=p.stem,left=p,right=p) for p in files]
    suffixes=c.get('images',{}).get('suffixes',{'left':'_0','right':'_1'})
    if set(suffixes)!= {'left','right'} or not all(isinstance(s,str) and s for s in suffixes.values()) or suffixes['left']==suffixes['right']:
        raise ValueError('Distinct left/right crop suffixes are required')
    pairs={}
    for p in files:
        matching=[v for v,s in suffixes.items() if p.stem.endswith(s)]
        if len(matching)!=1:
            raise ValueError('File must end with exactly one configured crop suffix: '+p.name)
        view=matching[0];image_id=p.stem[:-len(suffixes[view])]
        if not image_id:
            raise ValueError('Cropped image requires a frame prefix: '+p.name)
        if view in pairs.get(image_id,{}):
            raise ValueError('Duplicate crop partner (possibly two extensions): '+p.name)
        pairs.setdefault(image_id,{})[view]=p
    missing=[k for k,v in pairs.items() if set(v)!={'left','right'}]
    if missing:
        raise ValueError('Missing _0/_1 partner for frames: '+', '.join(sorted(missing)))
    return [dict(image_id=k,**pairs[k]) for k in sorted(pairs)]


def sensor_pixels(pixels:np.ndarray,calibration:dict,view:str,image_id:str=None)->np.ndarray:
    """Convert cropped input for reconstruction using offsets stored in the result."""
    if calibration.get('input_mode','full_frame')!='paired_crops':
        raise ValueError('Calibration was not configured for pre-cropped input')
    table=calibration.get('crop_origins_by_image',{})
    origin=table.get(image_id,{}).get(view,calibration.get('default_crop_origins',{}).get(view))
    if origin is None:
        raise ValueError('Crop origin unavailable for reconstruction; supply the recorded image_id or full-frame coordinates')
    return np.asarray(pixels,float)+validate_origin(origin)
