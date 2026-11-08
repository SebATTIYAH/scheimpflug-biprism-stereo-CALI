"""Outer-boundary proposals for grids mixing solid dots and annular markers.

Detection may use a resized image; subpixel measurement always uses the original.
No missing dot is synthesized and ring interiors are never used as dot contours.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def outer_candidates(binary, scale, min_area, max_area):
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    candidates=[]
    for contour in contours:
        # In an inverted dark-dot binary image, hole contours have positive
        # orientation. Reject them even if their outer boundary fails size gates.
        if cv2.contourArea(contour,oriented=True)>=0:continue
        area=cv2.contourArea(contour)/(scale*scale)
        if not min_area<area<max_area or len(contour)<12:continue
        (x,y),(a,b),_=cv2.fitEllipse(contour)
        if min(a,b)/max(a,b)<.4:continue
        if 4*np.pi*cv2.contourArea(contour)/max(cv2.arcLength(contour,True)**2,1)<.5:continue
        center=(np.array([x,y])+.5)/scale-.5
        full=(contour.astype(np.float32)+.5)/scale-.5
        candidates.append((center,full,area,min(a,b)/(2*scale)))
    # Largest concentric contour wins irrespective of OpenCV enumeration order.
    blobs=[]
    for item in sorted(candidates,key=lambda v:-v[2]):
        if any(np.linalg.norm(item[0]-v[0])<max(2.,.45*v[3]) and
               cv2.pointPolygonTest(v[1],tuple(map(float,item[0])),False)>=0 for v in blobs):
            continue
        blobs.append(item)
    return blobs


def ordered_grid(blobs, rows, cols, max_error):
    pts=np.asarray([v[0] for v in blobs],np.float32)
    # OpenCV clustering distances are in pixels. Normalize by typical dot radius,
    # not fixed sensor resolution, to keep the grid finder scale-independent.
    radius=np.median([v[3] for v in blobs])
    factor=4./max(radius,1.)
    ok,ordered=cv2.findCirclesGrid((pts*factor).reshape(-1,1,2),(cols,rows),
        flags=cv2.CALIB_CB_SYMMETRIC_GRID|cv2.CALIB_CB_CLUSTERING,
        blobDetector=None,parameters=cv2.CirclesGridFinderParameters())
    if not ok:return None,None
    grid=ordered.reshape(rows,cols,2)/factor
    if rows==cols:
        # Square grids have an additional transpose ambiguity. Choose the axis
        # closest to image horizontal as columns before applying sign flips.
        across_columns=np.mean(grid[:,-1,:]-grid[:,0,:],axis=0)
        across_rows=np.mean(grid[-1,:,:]-grid[0,:,:],axis=0)
        if abs(across_columns[0])<abs(across_rows[0]):grid=grid.transpose(1,0,2)
    if np.mean(grid[-1,:,1])<np.mean(grid[0,:,1]):grid=grid[::-1]
    if np.mean(grid[:,-1,0])<np.mean(grid[:,0,0]):grid=grid[:,::-1]
    selected=[int(np.argmin(np.linalg.norm(pts-p,axis=1))) for p in grid.reshape(-1,2)]
    lattice=np.array([[j,i,1.] for i in range(rows) for j in range(cols)])
    for _ in range(3):
        H=np.linalg.lstsq(lattice,pts[selected],rcond=None)[0]
        cost=np.linalg.norm((lattice@H)[:,None,:]-pts[None,:,:],axis=2)
        _,indices=linear_sum_assignment(cost);selected=indices.tolist()
    H=np.linalg.lstsq(lattice,pts[selected],rcond=None)[0]
    error=float(np.max(np.linalg.norm(lattice@H-pts[selected],axis=1)))
    if len(set(selected))!=rows*cols or error>max_error:return None,error
    return selected,error


def find_standard_grid(gray, rows, cols, min_area_px=150., max_area_px=15000., max_grid_error=3.):
    from .prism_workflow import validate_grid
    validate_grid(rows,cols)
    if not np.isfinite([min_area_px,max_area_px]).all() or not 0<min_area_px<max_area_px:
        raise ValueError('Dot areas require 0 < minimum < maximum (original px^2)')
    count=rows*cols;attempts=[]
    scales=sorted(set(min(1.,target/max(gray.shape)) for target in (1000.,1600.)))
    for requested_scale in scales:
        # Use the exact same scale in both axes, with explicit half-pixel mapping.
        scale=requested_scale
        work=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else gray
        for block,C in ((151,8),(251,10),(81,5),(151,15)):
            binary=cv2.adaptiveThreshold(work,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,C)
            blobs=outer_candidates(binary,scale,min_area_px,max_area_px)
            attempt=dict(scale=scale,block=block,C=C,outer_candidates=len(blobs),size_groups=[])
            attempts.append(attempt)
            if len(blobs)<count:continue
            areas=np.array([b[2] for b in blobs]);groups=[];seen=set()
            for area in areas:
                indices=tuple(np.flatnonzero((areas>.6*area)&(areas<1.65*area)))
                if indices not in seen and count<=len(indices)<=max(2*count+4,30):
                    seen.add(indices);groups.append(indices)
            # Dots form a repeated size population; prefer populations near the expected count.
            groups.sort(key=lambda indices:abs(len(indices)-count))
            for indices in groups[:12]:
                group=[blobs[i] for i in indices]
                selected,error=ordered_grid(group,rows,cols,max_grid_error)
                attempt['size_groups'].append(dict(count=len(group),max_grid_affine_error_px=error))
                if selected is not None:
                    attempt['selected_outer_area_median_px2']=float(np.median([group[i][2] for i in selected]))
                    return [(v[0],v[1]) for v in group],selected,attempts
    return None,None,attempts
