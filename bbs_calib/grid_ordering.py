from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def symmetry_permutations(grid: np.ndarray) -> list:
    candidates=[]
    bases=[grid]
    # Swapping rectangular grid axes is handled as a second detector shape below.
    if grid.shape[0]==grid.shape[1]:
        bases.append(grid.T)
    for a in bases:
        for b in (a,a[::-1],a[:,::-1],a[::-1,::-1]):
            if not any(np.array_equal(b,x) for x in candidates):
                candidates.append(b.copy())
    return candidates


def order_grid(centers: np.ndarray, rows: int, cols: int, anchors: list = None,
               tolerance: float = 5.) -> dict:
    centers=np.asarray(centers,float).reshape(-1,2)
    if len(centers)!=rows*cols:
        return dict(status='needs_review',mapping={},reason='Incomplete/extra grid: explicit reviewed IDs required; no predicted points inserted',candidates=[])
    alternatives=[]
    for nr,nc,transpose in [(rows,cols,False),(cols,rows,True)]:
        try:
            ok,ordered=cv2.findCirclesGrid(centers.astype(np.float32).reshape(-1,1,2),(nc,nr),
                                         flags=cv2.CALIB_CB_SYMMETRIC_GRID|cv2.CALIB_CB_CLUSTERING,
                                         blobDetector=None,parameters=cv2.CirclesGridFinderParameters())
        except cv2.error:
            ok=False
        if not ok:
            continue
        distances=cdist(ordered.reshape(-1,2),centers)
        a,b=linear_sum_assignment(distances)
        if distances[a,b].max()>1e-3:
            continue
        grid=b.reshape(nr,nc)
        if transpose:
            grid=grid.T
        for candidate in symmetry_permutations(grid):
            if not any(np.array_equal(candidate,x) for x in alternatives):
                alternatives.append(candidate)
    if not alternatives:
        return dict(status='needs_review',mapping={},reason='findCirclesGrid could not establish full grid',candidates=[])
    xy=np.array([[j,i,1.] for i in range(rows) for j in range(cols)])
    scores=[]
    for g in alternatives:
        uv=centers[g.ravel()]
        fit=np.linalg.lstsq(xy,uv,rcond=None)[0]
        scores.append(float(np.sqrt(np.mean(np.sum((xy@fit-uv)**2,axis=1)))))
    passing=[]
    anchors=anchors or []
    # Three known noncollinear IDs ensure the orientation is actually supplied.
    anchor_ids=[int(a['point_id']) for a in anchors]
    enough=(len(set(anchor_ids))>=3 and all(0<=i<rows*cols for i in anchor_ids))
    if enough:
        enough=np.linalg.matrix_rank(xy[anchor_ids])==3
    if enough:
        for idx,g in enumerate(alternatives):
            err=[np.linalg.norm(centers[g.ravel()[int(a['point_id'])]]-[float(a['u_px']),float(a['v_px'])]) for a in anchors]
            if max(err)<=tolerance:
                passing.append(idx)
    if len(passing)!=1:
        return dict(status='needs_review',mapping={},reason='Symmetric grid orientation unresolved: supply >=3 noncollinear ID anchors in each view',
                    candidates=[g.ravel().tolist() for g in alternatives],grid_rms_px=scores)
    g=alternatives[passing[0]]
    # Strong affine consistency is assessed, but moderate distortion must remain possible.
    nearest=cdist(centers,centers)+np.eye(len(centers))*1e10
    spacing=float(np.median(nearest.min(axis=1)))
    if scores[passing[0]]>max(1.,spacing*.15):
        return dict(status='needs_review',mapping={},reason='Grid geometry inconsistent even after anchors',candidates=[])
    return dict(status='confirmed',mapping={int(k):str(i) for i,k in enumerate(g.ravel())},
                reason='Unique symmetry candidate consistent with supplied noncollinear anchors',
                grid_rms_px=scores[passing[0]],candidates=[g.ravel().tolist()])
