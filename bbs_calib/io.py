from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, List, Dict
import cv2
import numpy as np


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serializable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path: Path, rows: List[dict], fields: List[str] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def new_output(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory is not empty; choose a new run directory: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Cannot decode image: ' + str(path))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Only 8/16-bit image inputs are supported')
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise ValueError('Cannot encode image: ' + str(path))
    encoded.tofile(str(path))


def display_u8(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, [1, 99])
    return np.clip((gray.astype(float)-lo)*255/max(hi-lo, 1), 0, 255).astype(np.uint8)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Explicit homogeneous coordinate transform; inverse uses solve, not guessing."""
    points = np.asarray(points, float).reshape(-1, 2)
    p = np.column_stack([points, np.ones(len(points))]) @ np.asarray(transform).T
    if np.any(np.abs(p[:, 2]) < 1e-12):
        raise ValueError('Invalid homogeneous coordinate')
    return p[:, :2]/p[:, 2:]


def roi_transform(roi: list, rotate90: int = 0, mirror: bool = False) -> tuple:
    """Returns full->processed homogeneous map and processed (width,height)."""
    x, y, w, h = roi
    T = np.array([[1., 0, -x], [0, 1, -y], [0, 0, 1]])
    if mirror:
        T = np.array([[-1., 0, w-1], [0, 1, 0], [0, 0, 1]]) @ T
    for _ in range(rotate90 % 4):
        T = np.array([[0., -1, h-1], [1, 0, 0], [0, 0, 1]]) @ T
        w, h = h, w
    return T, (w, h)
