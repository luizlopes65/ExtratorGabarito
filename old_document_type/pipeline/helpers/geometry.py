# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Optional
from .logger import logger
from .profiler import profile_time

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 0) -> Optional[np.ndarray]:
    """Crop with padding, gracefully reducing pad if the cell is too small."""
    while pad >= 0:
        cx1 = max(0, x1 + pad)
        cy1 = max(0, y1 + pad)
        cx2 = min(img.shape[1], x2 - pad)
        cy2 = min(img.shape[0], y2 - pad)
        if cx2 > cx1 and cy2 > cy1:
            return img[cy1:cy2, cx1:cx2].copy()
        pad -= 1
    return None

def cluster_positions(values: list[int], tolerance: int = 10) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if abs(v - np.mean(groups[-1])) <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(np.mean(g))) for g in groups]

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def find_document_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 40, 140)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area_img = gray.shape[0] * gray.shape[1]
    for cnt in contours[:20]:
        area = cv2.contourArea(cnt)
        if area < area_img * 0.25:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

@profile_time("preprocess_document")
def preprocess_document(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    doc = find_document_contour(gray)
    if doc is not None:
        img = four_point_transform(img, doc)
    return img
