# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Optional
from dataclasses import dataclass
from .logger import logger
from .profiler import profile_time
from .geometry import to_gray

OPTION_LABELS = ["B", "1", "2", "3"]
MIN_FILL_DENSITY = 0.03
MIN_INNER_DIFF = 5

@dataclass
class CellResult:
    label: Optional[str]
    confidence: float
    density: float
    fill_detected: bool

def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(to_gray(cell), (5, 5), 0)

def binarize_cell(gray: np.ndarray) -> np.ndarray:
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

def measure_band_density(bw: np.ndarray, y1: int, y2: int, x_margin: int = 4) -> float:
    h = y2 - y1
    if h <= 0:
        return 0.0
    y_inner_start = y1 + 2
    y_inner_end   = y2 - 2
    band = bw[y_inner_start:y_inner_end, x_margin:-x_margin]
    if band.size == 0:
        return 0.0
    return float(cv2.countNonZero(band)) / band.size

def estimate_bubble_x(gray: np.ndarray, bw: np.ndarray) -> int:
    proj = bw.sum(axis=0).astype(np.float32)
    if proj.max() <= 0:
        return gray.shape[1] // 2
    k = max(5, gray.shape[1] // 15)
    if k % 2 == 0:
        k += 1
    proj_smooth = cv2.GaussianBlur(proj.reshape(1, -1), (k, 1), 0).reshape(-1)
    return int(np.argmax(proj_smooth))

def component_candidates_in_band(gray, bw, y1, y2, x_center, x_tol_ratio=0.22):
    band_bw = bw[y1:y2, :]
    contours, _ = cv2.findContours(band_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    x_tol = max(12, int(W * x_tol_ratio))
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2 + y1
        if abs(cx - x_center) > x_tol:
            continue
        if w > W * 0.45 or h > (y2-y1) * 0.95:
            continue
        aspect = w / max(h, 1)
        if not (0.6 <= aspect <= 1.4):
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.35:
            continue
        r = int((w + h) / 4)
        candidates.append((cx, cy, r, area, circularity))
    return candidates

def score_candidate(gray, cx, cy, r):
    r_inner = max(3, int(r * 0.60))
    r_ring  = max(r_inner + 1, int(r * 0.95))
    mask_inner = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_inner, (cx, cy), r_inner, 255, -1)
    mask_outer = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_outer, (cx, cy), r_ring, 255, -1)
    cv2.circle(mask_outer, (cx, cy), max(1, r_inner), 0, -1)
    return {
        "mean_inner": cv2.mean(gray, mask=mask_inner)[0],
        "mean_ring":  cv2.mean(gray, mask=mask_outer)[0],
    }

def fallback_circle_in_band(gray, y1, y2, x_center):
    band = gray[y1:y2, :]
    circles = cv2.HoughCircles(
        band, cv2.HOUGH_GRADIENT, dp=1.15,
        minDist=max(10, (y2-y1)//2), param1=90, param2=10,
        minRadius=max(5, (y2-y1)//8), maxRadius=max(18, (y2-y1)//2)
    )
    candidates = []
    if circles is not None:
        for cx, cy_local, r in np.round(circles[0]).astype(int):
            cy = cy_local + y1
            if abs(cx - x_center) > max(12, int(gray.shape[1] * 0.22)):
                continue
            if r > gray.shape[1] * 0.22:
                continue
            candidates.append((cx, cy, r))
    return candidates

@profile_time("detect_filled_option_v4")
def detect_filled_option_v4(cell: np.ndarray, debug_name: Optional[str] = None) -> CellResult:
    if cell is None or cell.size == 0:
        return CellResult(label=None, confidence=0.0, density=0.0, fill_detected=False)

    gray = preprocess_cell(cell)
    bw   = binarize_cell(gray)

    H, W = gray.shape
    x_center   = estimate_bubble_x(gray, bw)
    band_edges = np.linspace(0, H, 5).astype(int)

    densities = []
    for i in range(4):
        d = measure_band_density(bw, band_edges[i], band_edges[i+1], x_margin=4)
        densities.append(d)

    max_density = max(densities)

    if max_density < MIN_FILL_DENSITY:
        return CellResult(label=None, confidence=1.0, density=max_density, fill_detected=False)

    band_results = []
    for i in range(4):
        y1 = band_edges[i]
        y2 = band_edges[i+1]

        candidates = component_candidates_in_band(gray, bw, y1, y2, x_center)
        chosen = None
        chosen_score = None

        if candidates:
            scored = [(score_candidate(gray, cx, cy, r)["mean_inner"],
                       abs(cx - x_center), -area, (cx, cy, r),
                       score_candidate(gray, cx, cy, r))
                      for cx, cy, r, area, _ in candidates]
            scored.sort(key=lambda t: (t[0], t[1], t[2]))
            _, _, _, chosen, chosen_score = scored[0]
        else:
            circles = fallback_circle_in_band(gray, y1, y2, x_center)
            if circles:
                scored = [(score_candidate(gray, cx, cy, r)["mean_inner"],
                           abs(cx - x_center), (cx, cy, r),
                           score_candidate(gray, cx, cy, r))
                          for cx, cy, r in circles]
                scored.sort(key=lambda t: (t[0], t[1]))
                _, _, chosen, chosen_score = scored[0]

        if chosen is None:
            cy = int((y1 + y2) / 2)
            chosen = (x_center, cy, max(6, min(W, y2-y1) // 6))
            chosen_score = score_candidate(gray, *chosen)

        cx, cy, r = chosen
        band_results.append({
            "label":      OPTION_LABELS[i],
            "cx": cx, "cy": cy, "r": r,
            "mean_inner": chosen_score["mean_inner"],
            "mean_ring":  chosen_score["mean_ring"],
            "density":    densities[i],
        })

    band_results.sort(key=lambda d: d["cy"])
    means = [d["mean_inner"] for d in band_results]
    
    if min(means) > np.mean(means) * 0.8:
        return CellResult(label=None, confidence=0,
                          density=max_density, fill_detected=False)
                          
    order = np.argsort(means)
    best   = int(order[0])
    second = int(order[1])
    diff   = float(means[second] - means[best])
    confidence = max(0.0, min(1.0, diff / 50.0))

    best_density = band_results[best]["density"]
    if best_density < MIN_FILL_DENSITY:
        return CellResult(label=None, confidence=confidence,
                          density=best_density, fill_detected=False)

    chosen_label = band_results[best]["label"]

    if diff < MIN_INNER_DIFF:
        return CellResult(label=None, confidence=confidence,
                          density=best_density, fill_detected=True)

    return CellResult(label=chosen_label, confidence=confidence,
                      density=best_density, fill_detected=True)
