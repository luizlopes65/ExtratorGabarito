# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import re
from typing import List, Tuple, Optional
# pyrefly: ignore [missing-import]
import pytesseract
from .logger import logger
from .profiler import profile_time
from .geometry import to_gray, crop, cluster_positions

GRID_CLUSTER_TOLERANCE = 12
ROW_HEIGHT_MIN = 20
COL_WIDTH_MIN = 25
NARROW_COL_RATIO: float = 0.40

def binarize_for_grid(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10
    )

def detect_grid_masks(bin_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bin_img.shape
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 28)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 28), 1))
    vertical = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    return vertical, horizontal

def extract_line_positions(line_img: np.ndarray, axis: str) -> List[int]:
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if axis == "vertical":
            if h > 40:
                coords.append(x + w // 2)
        else:
            if w > 40:
                coords.append(y + h // 2)
    return cluster_positions(coords, tolerance=GRID_CLUSTER_TOLERANCE)

def filter_margin_columns(col_intervals: List[Tuple[int, int]], img_width: int, expected_col_count: int = 0) -> List[Tuple[int, int]]:
    if not col_intervals:
        return col_intervals

    widths = [x2 - x1 for x1, x2 in col_intervals]
    median_w = float(np.median(widths))
    min_w = NARROW_COL_RATIO * median_w
    margin_px = int(img_width * 0.015)

    filtered = []
    for x1, x2 in col_intervals:
        w = x2 - x1
        cx = (x1 + x2) / 2
        if w < min_w:
            logger.debug(f"  [filter_cols] Dropping narrow column x={x1}-{x2} (w={w:.0f} < {min_w:.0f})")
            continue
        if cx < margin_px or cx > img_width - margin_px:
            logger.debug(f"  [filter_cols] Dropping margin column x={x1}-{x2} (cx={cx:.0f})")
            continue
        filtered.append((x1, x2))

    if expected_col_count > 0 and len(filtered) < expected_col_count:
        logger.debug(f"  [filter_cols] WARNING: Filtered to {len(filtered)} columns but QR expects {expected_col_count}")
        logger.debug(f"  [filter_cols] Reverting to original {len(col_intervals)} columns to preserve QR data")
        return col_intervals

    if not filtered:
        logger.debug("  [filter_cols] WARNING: all columns were filtered — reverting to original")
        return col_intervals

    return filtered

@profile_time("identify_header_and_student_rows")
def identify_header_and_student_rows(img, row_intervals, use_qr_mode=False):
    if not row_intervals:
        return None, []
    if len(row_intervals) < 2:
        return row_intervals[0], []

    heights = [y2 - y1 for y1, y2 in row_intervals]
    median_h = float(np.median(heights))

    header_row_idx = None
    
    if not use_qr_mode:
        gray = to_gray(img)
        img_w = img.shape[1]
        scan_x2 = int(img_w * 0.30)
    
        for i, (y1, y2) in enumerate(row_intervals[:6]):
            cell = crop(gray, 0, y1, scan_x2, y2, pad=2)
            if cell is None:
                continue
            try:
                txt = pytesseract.image_to_string(
                    cell, lang="por+eng",
                    config="--oem 3 --psm 6"
                ).upper()
            except Exception:
                txt = ""
            if re.search(r"NOME", txt):
                header_row_idx = i
                logger.debug(f"  [header] Found 'Nome' in row {i} via OCR → header_row_idx={i}")
                break
    else:
        logger.debug("  [header] Skipping OCR header detection (QR mode active)")

    if header_row_idx is None:
        n_candidates = min(5, len(row_intervals))
        candidates = [(y2 - y1, i) for i, (y1, y2) in enumerate(row_intervals[:n_candidates])]
        candidates.sort()
        header_row_idx = candidates[0][1]
        logger.debug(f"  [header] OCR fallback: shortest row among first {n_candidates} → idx={header_row_idx}")

    header_row = row_intervals[header_row_idx]
    min_student_h = max(40, median_h * 0.60)
    student_rows = []
    for iv in row_intervals[header_row_idx + 1:]:
        h = iv[1] - iv[0]
        if h >= min_student_h:
            student_rows.append(iv)

    return header_row, student_rows

@profile_time("get_table_structure")
def get_table_structure(img: np.ndarray, expected_question_count: int = 0):
    gray = to_gray(img)
    bw = binarize_for_grid(gray)
    vertical, horizontal = detect_grid_masks(bw)

    xs = cluster_positions(extract_line_positions(vertical, "vertical"), tolerance=GRID_CLUSTER_TOLERANCE)
    ys = cluster_positions(extract_line_positions(horizontal, "horizontal"), tolerance=GRID_CLUSTER_TOLERANCE)

    raw_col_intervals = [(xs[i], xs[i+1]) for i in range(len(xs)-1) if xs[i+1]-xs[i] >= COL_WIDTH_MIN]

    expected_col_count = expected_question_count + 1 if expected_question_count > 0 else 0
    col_intervals = filter_margin_columns(raw_col_intervals, img.shape[1], expected_col_count)

    all_row_heights = [(ys[i], ys[i+1], ys[i+1]-ys[i]) for i in range(len(ys)-1)]
    logger.debug("\nAll detected row positions (y1, y2, height):")
    for y1, y2, h in all_row_heights:
        status = "✓ KEPT" if h >= ROW_HEIGHT_MIN else f"✗ FILTERED (< {ROW_HEIGHT_MIN}px)"
        logger.debug(f"  Row {y1:4d}-{y2:4d}: {h:3d}px {status}")

    row_intervals = [(ys[i], ys[i+1]) for i in range(len(ys)-1) if ys[i+1]-ys[i] >= ROW_HEIGHT_MIN]

    return xs, ys, col_intervals, row_intervals

def find_name_col_idx(col_intervals: List[Tuple[int, int]]) -> int:
    if not col_intervals:
        return 0

    widths = [(x2 - x1, i) for i, (x1, x2) in enumerate(col_intervals)]
    img_cx = (col_intervals[0][0] + col_intervals[-1][1]) / 2

    left_cols = [(w, i) for w, i in widths if (col_intervals[i][0] + col_intervals[i][1]) / 2 < img_cx]
    if left_cols:
        best = max(left_cols, key=lambda t: t[0])
        logger.debug(f"  [name_col] Widest left-half column → idx={best[1]} x={col_intervals[best[1]]} w={best[0]}")
        return best[1]

    best = max(widths, key=lambda t: t[0])
    logger.debug(f"  [name_col] Widest overall column → idx={best[1]} x={col_intervals[best[1]]} w={best[0]}")
    return best[1]

def choose_question_columns(headers, col_intervals):
    name_col_idx = find_name_col_idx(col_intervals)
    question_cols = [
        i for i, h in enumerate(headers)
        if i != name_col_idx and re.fullmatch(r"\d{1,3}(?:-[A-Z])?", h)
    ]
    if not question_cols:
        question_cols = [i for i in range(len(col_intervals)) if i != name_col_idx]
    return name_col_idx, question_cols
