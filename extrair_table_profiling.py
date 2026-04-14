import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import time
import cProfile
import pstats
import io
import functools
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from collections import defaultdict
import concurrent.futures

# ============================================================
# CONFIGURAÇÕES
# ============================================================

IMAGE_PATH = "examples/image.png"
OUTPUT_CSV = "resultado_gabarito_v3.csv"
DEBUG_DIR = "debug_gabarito_v3"

TESSERACT_CMD = r"/opt/homebrew/bin/tesseract"
OCR_LANG = "por+eng"
OPTION_LABELS = ["B", "1", "2", "3"]

MIN_EXPECTED_QUESTION_COLS = 8
MIN_EXPECTED_STUDENT_ROWS = 3
MAX_ROTATION_CORRECTION_DEGREES = 8
GRID_CLUSTER_TOLERANCE = 12
ROW_HEIGHT_MIN = 45
COL_WIDTH_MIN = 25
EXPECTED_NUM_QUESTIONS = None

MIN_FILL_DENSITY = 0.05
MIN_INNER_DIFF = 5
MAX_SECOND_RATIO = 0.65

# ============================================================
# PROFILING — decorator + registro global
# ============================================================

_profile_stats: dict[str, dict] = defaultdict(lambda: {
    "calls": 0,
    "total_s": 0.0,
    "min_s": float("inf"),
    "max_s": 0.0,
})

def profile(fn: Callable) -> Callable:
    """Decorator leve: mede wall-time de cada chamada e acumula estatísticas."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        s = _profile_stats[fn.__name__]
        s["calls"]   += 1
        s["total_s"] += elapsed
        s["min_s"]    = min(s["min_s"], elapsed)
        s["max_s"]    = max(s["max_s"], elapsed)
        return result
    return wrapper

def print_profile_report():
    """Imprime tabela de profiling e relatório cProfile já capturado."""
    rows = []
    for name, s in _profile_stats.items():
        calls = s["calls"]
        total = s["total_s"]
        avg   = total / calls if calls else 0
        rows.append({
            "Função":          name,
            "Chamadas":        calls,
            "Total (ms)":      round(total * 1000, 2),
            "Média (ms)":      round(avg  * 1000, 3),
            "Mín (ms)":        round(s["min_s"] * 1000, 3),
            "Máx (ms)":        round(s["max_s"] * 1000, 3),
        })
    rows.sort(key=lambda r: -r["Total (ms)"])
    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("  RELATÓRIO DE PROFILING POR FUNÇÃO (wall-time, decorator)")
    print("=" * 78)
    print(df.to_string(index=False))
    df.to_csv("profile_por_funcao.csv", index=False, encoding="utf-8-sig")
    print("\n→ Salvo em profile_por_funcao.csv")

    # Resumo por camada
    camadas = {
        "1 · Pré-processamento": [
            "preprocess_document", "find_document_contour",
            "estimate_skew_angle", "four_point_transform", "rotate_image",
        ],
        "2 · Detecção de grade": [
            "get_table_structure", "binarize_for_grid", "detect_grid_masks",
            "extract_line_positions", "cluster_positions",
        ],
        "3 · OCR": [
            "ocr_headers", "ocr_names", "ocr_text_block",
            "run_ocr_boxes", "clean_question_header", "clean_name",
        ],
        "4 · Análise de células": [
            "detect_filled_option_v4", "preprocess_cell", "binarize_cell",
            "measure_band_density", "estimate_bubble_x",
            "component_candidates_in_band", "score_candidate",
            "fallback_circle_in_band",
        ],
        "5 · Montagem final": [
            "build_final_table", "identify_header_and_student_rows",
            "choose_question_columns",
        ],
    }

    print("\n" + "=" * 78)
    print("  RESUMO POR CAMADA")
    print("=" * 78)
    for camada, funcs in camadas.items():
        total_camada = sum(
            _profile_stats[f]["total_s"] for f in funcs if f in _profile_stats
        )
        calls_camada = sum(
            _profile_stats[f]["calls"] for f in funcs if f in _profile_stats
        )
        print(f"  {camada:<40} {total_camada*1000:>10.1f} ms  ({calls_camada} chamadas)")
    print()


# ============================================================
# ESTRUTURAS
# ============================================================

@dataclass
class OCRBox:
    text: str
    conf: float
    x: int; y: int; w: int; h: int

@dataclass
class CellResult:
    label: Optional[str]
    confidence: float
    density: float
    fill_detected: bool


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

Path(DEBUG_DIR).mkdir(exist_ok=True)


# ============================================================
# UTILITÁRIOS GERAIS
# ============================================================

def save_debug(name: str, img: np.ndarray):
    cv2.imwrite(str(Path(DEBUG_DIR) / name), img)

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

@profile
def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
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

@profile
def rotate_image(image: np.ndarray, angle: float, bg=255) -> np.ndarray:
    h, w   = image.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos    = abs(M[0, 0]);  sin = abs(M[0, 1])
    new_w  = int((h * sin) + (w * cos))
    new_h  = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    if len(image.shape) == 2:
        return cv2.warpAffine(image, M, (new_w, new_h), borderValue=bg)
    return cv2.warpAffine(image, M, (new_w, new_h), borderValue=(bg, bg, bg))

@profile
def cluster_positions(values: List[int], tolerance: int = 10) -> List[int]:
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

def crop(img: np.ndarray, x1, y1, x2, y2, pad=0) -> Optional[np.ndarray]:
    x1 = max(0, x1 + pad);  y1 = max(0, y1 + pad)
    x2 = min(img.shape[1], x2 - pad);  y2 = min(img.shape[0], y2 - pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


# ============================================================
# PRÉ-PROCESSAMENTO GEOMÉTRICO
# ============================================================

@profile
def find_document_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)
    edged    = cv2.Canny(blur, 40, 140)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area_img = gray.shape[0] * gray.shape[1]
    for cnt in contours[:20]:
        area = cv2.contourArea(cnt)
        if area < area_img * 0.25:
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

@profile
def estimate_skew_angle(gray: np.ndarray) -> float:
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 12
    )
    lines = cv2.HoughLinesP(
        bw, rho=1, theta=np.pi / 180, threshold=150,
        minLineLength=max(60, gray.shape[1] // 8), maxLineGap=20
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1;  dy = y2 - y1
        angle = 90.0 if dx == 0 else np.degrees(np.arctan2(dy, dx))
        if abs(angle) <= MAX_ROTATION_CORRECTION_DEGREES:
            angles.append(angle)
        elif abs(abs(angle) - 90) <= MAX_ROTATION_CORRECTION_DEGREES:
            angles.append(angle - 90 if angle > 0 else angle + 90)
    return float(np.median(angles)) if angles else 0.0

@profile
def preprocess_document(img: np.ndarray) -> np.ndarray:
    original = img.copy()
    gray     = to_gray(img)
    doc      = find_document_contour(gray)
    if doc is not None:
        img = four_point_transform(original, doc)
    gray2 = to_gray(img)
    angle = estimate_skew_angle(gray2)
    if abs(angle) > 0.15:
        img = rotate_image(img, angle * -1)
    save_debug("01_preprocessed.png", img)
    return img


# ============================================================
# DETECÇÃO DA GRADE
# ============================================================

@profile
def binarize_for_grid(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10
    )

@profile
def detect_grid_masks(bin_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bin_img.shape
    vk   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 28)))
    hk   = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 28), 1))
    vertical   = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, vk, iterations=1)
    horizontal = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, hk, iterations=1)
    return vertical, horizontal

@profile
def extract_line_positions(line_img: np.ndarray, axis: str) -> List[int]:
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if axis == "vertical":
            if h > 40: coords.append(x + w // 2)
        else:
            if w > 40: coords.append(y + h // 2)
    return cluster_positions(coords, tolerance=GRID_CLUSTER_TOLERANCE)

@profile
def get_table_structure(img: np.ndarray):
    gray = to_gray(img)
    bw   = binarize_for_grid(gray)
    vertical, horizontal = detect_grid_masks(bw)
    save_debug("02_grid_bw.png", bw)
    save_debug("03_vertical.png", vertical)
    save_debug("04_horizontal.png", horizontal)
    xs = cluster_positions(extract_line_positions(vertical,   "vertical"),   tolerance=GRID_CLUSTER_TOLERANCE)
    ys = cluster_positions(extract_line_positions(horizontal, "horizontal"), tolerance=GRID_CLUSTER_TOLERANCE)
    col_intervals = [(xs[i], xs[i+1]) for i in range(len(xs)-1) if xs[i+1]-xs[i] >= COL_WIDTH_MIN]
    row_intervals = [(ys[i], ys[i+1]) for i in range(len(ys)-1) if ys[i+1]-ys[i] >= ROW_HEIGHT_MIN]
    dbg = img.copy()
    for x in xs: cv2.line(dbg, (x, 0), (x, dbg.shape[0]-1), (0, 0, 255), 1)
    for y in ys: cv2.line(dbg, (0, y), (dbg.shape[1]-1, y), (255, 0, 0), 1)
    save_debug("05_detected_grid_lines.png", dbg)
    return xs, ys, col_intervals, row_intervals


# ============================================================
# OCR
# ============================================================

@profile
def run_ocr_boxes(img: np.ndarray, psm: int = 6,
                  whitelist: Optional[str] = None) -> List[OCRBox]:
    gray   = to_gray(img)
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    data  = pytesseract.image_to_data(gray, lang=OCR_LANG, config=config,
                                       output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data["text"])):
        text = normalize_whitespace(data["text"][i])
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if not text or conf < 0:
            continue
        boxes.append(OCRBox(
            text=text, conf=conf,
            x=safe_int(data["left"][i]), y=safe_int(data["top"][i]),
            w=safe_int(data["width"][i]), h=safe_int(data["height"][i])
        ))
    return boxes

@profile
def ocr_text_block(img: np.ndarray, psm: int = 6,
                   whitelist: Optional[str] = None) -> str:
    gray = to_gray(img)
    proc = cv2.GaussianBlur(gray, (3, 3), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    return normalize_whitespace(
        pytesseract.image_to_string(proc, lang=OCR_LANG, config=config)
    )

@profile
def clean_question_header(text: str) -> str:
    text = text.upper().replace(" ", "")
    text = text.replace("_", "-").replace(".", "-").replace("—", "-").replace("–", "-")
    m = re.search(r"(\d{1,3}(?:-[A-Z])?)", text)
    return m.group(1) if m else text

@profile
def clean_name(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"[^A-Za-zÀ-ÿ0-9\s\-']", "", text)
    return normalize_whitespace(text)


# ============================================================
# IDENTIFICAÇÃO DE REGIÕES
# ============================================================

@profile
def identify_header_and_student_rows(img, row_intervals):
    if not row_intervals:
        return None, []
    heights   = [y2 - y1 for y1, y2 in row_intervals]
    median_h  = np.median(heights)
    header_row = None
    student_rows = [iv for iv in row_intervals
                    if (iv[1]-iv[0]) >= max(55, median_h * 0.75)]
    for iv in row_intervals[:3]:
        if (iv[1]-iv[0]) < median_h * 0.9:
            header_row = iv
            break
    if header_row is None and row_intervals:
        header_row = row_intervals[0]
    return header_row, student_rows

@profile
def ocr_headers(img, col_intervals, header_row):
    y1, y2 = header_row
    # Cria uma lista pré-alocada para manter a ordem correta
    headers = [""] * len(col_intervals)
    
    def _process_header(idx, x1, x2):
        cell = crop(img, x1, y1, x2, y2, pad=3)
        if cell is None:
            return idx, ""
        
        text = clean_question_header(ocr_text_block(
            cell, psm=7,
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
        ))
        
        # Opcional: manter o debug visual
        vis = cell.copy()
        cv2.putText(vis, text, (5, min(20, vis.shape[0]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        save_debug(f"header_col_{idx+1}.png", vis)
        
        return idx, text

    # Executa o OCR em paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_header, idx, x1, x2) 
                   for idx, (x1, x2) in enumerate(col_intervals)]
        
        for future in concurrent.futures.as_completed(futures):
            idx, text = future.result()
            headers[idx] = text

    return headers


@profile
def ocr_names(img, name_col, candidate_rows):
    x1, x2 = name_col
    # Cria uma lista pré-alocada para manter a ordem correta
    names = [""] * len(candidate_rows)

    def _process_name(idx, y1, y2):
        cell = crop(img, x1, y1, x2, y2, pad=4)
        if cell is None:
            return idx, ""
        
        text = clean_name(ocr_text_block(cell, psm=6))
        if re.search(r"\bTOTAL\b", text.upper()):
            text = "__TOTAL__"
            
        # Opcional: manter o debug visual
        vis = cell.copy()
        cv2.putText(vis, (text or "(vazio)")[:40], (5, min(20, vis.shape[0]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        save_debug(f"name_row_{idx+1}.png", vis)
        
        return idx, text

    # Executa o OCR em paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process_name, i, y1, y2) 
                   for i, (y1, y2) in enumerate(candidate_rows)]
        
        for future in concurrent.futures.as_completed(futures):
            idx, text = future.result()
            names[idx] = text

    return names


# ============================================================
# ANÁLISE DE CÉLULAS
# ============================================================

@profile
def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(to_gray(cell), (5, 5), 0)

@profile
def binarize_cell(gray: np.ndarray) -> np.ndarray:
    bw     = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

@profile
def measure_band_density(bw: np.ndarray, y1: int, y2: int,
                          x_margin: int = 4) -> float:
    h = y2 - y1
    if h <= 0:
        return 0.0
    y_inner_start = y1 + 2
    y_inner_end   = y2 - 2
    band = bw[y_inner_start:y_inner_end, x_margin:-x_margin]
    if band.size == 0:
        return 0.0
    return float(cv2.countNonZero(band)) / band.size

@profile
def estimate_bubble_x(gray: np.ndarray, bw: np.ndarray) -> int:
    proj = bw.sum(axis=0).astype(np.float32)
    if proj.max() <= 0:
        return gray.shape[1] // 2
    k = max(5, gray.shape[1] // 15)
    if k % 2 == 0:
        k += 1
    proj_smooth = cv2.GaussianBlur(proj.reshape(1, -1), (k, 1), 0).reshape(-1)
    return int(np.argmax(proj_smooth))

@profile
def component_candidates_in_band(gray, bw, y1, y2, x_center,
                                   x_tol_ratio=0.22):
    band_bw   = bw[y1:y2, :]
    contours, _ = cv2.findContours(band_bw, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
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

@profile
def score_candidate(gray, cx, cy, r):
    r_inner = max(3, int(r * 0.60))
    r_ring  = max(r_inner + 1, int(r * 0.95))
    mask_inner = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_inner, (cx, cy), r_inner, 255, -1)
    mask_outer = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_outer, (cx, cy), r_ring,  255, -1)
    cv2.circle(mask_outer, (cx, cy), max(1, r_inner), 0, -1)
    return {
        "mean_inner": cv2.mean(gray, mask=mask_inner)[0],
        "mean_ring":  cv2.mean(gray, mask=mask_outer)[0],
    }

@profile
def fallback_circle_in_band(gray, y1, y2, x_center):
    band    = gray[y1:y2, :]
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

@profile
def detect_filled_option_v4(cell: np.ndarray,
                              debug_name: Optional[str] = None) -> CellResult:
    if cell is None or cell.size == 0:
        return CellResult(label=None, confidence=0.0, density=0.0,
                          fill_detected=False)

    gray = preprocess_cell(cell)
    bw   = binarize_cell(gray)

    H, W       = gray.shape
    x_center   = estimate_bubble_x(gray, bw)
    band_edges = np.linspace(0, H, 5).astype(int)

    densities = []
    for i in range(4):
        d = measure_band_density(bw, band_edges[i], band_edges[i+1], x_margin=4)
        densities.append(d)

    max_density = max(densities)

    if max_density < MIN_FILL_DENSITY:
        if debug_name:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, f"BRANCO d={max_density:.3f}", (5, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1,
                        cv2.LINE_AA)
            save_debug(f"{debug_name}_cell.png", vis)
        return CellResult(label=None, confidence=1.0, density=max_density,
                          fill_detected=False)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (x_center, 0), (x_center, H-1), (255, 0, 0), 1)

    band_results = []
    for i in range(4):
        y1 = band_edges[i];  y2 = band_edges[i+1]
        cv2.rectangle(vis, (0, y1), (W-1, y2-1), (220, 220, 220), 1)

        candidates = component_candidates_in_band(gray, bw, y1, y2, x_center)
        chosen = None;  chosen_score = None

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
            cy_    = int((y1 + y2) / 2)
            chosen = (x_center, cy_, max(6, min(W, y2-y1) // 6))
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
    if min(means) > np.mean(means) / 2:
        return CellResult(label=None, confidence=0, density=0,
                          fill_detected=False)
    order = np.argsort(means)

    best   = int(order[0]);  second = int(order[1])
    diff   = float(means[second] - means[best])
    confidence = max(0.0, min(1.0, diff / 50.0))

    best_density = band_results[best]["density"]
    if best_density < MIN_FILL_DENSITY:
        if debug_name:
            cv2.putText(vis, f"BRANCO-artefato d={best_density:.3f}",
                        (5, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 128, 255), 1, cv2.LINE_AA)
            save_debug(f"{debug_name}_cell.png", vis)
        return CellResult(label=None, confidence=confidence,
                          density=best_density, fill_detected=False)

    for d in band_results:
        cv2.circle(vis, (d["cx"], d["cy"]), d["r"], (0, 255, 0), 1)
        cv2.putText(vis, d["label"], (d["cx"]+d["r"]+3, d["cy"]+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                    cv2.LINE_AA)

    chosen_label = band_results[best]["label"]
    cv2.putText(vis,
                f"pick={chosen_label} conf={confidence:.2f} d={best_density:.2f}",
                (5, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
                cv2.LINE_AA)

    if debug_name:
        save_debug(f"{debug_name}_cell.png", vis)

    if diff < MIN_INNER_DIFF:
        return CellResult(label=None, confidence=confidence,
                          density=best_density, fill_detected=True)

    return CellResult(label=chosen_label, confidence=confidence,
                      density=best_density, fill_detected=True)


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def choose_question_columns(headers, col_intervals):
    question_cols = [i for i, h in enumerate(headers)
                     if re.fullmatch(r"\d{1,3}(?:-[A-Z])?", h)]
    if not question_cols and len(col_intervals) >= 2:
        question_cols = list(range(1, len(col_intervals)))
    return 0, question_cols

@profile
def build_final_table(img: np.ndarray):
    xs, ys, col_intervals, row_intervals = get_table_structure(img)

    if len(col_intervals) < 2:
        raise RuntimeError("Não foi possível detectar colunas suficientes.")
    if len(row_intervals) < 2:
        raise RuntimeError("Não foi possível detectar linhas suficientes.")

    header_row, candidate_student_rows = identify_header_and_student_rows(
        img, row_intervals)
    if header_row is None:
        raise RuntimeError("Não foi possível identificar a linha de cabeçalho.")

    headers = ocr_headers(img, col_intervals, header_row)
    name_col_idx, question_col_indices = choose_question_columns(headers, col_intervals)

    if EXPECTED_NUM_QUESTIONS is not None:
        question_col_indices = question_col_indices[:EXPECTED_NUM_QUESTIONS]

    if len(question_col_indices) < MIN_EXPECTED_QUESTION_COLS:
        limit = 1 + max(MIN_EXPECTED_QUESTION_COLS, EXPECTED_NUM_QUESTIONS or 0)
        question_col_indices = list(range(1, min(len(col_intervals), limit)))

    name_col  = col_intervals[name_col_idx]
    raw_names = ocr_names(img, name_col, candidate_student_rows)

    student_rows, student_names = [], []
    for interval, name in zip(candidate_student_rows, raw_names):
        if not name or name == "__TOTAL__" or len(name) < 3:
            continue
        student_rows.append(interval)
        student_names.append(name)

    if len(student_rows) < MIN_EXPECTED_STUDENT_ROWS:
        fallback      = (candidate_student_rows[-5:]
                         if len(candidate_student_rows) >= 5
                         else candidate_student_rows)
        student_rows  = fallback
        student_names = raw_names[-len(fallback):]

    question_headers = []
    for idx in question_col_indices:
        h = headers[idx] if idx < len(headers) else ""
        if not re.fullmatch(r"\d{1,3}(?:-[A-Z])?", h or ""):
            h = f"Q{len(question_headers)+1}"
        question_headers.append(h)

    seen = {}
    final_question_headers = []
    for h in question_headers:
        if h not in seen:
            seen[h] = 1;  final_question_headers.append(h)
        else:
            seen[h] += 1; final_question_headers.append(f"{h}_{seen[h]}")

    records, conf_records, density_records = [], [], []

    for row_i, ((y1, y2), student_name) in enumerate(
            zip(student_rows, student_names)):
        rec   = {"Nome": student_name}
        c_rec = {"Nome": student_name}
        d_rec = {"Nome": student_name}

        for q_pos, col_idx in enumerate(question_col_indices):
            x1, x2 = col_intervals[col_idx]
            q_name  = final_question_headers[q_pos]
            cell    = crop(img, x1, y1, x2, y2, pad=4)
            result  = detect_filled_option_v4(
                cell, debug_name=f"row{row_i+1}_{q_name}")
            rec[q_name]   = result.label if result.label is not None else ""
            c_rec[q_name] = round(result.confidence, 3)
            d_rec[q_name] = round(result.density, 3)

        records.append(rec)
        conf_records.append(c_rec)
        density_records.append(d_rec)

    df         = pd.DataFrame(records)
    df_conf    = pd.DataFrame(conf_records)
    df_density = pd.DataFrame(density_records)

    return df, df_conf, df_density, {
        "headers_raw":              headers,
        "question_columns":         question_col_indices,
        "question_headers_final":   final_question_headers,
        "student_names":            student_names,
    }


# ============================================================
# MAIN com cProfile + relatório por função
# ============================================================

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(
            f"Não foi possível abrir a imagem: {IMAGE_PATH}")

    # ── cProfile: captura tudo (inclusive OpenCV e Tesseract internos)
    profiler = cProfile.Profile()
    profiler.enable()

    t_total_start = time.perf_counter()
    pre = preprocess_document(img)
    df, df_conf, df_density, meta = build_final_table(pre)
    t_total_end = time.perf_counter()

    profiler.disable()

    # ── Salva CSVs de resultado
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    conf_path    = OUTPUT_CSV.replace(".csv", "_confianca.csv")
    density_path = OUTPUT_CSV.replace(".csv", "_densidade.csv")
    df_conf.to_csv(conf_path,    index=False, encoding="utf-8-sig")
    df_density.to_csv(density_path, index=False, encoding="utf-8-sig")

    # ── Relatório do decorator (wall-time por função)
    print_profile_report()

    # ── Relatório cProfile (top 30 por tempo cumulativo)
    print("\n" + "=" * 78)
    print("  RELATÓRIO cProfile — top 30 funções por tempo cumulativo")
    print("=" * 78)
    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)
    ps.sort_stats("cumulative")
    ps.print_stats(30)
    print(stream.getvalue())

    # Salva cProfile completo
    profiler.dump_stats("profile_cprofile.prof")
    print("→ Arquivo .prof salvo em profile_cprofile.prof")
    print(f"  (abra com: python -m snakeviz profile_cprofile.prof)")

    # ── Tempo total
    print(f"\n  Tempo total de execução: "
          f"{(t_total_end - t_total_start)*1000:.0f} ms\n")

    # ── Resultados
    print("Cabeçalhos OCR brutos:")
    print(meta["headers_raw"])
    print("\nCabeçalhos finais de questões:")
    print(meta["question_headers_final"])
    print("\nNomes detectados:")
    for n in meta["student_names"]:
        print("-", n)
    print("\nTabela extraída:")
    print(df.to_string(index=False))

    print(f"\nCSV principal:  {OUTPUT_CSV}")
    print(f"CSV confiança:  {conf_path}")
    print(f"CSV densidade:  {density_path}")
    print(f"Debug:          {DEBUG_DIR}/")


if __name__ == "__main__":
    main()