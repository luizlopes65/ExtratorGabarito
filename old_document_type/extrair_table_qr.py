# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import pytesseract
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
# pyrefly: ignore [missing-import]
from pyzbar import pyzbar
import json

# ============================================================
# CONFIGURAÇÕES
# ============================================================

IMAGE_PATH = "examples/page_002.png"
OUTPUT_CSV = "resultados/resultado_gabarito_qr.csv"
DEBUG_DIR = "debug/debug_gabarito_qr"
ENABLE_DEBUG_IMAGES = False


TESSERACT_CMD = r"/opt/homebrew/bin/tesseract"

OCR_LANG = "por+eng"

# QR code data (can be set externally via main.py)
# Placeholder QR data for testing (format: "q1;q2;q3.student1;student2;student3")
QR_DATA = "1;2;3;4;5;6;7;8;9;10.João Silva;Maria Santos;Pedro Oliveira;Ana Costa;Carlos Souza"

OPTION_LABELS = ["B", "1", "2", "3"]

MIN_EXPECTED_QUESTION_COLS = 5
MIN_EXPECTED_STUDENT_ROWS = 3

GRID_CLUSTER_TOLERANCE = 12
ROW_HEIGHT_MIN = 20
COL_WIDTH_MIN = 25

EXPECTED_NUM_QUESTIONS = None


EXPECTED_QUESTION_HEADERS = []

MIN_FILL_DENSITY = 0.03
MIN_INNER_DIFF = 5
MAX_SECOND_RATIO = 0.65


NARROW_COL_RATIO: float = 0.40   # drop if width < median_width * this

WIDE_NAME_COL_RATIO: float = 1.5  # name col width >= median * this (informational)

# ============================================================
# ESTRUTURAS
# ============================================================

@dataclass
class OCRBox:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int

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

# Path(DEBUG_DIR).mkdir(exist_ok=True)  # Now handled inside save_debug

# ============================================================
# UTILITÁRIOS GERAIS
# ============================================================

def save_debug(name: str, img: np.ndarray):
    if not ENABLE_DEBUG_IMAGES:
        return
    Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(DEBUG_DIR) / name), img)

def clear_debug_dir(debug_dir: str):
    if not ENABLE_DEBUG_IMAGES:
        return
    debug_path = Path(debug_dir)
    if debug_path.exists():
        shutil.rmtree(debug_path)
    debug_path.mkdir(parents=True, exist_ok=True)

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

# ============================================================
# PRÉ-PROCESSAMENTO GEOMÉTRICO
# ============================================================

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

def preprocess_document(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    doc = find_document_contour(gray)
    if doc is not None:
        img = four_point_transform(img, doc)
    save_debug("01_preprocessed.png", img)
    return img

# ============================================================
# DETECÇÃO DA GRADE
# ============================================================

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


# ============================================================
# FIX 1: Filter spurious margin columns
# ============================================================

def filter_margin_columns(col_intervals: List[Tuple[int, int]], img_width: int, expected_col_count: int = 0) -> List[Tuple[int, int]]:
    """
    Remove columns that are artifacts of vertical margin text (URL, watermark, etc.).

    Strategy:
    1. Compute the median column width among all intervals.
    2. Drop any column narrower than NARROW_COL_RATIO * median_width.
    3. Also drop columns whose centre is in the outermost margin of the image width,
       which is where rotated margin text typically lives.
    4. If expected_col_count is provided (from QR code), preserve that many columns
       even if they're in the margin area.
    
    Args:
        col_intervals: List of (x1, x2) tuples representing column boundaries
        img_width: Width of the image in pixels
        expected_col_count: Expected number of columns from QR code (0 = not provided)
    """
    if not col_intervals:
        return col_intervals

    widths = [x2 - x1 for x1, x2 in col_intervals]
    median_w = float(np.median(widths))
    min_w = NARROW_COL_RATIO * median_w

    # Reduce margin to 1.5% to avoid filtering valid rightmost columns
    margin_px = int(img_width * 0.015)

    filtered = []
    for x1, x2 in col_intervals:
        w = x2 - x1
        cx = (x1 + x2) / 2
        if w < min_w:
            print(f"  [filter_cols] Dropping narrow column x={x1}-{x2} (w={w:.0f} < {min_w:.0f})")
            continue
        if cx < margin_px or cx > img_width - margin_px:
            print(f"  [filter_cols] Dropping margin column x={x1}-{x2} (cx={cx:.0f})")
            continue
        filtered.append((x1, x2))

    # If we have expected column count from QR and we filtered too many, keep the original
    if expected_col_count > 0 and len(filtered) < expected_col_count:
        print(f"  [filter_cols] WARNING: Filtered to {len(filtered)} columns but QR expects {expected_col_count}")
        print(f"  [filter_cols] Reverting to original {len(col_intervals)} columns to preserve QR data")
        return col_intervals

    if not filtered:
        print("  [filter_cols] WARNING: all columns were filtered — reverting to original")
        return col_intervals

    return filtered


# ============================================================
# FIX 2: Robust header-row identification
# ============================================================

def identify_header_and_student_rows(img, row_intervals):
    """
    Identify the header row (question numbers) and student rows.

    Improved logic vs original:
    - The header row is the row whose OCR content matches question-number
      patterns (digits / digits-letter) across most columns.
    - Fallback: shortest row among the first few candidates — but now we
      look up to the first 5 rows (not 3), because Image 1 has:
        row0 = very tall QR block
        row1 = header (short, has "Nome do Aluno / 1 / 2 / 3 …")
        row2+ = students (taller)
      The original code could misidentify row1 if the QR block was split.
    - Student rows: all rows after the header that pass the height test.
    """
    if not row_intervals:
        return None, []
    if len(row_intervals) < 2:
        return row_intervals[0], []

    heights = [y2 - y1 for y1, y2 in row_intervals]
    median_h = float(np.median(heights))

    # ---------- try OCR-based detection first ----------
    # Quick OCR of first column of each candidate row to find "Nome do Aluno"
    gray = to_gray(img)
    img_w = img.shape[1]
    # Use roughly the left 30% of the image as the name-column region for scanning
    scan_x2 = int(img_w * 0.30)

    header_row_idx = None
    for i, (y1, y2) in enumerate(row_intervals[:6]):
        cell = crop(gray, 0, y1, scan_x2, y2, pad=2)
        if cell is None:
            continue
        try:
            txt = pytesseract.image_to_string(
                cell, lang=OCR_LANG,
                config="--oem 3 --psm 6"
            ).upper()
        except Exception:
            txt = ""
        if re.search(r"NOME", txt):
            header_row_idx = i
            print(f"  [header] Found 'Nome' in row {i} via OCR → header_row_idx={i}")
            break

    # ---------- fallback: shortest row among first N candidates ----------
    if header_row_idx is None:
        n_candidates = min(5, len(row_intervals))
        candidates = [(y2 - y1, i) for i, (y1, y2) in enumerate(row_intervals[:n_candidates])]
        candidates.sort()
        header_row_idx = candidates[0][1]
        print(f"  [header] OCR fallback: shortest row among first {n_candidates} → idx={header_row_idx}")

    header_row = row_intervals[header_row_idx]

    # Student rows: everything after the header that is tall enough
    min_student_h = max(40, median_h * 0.60)
    student_rows = []
    for iv in row_intervals[header_row_idx + 1:]:
        h = iv[1] - iv[0]
        if h >= min_student_h:
            student_rows.append(iv)

    return header_row, student_rows


# ============================================================
# Main grid structure (uses both fixes)
# ============================================================

def get_table_structure(img: np.ndarray, expected_question_count: int = 0):
    """
    Extract table structure from image.
    
    Args:
        img: Input image
        expected_question_count: Expected number of question columns from QR code (0 = unknown)
                                This helps preserve rightmost columns that might otherwise be filtered
    """
    gray = to_gray(img)
    bw = binarize_for_grid(gray)
    vertical, horizontal = detect_grid_masks(bw)
    save_debug("02_grid_bw.png", bw)
    save_debug("03_vertical.png", vertical)
    save_debug("04_horizontal.png", horizontal)

    xs = cluster_positions(extract_line_positions(vertical, "vertical"), tolerance=GRID_CLUSTER_TOLERANCE)
    ys = cluster_positions(extract_line_positions(horizontal, "horizontal"), tolerance=GRID_CLUSTER_TOLERANCE)

    raw_col_intervals = [(xs[i], xs[i+1]) for i in range(len(xs)-1) if xs[i+1]-xs[i] >= COL_WIDTH_MIN]

    # ── FIX 1: remove margin/artifact columns ──
    # Add 1 to expected count to account for the name/ID column
    expected_col_count = expected_question_count + 1 if expected_question_count > 0 else 0
    col_intervals = filter_margin_columns(raw_col_intervals, img.shape[1], expected_col_count)

    all_row_heights = [(ys[i], ys[i+1], ys[i+1]-ys[i]) for i in range(len(ys)-1)]
    print(f"\nAll detected row positions (y1, y2, height):")
    for y1, y2, h in all_row_heights:
        status = "✓ KEPT" if h >= ROW_HEIGHT_MIN else f"✗ FILTERED (< {ROW_HEIGHT_MIN}px)"
        print(f"  Row {y1:4d}-{y2:4d}: {h:3d}px {status}")

    row_intervals = [(ys[i], ys[i+1]) for i in range(len(ys)-1) if ys[i+1]-ys[i] >= ROW_HEIGHT_MIN]

    dbg = img.copy()
    for x in xs:
        cv2.line(dbg, (x, 0), (x, dbg.shape[0]-1), (0, 0, 255), 1)
    for y in ys:
        cv2.line(dbg, (0, y), (dbg.shape[1]-1, y), (255, 0, 0), 1)
    save_debug("05_detected_grid_lines.png", dbg)

    return xs, ys, col_intervals, row_intervals

# ============================================================
# OCR
# ============================================================

def run_ocr_boxes(img: np.ndarray, psm: int = 6, whitelist: Optional[str] = None) -> List[OCRBox]:
    gray = to_gray(img)
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    data = pytesseract.image_to_data(gray, lang=OCR_LANG, config=config, output_type=pytesseract.Output.DICT)
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

def ocr_text_block(img: np.ndarray, psm: int = 6, whitelist: Optional[str] = None) -> str:
    gray = to_gray(img)
    proc = cv2.GaussianBlur(gray, (3, 3), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    result = normalize_whitespace(pytesseract.image_to_string(proc, lang=OCR_LANG, config=config))
    if len(result) < 2 or not re.search(r"\d", result):
        result_raw = normalize_whitespace(pytesseract.image_to_string(gray, lang=OCR_LANG, config=config))
        if len(result_raw) > len(result):
            result = result_raw
    return result

def normalize_question_token(text: str) -> str:
    text = (text or "").upper().strip()
    text = text.replace("_", "-").replace(".", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(\d+)([A-Z])$", r"\1-\2", text)
    return text

def extract_question_number_and_suffix(text: str) -> Tuple[Optional[int], Optional[str]]:
    token = normalize_question_token(text)
    m = re.fullmatch(r"(\d{1,3})(?:-([A-Z]))?", token)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)

def clean_question_header(text: str) -> str:
    text = normalize_question_token(text)

    matches = re.findall(r"\d{1,3}(?:-[A-Z])?", text)
    if not matches:
        text2 = text.replace("G", "6").replace("O", "0").replace("I", "1").replace("S", "5")
        text2 = re.sub(r"(\d+)([A-Z])$", r"\1-\2", text2)
        matches = re.findall(r"\d{1,3}(?:-[A-Z])?", text2)
        if not matches:
            return text

    matches_with_suffix = [m for m in matches if "-" in m]
    if matches_with_suffix:
        return matches_with_suffix[-1]

    result = matches[-1]
    if len(result) == 2 and result[0] == result[1]:
        result = result[0]
    return result

def clean_name(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"[^A-Za-zÀ-ÿ0-9\s\-']", "", text)
    return normalize_whitespace(text)

def question_header_similarity_score(observed: str, expected: str) -> int:
    observed = normalize_question_token(observed)
    expected = normalize_question_token(expected)

    if observed == expected:
        return 0

    obs_num, obs_suffix = extract_question_number_and_suffix(observed)
    exp_num, exp_suffix = extract_question_number_and_suffix(expected)

    score = 0

    if obs_num is None:
        score += 100
    else:
        score += abs(obs_num - exp_num) * 10

    if obs_suffix != exp_suffix:
        if obs_suffix is None or exp_suffix is None:
            score += 8
        else:
            score += abs(ord(obs_suffix) - ord(exp_suffix)) + 3

    obs_digits = "".join(re.findall(r"\d", observed))
    exp_digits = "".join(re.findall(r"\d", expected))
    if obs_digits and exp_digits and obs_digits != exp_digits:
        score += 12

    if "-" in observed and "-" not in expected:
        score += 2
    if "-" not in observed and "-" in expected:
        score += 4

    if observed and expected and observed[0] != expected[0]:
        score += 1

    return score

def reconcile_question_headers(question_headers: List[str],
                               expected_headers: Optional[List[str]] = None) -> List[str]:
    """
    Reconcilia cabeçalhos OCR com uma lista canônica opcional.

    Regras:
    - Se expected_headers estiver vazia, mantém o comportamento atual.
    - Se expected_headers existir e tiver tamanho compatível, usa alinhamento por posição
      para substituir/corrigir cabeçalhos OCR ruins.
    - Se o OCR estiver claramente bom, preserva o valor OCR.
    - Se estiver ruim/ausente, usa o valor esperado da mesma coluna.
    """
    final_headers = []

    if expected_headers:
        normalized_expected = [normalize_question_token(h) for h in expected_headers if normalize_question_token(h)]
        if len(normalized_expected) >= len(question_headers):
            for i, observed in enumerate(question_headers):
                expected = normalized_expected[i]
                observed_norm = normalize_question_token(observed)

                if not observed_norm:
                    final_headers.append(expected)
                    continue

                if not re.search(r"\d", observed_norm):
                    final_headers.append(expected)
                    continue

                score = question_header_similarity_score(observed_norm, expected)

                # Preserve OCR only when it is reasonably close to the expected token.
                if score <= 8:
                    final_headers.append(observed_norm)
                else:
                    final_headers.append(expected)
            return final_headers

    # Fallback atual: inferência baseada no padrão anterior
    for i, h in enumerate(question_headers):
        h = normalize_question_token(h)

        if i > 0 and re.match(r"^\d+$", h):
            prev = final_headers[-1]
            m = re.match(r"(\d+)-([A-Z])", prev)
            if m and m.group(1) == h:
                next_letter = chr(ord(m.group(2)) + 1)
                h = f"{h}-{next_letter}"
                final_headers.append(h)
                continue

        if len(h) <= 2 and h.isdigit() and i > 0:
            prev = final_headers[-1]
            m = re.match(r"(\d+)-([A-Z])", prev)
            if m:
                base_num = m.group(1)
                prev_letter = m.group(2)
                if h == base_num or h == base_num[-1]:
                    next_letter = chr(ord(prev_letter) + 1)
                    h = f"{base_num}-{next_letter}"
                elif len(h) == 1 and i + 1 < len(question_headers):
                    next_h = normalize_question_token(question_headers[i + 1])
                    if next_h.isdigit():
                        next_num = int(next_h)
                        base_int = int(base_num)
                        curr_int = int(h)
                        if abs(next_num - base_int) <= 1 and curr_int != base_int + 1:
                            next_letter = chr(ord(prev_letter) + 1)
                            h = f"{base_num}-{next_letter}"

        final_headers.append(h)

    return final_headers

# ============================================================
# IDENTIFICAÇÃO DE REGIÕES
# ============================================================

# ============================================================
# QR CODE DETECTION AND PARSING
# ============================================================

def detect_qr_code(img: np.ndarray) -> Optional[str]:
    """
    Detect and decode QR code from the top-left corner of the image.
    Uses multiple preprocessing techniques for robust detection.
    Enhanced with additional methods from teste_qr_extraction.py
    
    Returns:
        QR code data as string, or None if not found
    """
    # Try multiple region sizes (30%, 40%, 50%)
    h, w = img.shape[:2]
    regions_to_try = [
        ("Top-Left 30%", img[0:int(h*0.3), 0:int(w*0.3)]),
        ("Top-Left 40%", img[0:int(h*0.4), 0:int(w*0.4)]),
        ("Top-Left 50%", img[0:int(h*0.5), 0:int(w*0.5)]),
    ]
    
    for region_name, qr_region in regions_to_try:
        # Convert to grayscale for preprocessing
        if len(qr_region.shape) == 3:
            gray_region = cv2.cvtColor(qr_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = qr_region
        
        # Try multiple preprocessing techniques (from teste_qr_extraction.py)
        # IMPORTANT: Try original color image first, as pyzbar works best with it
        preprocessing_methods = [
            ("Original", qr_region),  # Try color image first
            ("Grayscale", gray_region),
            ("Binary Threshold", cv2.threshold(gray_region, 127, 255, cv2.THRESH_BINARY)[1]),
            ("Adaptive Threshold", cv2.adaptiveThreshold(gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("Otsu Threshold", cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("Inverted", cv2.bitwise_not(gray_region)),
            ("Contrast Enhanced", cv2.equalizeHist(gray_region)),
        ]
    
        qr_codes = None
        successful_method = None
        
        for method_name, processed in preprocessing_methods:
            qr_codes = pyzbar.decode(processed)
            if qr_codes:
                successful_method = method_name
                break
        
        if qr_codes:
            qr_data = qr_codes[0].data.decode('utf-8')
            print(f"[QR] Detected QR code in {region_name} using {successful_method}: {qr_data}")
            
            # Save debug image showing QR detection
            if len(qr_region.shape) == 3:
                qr_debug = qr_region.copy()
            else:
                qr_debug = cv2.cvtColor(qr_region.copy(), cv2.COLOR_GRAY2BGR)
            for qr in qr_codes:
                points = qr.polygon
                if len(points) == 4:
                    pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                    cv2.polylines(qr_debug, [pts], True, (0, 255, 0), 3)
            save_debug("qr_code_detected.png", qr_debug)
            
            return qr_data
    
    print("[QR] No QR code detected in any region (tried 30%, 40%, 50%)")
    return None


def parse_qr_data(qr_data: str) -> Tuple[List[str], List[str], List[str], dict]:
    """
    Parse QR code data to extract question numbers, student IDs, and metadata.
    
    Expected format (new):
    "F;ID_PROF;ID_ESCOLA;ANO_ESCOLAR;BIMESTRE;DATA;QUESTÕES_DETALHADAS;IDS_ALUNOS;PÁGINA"
    Example: "F;P217;E25;Primeiro;Primeiro;2026_04_13;1,2,3,47,5,6A,6B;A3859,A3860,A3861,...;2"
    
    Format breakdown:
    - Field 0: Type (F)
    - Field 1: ID_PROF (Professor ID)
    - Field 2: ID_ESCOLA (School ID)
    - Field 3: ANO_ESCOLAR (School year)
    - Field 4: BIMESTRE (Bimester)
    - Field 5: DATA (Date)
    - Field 6: QUESTÕES_DETALHADAS (Questions, comma-separated)
    - Field 7: IDS_ALUNOS (Student IDs, comma-separated)
    - Field 8: PÁGINA (Page number)
    
    Legacy format (fallback):
    "question1;question2;question3.student1;student2;student3"
    
    Args:
        qr_data: Raw QR code string
        
    Returns:
        Tuple of (question_headers, student_ids, student_names, metadata_dict)
        metadata_dict contains: id_prof, id_escola, ano_escolar, bimestre, data, pagina
    """
    metadata = {
        'id_prof': None,
        'id_escola': None,
        'ano_escolar': None,
        'bimestre': None,
        'data': None,
        'pagina': None
    }
    
    try:
        # Try new format first (semicolon-separated fields)
        if ';' in qr_data and ',' in qr_data:
            parts = qr_data.split(';')
            
            # New format should have at least 8 fields
            if len(parts) >= 8:
                # Extract metadata fields
                metadata['id_prof'] = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
                metadata['id_escola'] = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
                metadata['ano_escolar'] = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
                metadata['bimestre'] = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
                metadata['data'] = parts[5].strip() if len(parts) > 5 and parts[5].strip() else None
                
                # Field 6: Questions (comma-separated)
                questions_str = parts[6].strip()
                questions = [q.strip() for q in questions_str.split(',') if q.strip()]
                
                # Field 7: Student IDs (comma-separated)
                ids_str = parts[7].strip()
                student_ids = [sid.strip() for sid in ids_str.split(',') if sid.strip()]
                
                # Field 8: Page number (if exists)
                if len(parts) > 8:
                    metadata['pagina'] = parts[8].strip() if parts[8].strip() else None
                
                print(f"[QR Parse] New format detected")
                print(f"[QR Parse] ID_PROF: {metadata['id_prof']}")
                print(f"[QR Parse] ID_ESCOLA: {metadata['id_escola']}")
                print(f"[QR Parse] ANO_ESCOLAR: {metadata['ano_escolar']}")
                print(f"[QR Parse] BIMESTRE: {metadata['bimestre']}")
                print(f"[QR Parse] DATA: {metadata['data']}")
                print(f"[QR Parse] PÁGINA: {metadata['pagina']}")
                print(f"[QR Parse] Extracted {len(questions)} questions: {questions}")
                print(f"[QR Parse] Extracted {len(student_ids)} student IDs: {student_ids}")
                
                # Return empty list for student names (we only have IDs)
                return questions, student_ids, [], metadata
        
        # Fallback to legacy format: "questions.students"
        if '.' in qr_data:
            parts = qr_data.split('.', 1)
            
            if len(parts) != 2:
                print(f"[QR Parse] Invalid legacy format: expected 'questions.students', got: {qr_data}")
                return [], [], [], metadata
            
            questions_str, students_str = parts
            
            # Parse questions (separated by ;)
            questions = [q.strip() for q in questions_str.split(';') if q.strip()]
            
            # Parse student names (separated by ;)
            students = [s.strip() for s in students_str.split(';') if s.strip()]
            
            print(f"[QR Parse] Legacy format detected")
            print(f"[QR Parse] Extracted {len(questions)} questions: {questions}")
            print(f"[QR Parse] Extracted {len(students)} students: {students}")
            
            # Return empty list for IDs (legacy format doesn't have them)
            return questions, [], students, metadata
        
        print(f"[QR Parse] Unrecognized format: {qr_data}")
        return [], [], [], metadata
        
    except Exception as e:
        print(f"[QR Parse] Error parsing QR data: {e}")
        import traceback
        traceback.print_exc()
        return [], [], [], metadata


def get_qr_headers(img, col_intervals, header_row, qr_questions: List[str]) -> List[str]:
    """
    Generate headers for columns based on QR code data.
    
    Args:
        img: Image (unused, kept for compatibility)
        col_intervals: List of column intervals
        header_row: Header row interval (unused, kept for compatibility)
        qr_questions: List of question numbers from QR code
        
    Returns:
        List of headers matching column count
    """
    headers = []
    
    # First column is always the name column (empty header)
    headers.append("")
    
    # Remaining columns are question columns
    for idx in range(1, len(col_intervals)):
        if idx - 1 < len(qr_questions):
            headers.append(qr_questions[idx - 1])
        else:
            # If we run out of QR questions, generate placeholder
            headers.append(f"Q{idx}")
    
    print(f"[QR Headers] Generated {len(headers)} headers: {headers}")
    
    # Save debug visualization
    y1, y2 = header_row
    for idx, (x1, x2) in enumerate(col_intervals):
        cell = crop(img, x1, y1, x2, y2, pad=3)
        if cell is not None:
            vis = cv2.cvtColor(cell.copy(), cv2.COLOR_GRAY2BGR) if len(cell.shape) == 2 else cell.copy()
            text = headers[idx] if idx < len(headers) else ""
            cv2.putText(vis, text, (5, min(20, vis.shape[0]-5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            save_debug(f"header_col_{idx+1}_qr.png", vis)
    
    return headers


def get_qr_names(img, name_col, candidate_rows, qr_students: List[str]) -> List[str]:
    """
    Generate student names for rows based on QR code data.
    
    Args:
        img: Image (unused, kept for compatibility)
        name_col: Name column interval (unused, kept for compatibility)
        candidate_rows: List of row intervals
        qr_students: List of student names from QR code
        
    Returns:
        List of student names matching row count
    """
    names = []
    
    for i, (y1, y2) in enumerate(candidate_rows):
        if i < len(qr_students):
            names.append(qr_students[i])
        else:
            # If we run out of QR students, use placeholder
            names.append(f"Student_{i+1}")
    
    print(f"[QR Names] Generated {len(names)} names: {names}")
    
    # Save debug visualization
    x1, x2 = name_col
    for i, (y1, y2) in enumerate(candidate_rows):
        cell = crop(img, x1, y1, x2, y2, pad=4)
        if cell is not None:
            vis = cv2.cvtColor(cell.copy(), cv2.COLOR_GRAY2BGR) if len(cell.shape) == 2 else cell.copy()
            text = names[i] if i < len(names) else ""
            cv2.putText(vis, (text or "(vazio)")[:40], (5, min(20, vis.shape[0]-5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            save_debug(f"name_row_{i+1}_qr.png", vis)
    
    return names


# ============================================================
# LEGACY OCR FUNCTIONS (kept for reference/fallback)
# ============================================================

def ocr_headers(img, col_intervals, header_row, expected_headers: Optional[List[str]] = None):
    """LEGACY: OCR-based header extraction (replaced by QR code)"""
    y1, y2 = header_row
    headers = []
    for idx, (x1, x2) in enumerate(col_intervals):
        cell = crop(img, x1, y1, x2, y2, pad=3)
        text = clean_question_header(ocr_text_block(
            cell, psm=6,
            whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
        ))
        headers.append(text)
        vis = cell.copy()
        cv2.putText(vis, text, (5, min(20, vis.shape[0]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        save_debug(f"header_col_{idx+1}.png", vis)
    return headers

def ocr_names(img, name_col, candidate_rows):
    """LEGACY: OCR-based name extraction (replaced by QR code)"""
    names = []
    x1, x2 = name_col
    for i, (y1, y2) in enumerate(candidate_rows):
        cell = crop(img, x1, y1, x2, y2, pad=4)
        if cell is None:
            names.append("")
            continue
        text = clean_name(ocr_text_block(cell, psm=6))
        if re.search(r"\bTOTAL\b", text.upper()):
            names.append("__TOTAL__")
        else:
            names.append(text)
        vis = cell.copy()
        cv2.putText(vis, (text or "(vazio)")[:40], (5, min(20, vis.shape[0]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        save_debug(f"name_row_{i+1}.png", vis)
    return names

# ============================================================
# DETECÇÃO DE MARCAÇÃO (unchanged from v3/v4)
# ============================================================

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

def detect_filled_option_v4(cell: np.ndarray, debug_name: Optional[str] = None) -> CellResult:
    if cell is None or cell.size == 0:
        if debug_name:
            placeholder = np.zeros((40, 60, 3), dtype=np.uint8)
            placeholder[:] = (0, 0, 180)
            cv2.putText(placeholder, "EMPTY", (2, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            save_debug(f"{debug_name}_cell.png", placeholder)
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
        if debug_name:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, f"BRANCO d={max_density:.3f}", (5, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1, cv2.LINE_AA)
            save_debug(f"{debug_name}_cell.png", vis)
        return CellResult(label=None, confidence=1.0, density=max_density, fill_detected=False)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (x_center, 0), (x_center, H-1), (255, 0, 0), 1)

    band_results = []
    for i in range(4):
        y1 = band_edges[i]
        y2 = band_edges[i+1]
        cv2.rectangle(vis, (0, y1), (W-1, y2-1), (220, 220, 220), 1)

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
    # Check if all bubbles have similar intensity (no clear winner)
    # Changed from avg/2 to avg*0.7 to be more lenient
    if min(means) > np.mean(means) * 0.8:
        if debug_name:
            for d in band_results:
                cv2.circle(vis, (d["cx"], d["cy"]), d["r"], (0, 255, 0), 1)
                cv2.putText(vis, d["label"], (d["cx"]+d["r"]+3, d["cy"]+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"NO CLEAR WINNER means={[round(m,1) for m in means]}", (5, H-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 255), 1, cv2.LINE_AA)
            save_debug(f"{debug_name}_cell.png", vis)
        return CellResult(label=None, confidence=0,
                          density=max_density, fill_detected=False)
    order = np.argsort(means)

    best   = int(order[0])
    second = int(order[1])
    diff   = float(means[second] - means[best])
    confidence = max(0.0, min(1.0, diff / 50.0))

    best_density = band_results[best]["density"]
    if best_density < MIN_FILL_DENSITY:
        if debug_name:
            cv2.putText(vis, f"BRANCO-artefato d={best_density:.3f}", (5, H-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1, cv2.LINE_AA)
            save_debug(f"{debug_name}_cell.png", vis)
        return CellResult(label=None, confidence=confidence,
                          density=best_density, fill_detected=False)

    for d in band_results:
        cv2.circle(vis, (d["cx"], d["cy"]), d["r"], (0, 255, 0), 1)
        cv2.putText(vis, d["label"], (d["cx"]+d["r"]+3, d["cy"]+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    chosen_label = band_results[best]["label"]
    cv2.putText(vis, f"pick={chosen_label} conf={confidence:.2f} d={best_density:.2f}",
                (5, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

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

def find_name_col_idx(col_intervals: List[Tuple[int, int]]) -> int:
    """
    The name column is always the widest column in the table.
    It sits to the left of all the bubble columns, which are narrow and uniform.

    Strategy:
    1. Find the widest column overall.
    2. As a sanity check, prefer a column in the left half of the image.
       If the widest column is on the right (e.g. a spurious right-margin col
       that slipped through), fall back to the widest column in the left half.
    """
    if not col_intervals:
        return 0

    widths = [(x2 - x1, i) for i, (x1, x2) in enumerate(col_intervals)]
    img_cx = (col_intervals[0][0] + col_intervals[-1][1]) / 2  # rough image centre

    # widest in left half
    left_cols = [(w, i) for w, i in widths if (col_intervals[i][0] + col_intervals[i][1]) / 2 < img_cx]
    if left_cols:
        best = max(left_cols, key=lambda t: t[0])
        print(f"  [name_col] Widest left-half column → idx={best[1]} "
              f"x={col_intervals[best[1]]} w={best[0]}")
        return best[1]

    # fallback: globally widest
    best = max(widths, key=lambda t: t[0])
    print(f"  [name_col] Widest overall column → idx={best[1]} "
          f"x={col_intervals[best[1]]} w={best[0]}")
    return best[1]


def choose_question_columns(headers, col_intervals):
    # name column is the widest; question columns are everything else that
    # OCR identified as a question number pattern
    name_col_idx = find_name_col_idx(col_intervals)
    question_cols = [
        i for i, h in enumerate(headers)
        if i != name_col_idx and re.fullmatch(r"\d{1,3}(?:-[A-Z])?", h)
    ]
    if not question_cols:
        # fallback: all columns except the name column
        question_cols = [i for i in range(len(col_intervals)) if i != name_col_idx]
    return name_col_idx, question_cols

def build_final_table(img: np.ndarray, qr_data: Optional[str] = None):
    """
    Build the final extraction table.
    
    Args:
        img: Preprocessed image
        qr_data: Optional QR code data. If provided, uses QR-based extraction.
                If None, attempts to detect QR code automatically.
    """
    # ============================================================
    # QR CODE EXTRACTION MODE - DETECT FIRST
    # ============================================================
    
    # Try to detect QR code if not provided
    if qr_data is None:
        qr_data = detect_qr_code(img)
    
    # Parse QR data if available
    qr_questions = []
    qr_ids = []
    qr_students = []
    use_qr_mode = False
    qr_metadata = {}
    expected_question_count = 0
    
    if qr_data:
        qr_questions, qr_ids, qr_students, qr_metadata = parse_qr_data(qr_data)
        # New format: has questions and IDs (no names)
        # Legacy format: has questions and names (no IDs)
        if qr_questions and (qr_ids or qr_students):
            use_qr_mode = True
            expected_question_count = len(qr_questions)
            if qr_ids:
                print(f"[QR Mode] Using QR code data (new format): {len(qr_questions)} questions, {len(qr_ids)} student IDs")
                print(f"[QR Metadata] ID_PROF={qr_metadata.get('id_prof')}, ID_ESCOLA={qr_metadata.get('id_escola')}, "
                      f"ANO_ESCOLAR={qr_metadata.get('ano_escolar')}, BIMESTRE={qr_metadata.get('bimestre')}")
            else:
                print(f"[QR Mode] Using QR code data (legacy format): {len(qr_questions)} questions, {len(qr_students)} students")
        else:
            print("[QR Mode] QR data parsing failed, falling back to OCR mode")
    else:
        print("[QR Mode] No QR code detected, falling back to OCR mode")
    
    # Now extract table structure with expected column count from QR
    xs, ys, col_intervals, row_intervals = get_table_structure(img, expected_question_count)

    if len(col_intervals) < 2:
        raise RuntimeError("Não foi possível detectar colunas suficientes.")
    if len(row_intervals) < 2:
        raise RuntimeError("Não foi possível detectar linhas suficientes.")

    # ── FIX 2: robust header detection ──
    header_row, candidate_student_rows = identify_header_and_student_rows(img, row_intervals)
    if header_row is None:
        raise RuntimeError("Não foi possível identificar a linha de cabeçalho.")
    
    # Extract headers using QR or OCR
    if use_qr_mode:
        # QR mode: Build headers directly from QR data (no OCR needed)
        headers = ["Nome/ID"] + qr_questions
        print(f"[QR Mode] Headers from QR code (no OCR): {headers}")
    else:
        headers = ocr_headers(img, col_intervals, header_row, EXPECTED_QUESTION_HEADERS)
    
    name_col_idx, question_col_indices = choose_question_columns(headers, col_intervals)

    if not EXPECTED_QUESTION_HEADERS:
        all_non_name_cols = [i for i in range(len(col_intervals)) if i != name_col_idx]
        if len(question_col_indices) < len(all_non_name_cols):
            print(f"  [headers] Preserving invalid-OCR columns: using all {len(all_non_name_cols)} non-name columns")
            question_col_indices = all_non_name_cols

    # Only filter columns that are clearly BEFORE the name column
    # Don't filter columns after the name column yet - we need all question columns
    if col_intervals:
        name_x1 = col_intervals[name_col_idx][0]

        valid_cols = []
        for i, (x1, x2) in enumerate(col_intervals):
            if x2 <= name_x1:
                print(f"  [Fix3] Dropping pre-name column idx={i} x=({x1},{x2})")
                continue
            valid_cols.append((x1, x2))

        if len(valid_cols) >= 2 and len(valid_cols) < len(col_intervals):
            # Rebuild indices to match the pruned col_intervals
            old_to_new = {}
            new_i = 0
            for old_i, col in enumerate(col_intervals):
                if col in valid_cols:
                    old_to_new[old_i] = new_i
                    new_i += 1
            col_intervals = valid_cols
            name_col_idx  = old_to_new.get(name_col_idx, 0)
            question_col_indices = [
                old_to_new[i] for i in question_col_indices if i in old_to_new
            ]
            # Re-run header extraction on the pruned column set
            if use_qr_mode:
                # QR mode: Rebuild headers directly from QR data (no OCR)
                headers = ["Nome/ID"] + qr_questions[:len(col_intervals)-1]
                print(f"[QR Mode] Rebuilt headers after pruning (no OCR): {headers}")
            else:
                headers = ocr_headers(img, col_intervals, header_row, EXPECTED_QUESTION_HEADERS)
            # Re-derive question columns from fresh headers
            name_col_idx, question_col_indices = choose_question_columns(headers, col_intervals)
            if not EXPECTED_QUESTION_HEADERS:
                all_non_name_cols = [i for i in range(len(col_intervals)) if i != name_col_idx]
                if len(question_col_indices) < len(all_non_name_cols):
                    print(f"  [headers] Preserving invalid-OCR columns after pruning: using all {len(all_non_name_cols)} non-name columns")
                    question_col_indices = all_non_name_cols
            print(f"  [Fix3] col_intervals pruned to {len(col_intervals)} columns")

    if EXPECTED_NUM_QUESTIONS is not None:
        question_col_indices = question_col_indices[:EXPECTED_NUM_QUESTIONS]

    if len(question_col_indices) < MIN_EXPECTED_QUESTION_COLS:
        max_q = max(MIN_EXPECTED_QUESTION_COLS, EXPECTED_NUM_QUESTIONS or 0)
        question_col_indices = [
            i for i in range(len(col_intervals)) if i != name_col_idx
        ][:max_q]

    name_col = col_intervals[name_col_idx]

    print(f"\n[build] candidate_student_rows ({len(candidate_student_rows)} rows):")
    for i, (y1, y2) in enumerate(candidate_student_rows):
        print(f"  candidate {i}: y=({y1},{y2}) h={y2-y1}")

    # Extract student IDs/names using QR or OCR
    student_rows, student_ids, student_names = [], [], []
    
    if use_qr_mode:
        # QR mode: Use data directly from QR code (no OCR needed)
        if qr_ids:
            # New format: use IDs from QR code
            num_students = len(qr_ids)
            student_rows = candidate_student_rows[:num_students]
            student_ids = qr_ids[:num_students]
            student_names = [""] * num_students  # No names in new format, only IDs
            print(f"[QR Mode] Using {len(student_ids)} student IDs from QR code (no OCR)")
        else:
            # Legacy format: use names from QR code
            num_students = len(qr_students)
            student_rows = candidate_student_rows[:num_students]
            student_ids = []  # No IDs in legacy format
            student_names = qr_students[:num_students]
            print(f"[QR Mode] Using {len(student_names)} student names from QR code (no OCR, legacy)")
    else:
        # OCR mode: filter names as before
        raw_names = ocr_names(img, name_col, candidate_student_rows)
        for interval, name in zip(candidate_student_rows, raw_names):
            y1, y2 = interval
            print(f"  [filter] y=({y1},{y2}) h={y2-y1}  name={name!r}  "
                  f"→ {'KEEP' if name and name != '__TOTAL__' and len(name) >= 3 else 'DROP'}")
            if not name or name == "__TOTAL__" or len(name) < 3:
                continue
            student_rows.append(interval)
            student_ids.append("")  # No IDs in OCR mode
            student_names.append(name)

        if len(student_rows) < MIN_EXPECTED_STUDENT_ROWS:
            fallback = candidate_student_rows[-5:] if len(candidate_student_rows) >= 5 else candidate_student_rows
            student_rows  = fallback
            student_ids = [""] * len(fallback)
            student_names = raw_names[-len(fallback):]

    question_headers = []
    for idx in question_col_indices:
        h = headers[idx] if idx < len(headers) else ""
        question_headers.append(h)

    final_question_headers = reconcile_question_headers(
        question_headers,
        EXPECTED_QUESTION_HEADERS
    )
    
    # Ensure unique column headers by appending suffix for duplicates
    seen = {}
    unique_headers = []
    for h in final_question_headers:
        if h in seen:
            seen[h] += 1
            unique_h = f"{h}_{seen[h]}"
            unique_headers.append(unique_h)
            print(f"  [headers] Duplicate header '{h}' renamed to '{unique_h}'")
        else:
            seen[h] = 0
            unique_headers.append(h)
    final_question_headers = unique_headers

    records, conf_records, density_records = [], [], []

    print(f"\n[build] student_rows intervals:")
    for i, (y1, y2) in enumerate(student_rows):
        id_str = f"ID={student_ids[i]!r}" if (student_ids and i < len(student_ids) and student_ids[i]) else ""
        name_str = f"name={student_names[i]!r}" if i < len(student_names) else ""
        print(f"  row{i+1}: y=({y1},{y2}) h={y2-y1}  {id_str} {name_str}")

    for row_i, ((y1, y2), student_id, student_name) in enumerate(zip(student_rows, student_ids, student_names)):
        # Include ID column if we have IDs (new format), or Nome column if we have names (legacy/OCR)
        if student_id:
            rec   = {"ID": student_id}
            c_rec = {"ID": student_id}
            d_rec = {"ID": student_id}
        else:
            rec   = {"Nome": student_name}
            c_rec = {"Nome": student_name}
            d_rec = {"Nome": student_name}

        # Debug: save the full student row so we can see what region is being scanned
        if y2 <= y1:
            print(f"  [WARN] row{row_i+1} has degenerate interval y=({y1},{y2}), skipping")
            records.append(rec)
            conf_records.append(c_rec)
            density_records.append(d_rec)
            continue

        row_crop = crop(img, 0, y1, img.shape[1], y2, pad=0)
        if row_crop is not None:
            save_debug(f"row{row_i+1}_FULLROW.png", row_crop)
        else:
            print(f"  [WARN] row{row_i+1} FULLROW crop returned None y=({y1},{y2}) img_h={img.shape[0]}")

        for q_pos, col_idx in enumerate(question_col_indices):
            # Guard: col_idx must be a valid index into col_intervals
            if col_idx >= len(col_intervals):
                print(f"  [WARN] col_idx={col_idx} out of range (len={len(col_intervals)}), skipping")
                continue

            x1, x2 = col_intervals[col_idx]
            q_name  = final_question_headers[q_pos]
            debug_name = f"row{row_i+1}_{q_name}"
            
            # Debug: print first row processing details
            if row_i == 0:
                print(f"  [DEBUG row1] Processing q_pos={q_pos} col_idx={col_idx} q_name={q_name} x=({x1},{x2})")
            
            cell    = crop(img, x1, y1, x2, y2, pad=4)

            if cell is None:
                # Cell is geometrically degenerate — log it and write a red debug tile
                print(f"  [WARN] crop returned None for {debug_name} "
                      f"col=({x1},{x2}) row=({y1},{y2})")
                # Write a small red placeholder so the debug folder always has an entry
                placeholder = np.zeros((40, 60, 3), dtype=np.uint8)
                placeholder[:] = (0, 0, 180)
                cv2.putText(placeholder, "NONE", (4, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                save_debug(f"{debug_name}_cell.png", placeholder)
                rec[q_name]   = ""
                c_rec[q_name] = 0.0
                d_rec[q_name] = 0.0
                continue

            result  = detect_filled_option_v4(cell, debug_name=debug_name)
            rec[q_name]   = result.label if result.label is not None else ""
            c_rec[q_name] = round(result.confidence, 3)
            d_rec[q_name] = round(result.density, 3)

        records.append(rec)
        conf_records.append(c_rec)
        density_records.append(d_rec)

    df          = pd.DataFrame(records)
    df_conf     = pd.DataFrame(conf_records)
    df_density  = pd.DataFrame(density_records)

    return df, df_conf, df_density, {
        "headers_raw": headers,
        "question_columns": question_col_indices,
        "question_headers_final": final_question_headers,
        "student_names": student_names,
        "qr_metadata": qr_metadata
    }

# ============================================================
# BATCH / SINGLE PROCESSING (unchanged)
# ============================================================

def process_single_image(image_path: str, output_csv: str, debug_dir: str) -> bool:
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ❌ Erro ao abrir imagem: {image_path}")
            return False

        global DEBUG_DIR
        original_debug_dir = DEBUG_DIR
        DEBUG_DIR = debug_dir
        clear_debug_dir(DEBUG_DIR)

        # Detect QR code BEFORE preprocessing to avoid distortion
        qr_data = detect_qr_code(img)
        
        pre = preprocess_document(img)
        df, df_conf, df_density, meta = build_final_table(pre, qr_data=qr_data)

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        conf_path = output_csv.replace(".csv", "_confianca.csv")
        density_path = output_csv.replace(".csv", "_densidade.csv")
        df_conf.to_csv(conf_path, index=False, encoding="utf-8-sig")
        df_density.to_csv(density_path, index=False, encoding="utf-8-sig")
        
        # Save QR metadata as JSON
        metadata_path = output_csv.replace(".csv", "_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta.get('qr_metadata', {}), f, indent=2, ensure_ascii=False)

        DEBUG_DIR = original_debug_dir

        print(f"  ✓ Processado: {Path(image_path).name}")
        print(f"    - {len(meta['student_names'])} alunos, {len(meta['question_headers_final'])} questões")
        return True

    except Exception as e:
        print(f"  ❌ Erro ao processar {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_batch(input_folder: str, output_dir: str = "resultados/batch"):
    input_path = Path(input_folder)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    pdf_extension = {'.pdf'}
    
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    pdf_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in pdf_extension
    ]

    if not image_files and not pdf_files:
        print(f"Nenhuma imagem ou PDF encontrado em: {input_folder}")
        return
    
    total_files = len(image_files) + len(pdf_files)

    print(f"\n{'='*60}")
    print(f"PROCESSAMENTO EM LOTE")
    print(f"{'='*60}")
    print(f"Total de arquivos: {total_files} ({len(image_files)} imagens, {len(pdf_files)} PDFs)")

    results = []
    successful = 0
    failed = 0
    file_counter = 0

    # Process image files
    for image_file in image_files:
        file_counter += 1
        print(f"[{file_counter}/{total_files}] Processando imagem: {image_file.name}")
        image_stem = image_file.stem
        image_output_dir = output_path / image_stem
        image_output_dir.mkdir(exist_ok=True)
        output_csv = str(image_output_dir / "resultado.csv")
        debug_dir = str(image_output_dir / "debug")
        success = process_single_image(str(image_file), output_csv, debug_dir)
        results.append({'file': image_file.name, 'success': success, 'output_dir': str(image_output_dir)})
        if success:
            successful += 1
        else:
            failed += 1
        print()
    
    # Process PDF files
    try:
        from pdf_utils import pdf_to_images
        pdf_support = True
    except ImportError:
        pdf_support = False
        if pdf_files:
            print("AVISO: pdf2image não disponível. PDFs serão ignorados.")
    
    if pdf_support:
        for pdf_file in pdf_files:
            file_counter += 1
            print(f"[{file_counter}/{total_files}] Processando PDF: {pdf_file.name}")
            try:
                # Convert PDF to images
                images = pdf_to_images(str(pdf_file))
                print(f"  PDF contém {len(images)} página(s)")
                
                # Process each page
                for page_num, img in enumerate(images, 1):
                    pdf_stem = f"{pdf_file.stem}_page{page_num}"
                    pdf_output_dir = output_path / pdf_stem
                    pdf_output_dir.mkdir(exist_ok=True)
                    
                    # Save temporary image
                    temp_img_path = pdf_output_dir / "temp_page.png"
                    cv2.imwrite(str(temp_img_path), img)
                    
                    output_csv = str(pdf_output_dir / "resultado.csv")
                    debug_dir = str(pdf_output_dir / "debug")
                    
                    print(f"    Processando página {page_num}/{len(images)}...")
                    success = process_single_image(str(temp_img_path), output_csv, debug_dir)
                    
                    # Clean up temp image
                    temp_img_path.unlink()
                    
                    results.append({
                        'file': f"{pdf_file.name} (página {page_num})",
                        'success': success,
                        'output_dir': str(pdf_output_dir)
                    })
                    if success:
                        successful += 1
                    else:
                        failed += 1
                
            except Exception as e:
                print(f"  ERRO ao processar PDF: {e}")
                results.append({
                    'file': pdf_file.name,
                    'success': False,
                    'output_dir': 'N/A'
                })
                failed += 1
            print()

    print(f"\n{'='*60}")
    print(f"RESUMO: Sucesso: {successful} | Falhas: {failed}")
    print(f"{'='*60}")
    
    # Generate master table combining all results
    if successful > 0:
        print(f"\n{'='*60}")
        print("Gerando tabela mestre com todos os resultados...")
        try:
            from create_master_table import create_master_table
            master_file = create_master_table(str(output_path))
            if master_file:
                print(f"✅ Tabela mestre criada: {master_file}")
            else:
                print("⚠️  Nenhum resultado válido encontrado para tabela mestre")
        except Exception as e:
            print(f"❌ Erro ao criar tabela mestre: {e}")
        print(f"{'='*60}")

    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"RESUMO DO PROCESSAMENTO EM LOTE\n{'='*60}\n")
        f.write(f"Total de arquivos: {total_files}\n")
        f.write(f"  Imagens: {len(image_files)}\n")
        f.write(f"  PDFs: {len(pdf_files)}\n")
        f.write(f"Sucesso: {successful} | Falhas: {failed}\n\n")
        for result in results:
            status = "✓ SUCESSO" if result['success'] else "✗ FALHA"
            f.write(f"{status}: {result['file']}\n  Saída: {result['output_dir']}\n\n")
    print(f"Resumo salvo em: {summary_file}")


def main(qr_data: Optional[str] = None):
    """
    Main extraction function.
    
    Args:
        qr_data: Optional QR code data string. If None, auto-detects QR code from image.
                If auto-detection fails, falls back to global QR_DATA placeholder.
    """
    clear_debug_dir(DEBUG_DIR)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {IMAGE_PATH}")

    # Detect QR code BEFORE preprocessing to avoid distortion
    if qr_data is None:
        qr_data = detect_qr_code(img)
    
    pre = preprocess_document(img)
    
    # Pass the detected QR data to build_final_table
    df, df_conf, df_density, meta = build_final_table(pre, qr_data=qr_data)


    print("\nCabeçalhos finais de questões:")
    print(meta["question_headers_final"])
    print("\nTabela extraída:")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    conf_path    = OUTPUT_CSV.replace(".csv", "_confianca.csv")
    density_path = OUTPUT_CSV.replace(".csv", "_densidade.csv")
    df_conf.to_csv(conf_path, index=False, encoding="utf-8-sig")
    df_density.to_csv(density_path, index=False, encoding="utf-8-sig")

    print(f"\nCSV principal salvo em:   {OUTPUT_CSV}")
    print(f"CSV de confiança salvo em: {conf_path}")
    print(f"CSV de densidade salvo em: {density_path}")
    print(f"Debug salvo em:            {DEBUG_DIR}")

if __name__ == "__main__":
    main()