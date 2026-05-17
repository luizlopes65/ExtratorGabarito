"""
extrator_questionario.py
========================
Extrai respostas de bolinhas do questionário LibrasMat (3 páginas).

Lógica de coordenadas (novos JSONs):
  Páginas 1 e 2 → cada caixa envolve UMA bolinha individual.
                  Basta medir se está preenchida (filled) ou não.
  Página 3      → cada caixa envolve uma linha inteira com 5 bolinhas.
                  Detecta qual das 5 está marcada.
  Última caixa de cada página → QR Code.

Schema de questões:

  Página 1:
    Q1  - Ano da escola     → C1(9ano) C2(1ano) C3(2ano) C4(3ano)
    Q2  - Tipo de escola    → C5(Municipal) C6(Estadual) C7(Federal) C8(Privada)
    Q3  - Sexo              → C9(Feminino) C10(Masculino)
    Q4  - Disciplinas       → C11(Humanas) C12(Biologicas) C13(Exatas)
    Q5  - Likert q1         → C14..C18  (1=Discordo fort. … 5=Concordo fort.)
    Q6  - Likert q2         → C19..C23
    Q7  - Likert q3         → C24..C28
    Q8  - Likert q4         → C29..C33
    QR  - QR Code           → C34

  Página 2:
    Q9  - Likert q5         → C1..C5
    Q10 - Likert q6         → C6..C10
    Q11 - Likert q7         → C11..C15
    Q12 - Likert q8         → C16..C20
    Q13 - Likert q9         → C21..C25
    Q14 - Likert q10        → C26..C30
    QR  - QR Code           → C31

  Página 3:
    P1..P7 - Preferência A/B → C1..C7  (5 bolinhas por linha)
    QR     - QR Code         → C8

Uso:
    python extrator_questionario.py pagina_1.jpg pagina_2.jpg pagina_3.jpg

Saída: resultado_questionario.json  +  pasta debug_questionario/
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("[WARN] pyzbar não encontrado — leitura de QR Code desativada.")

# ─────────────────────────────────────────────────────────────
# COORDENADAS EMBUTIDAS
# ─────────────────────────────────────────────────────────────

COORDS: Dict[int, Dict[str, Dict]] = {
    1: {
        "C1":  {"x1": 232,  "y1": 373,  "x2": 289,  "y2": 423},
        "C2":  {"x1": 443,  "y1": 374,  "x2": 492,  "y2": 419},
        "C3":  {"x1": 651,  "y1": 378,  "x2": 696,  "y2": 414},
        "C4":  {"x1": 853,  "y1": 377,  "x2": 903,  "y2": 417},
        "C5":  {"x1": 238,  "y1": 624,  "x2": 283,  "y2": 671},
        "C6":  {"x1": 505,  "y1": 625,  "x2": 553,  "y2": 672},
        "C7":  {"x1": 759,  "y1": 623,  "x2": 811,  "y2": 676},
        "C8":  {"x1": 996,  "y1": 621,  "x2": 1052, "y2": 677},
        "C9":  {"x1": 239,  "y1": 787,  "x2": 284,  "y2": 834},
        "C10": {"x1": 501,  "y1": 789,  "x2": 548,  "y2": 831},
        "C11": {"x1": 240,  "y1": 954,  "x2": 282,  "y2": 993},
        "C12": {"x1": 504,  "y1": 952,  "x2": 553,  "y2": 993},
        "C13": {"x1": 787,  "y1": 948,  "x2": 834,  "y2": 996},
        "C14": {"x1": 338,  "y1": 1135, "x2": 404,  "y2": 1180},
        "C15": {"x1": 592,  "y1": 1139, "x2": 656,  "y2": 1176},
        "C16": {"x1": 841,  "y1": 1136, "x2": 916,  "y2": 1181},
        "C17": {"x1": 1105, "y1": 1139, "x2": 1170, "y2": 1181},
        "C18": {"x1": 1345, "y1": 1135, "x2": 1405, "y2": 1178},
        "C19": {"x1": 339,  "y1": 1395, "x2": 404,  "y2": 1438},
        "C20": {"x1": 581,  "y1": 1392, "x2": 662,  "y2": 1444},
        "C21": {"x1": 841,  "y1": 1392, "x2": 909,  "y2": 1438},
        "C22": {"x1": 1094, "y1": 1394, "x2": 1158, "y2": 1439},
        "C23": {"x1": 1347, "y1": 1396, "x2": 1414, "y2": 1441},
        "C24": {"x1": 332,  "y1": 1661, "x2": 412,  "y2": 1709},
        "C25": {"x1": 588,  "y1": 1664, "x2": 662,  "y2": 1705},
        "C26": {"x1": 844,  "y1": 1663, "x2": 909,  "y2": 1707},
        "C27": {"x1": 1097, "y1": 1661, "x2": 1154, "y2": 1708},
        "C28": {"x1": 1341, "y1": 1665, "x2": 1416, "y2": 1704},
        "C29": {"x1": 346,  "y1": 1927, "x2": 395,  "y2": 1972},
        "C30": {"x1": 597,  "y1": 1931, "x2": 650,  "y2": 1969},
        "C31": {"x1": 848,  "y1": 1929, "x2": 902,  "y2": 1971},
        "C32": {"x1": 1103, "y1": 1929, "x2": 1161, "y2": 1971},
        "C33": {"x1": 1355, "y1": 1929, "x2": 1404, "y2": 1971},
        "C34": {"x1": 1340, "y1": 2199, "x2": 1454, "y2": 2318},
    },
    2: {
        "C1":  {"x1": 341,  "y1": 293,  "x2": 398,  "y2": 340},
        "C2":  {"x1": 601,  "y1": 293,  "x2": 650,  "y2": 338},
        "C3":  {"x1": 842,  "y1": 293,  "x2": 909,  "y2": 337},
        "C4":  {"x1": 1091, "y1": 293,  "x2": 1167, "y2": 339},
        "C5":  {"x1": 1345, "y1": 293,  "x2": 1413, "y2": 341},
        "C6":  {"x1": 334,  "y1": 549,  "x2": 408,  "y2": 597},
        "C7":  {"x1": 588,  "y1": 551,  "x2": 658,  "y2": 598},
        "C8":  {"x1": 838,  "y1": 551,  "x2": 914,  "y2": 599},
        "C9":  {"x1": 1093, "y1": 554,  "x2": 1164, "y2": 598},
        "C10": {"x1": 1322, "y1": 552,  "x2": 1406, "y2": 595},
        "C11": {"x1": 340,  "y1": 812,  "x2": 407,  "y2": 860},
        "C12": {"x1": 592,  "y1": 815,  "x2": 655,  "y2": 857},
        "C13": {"x1": 843,  "y1": 815,  "x2": 911,  "y2": 855},
        "C14": {"x1": 1096, "y1": 815,  "x2": 1158, "y2": 856},
        "C15": {"x1": 1346, "y1": 810,  "x2": 1409, "y2": 859},
        "C16": {"x1": 338,  "y1": 1075, "x2": 405,  "y2": 1124},
        "C17": {"x1": 596,  "y1": 1080, "x2": 654,  "y2": 1121},
        "C18": {"x1": 839,  "y1": 1080, "x2": 913,  "y2": 1121},
        "C19": {"x1": 1097, "y1": 1081, "x2": 1165, "y2": 1125},
        "C20": {"x1": 1345, "y1": 1076, "x2": 1402, "y2": 1123},
        "C21": {"x1": 344,  "y1": 1342, "x2": 400,  "y2": 1390},
        "C22": {"x1": 591,  "y1": 1338, "x2": 649,  "y2": 1385},
        "C23": {"x1": 847,  "y1": 1343, "x2": 900,  "y2": 1387},
        "C24": {"x1": 1099, "y1": 1349, "x2": 1160, "y2": 1388},
        "C25": {"x1": 1336, "y1": 1346, "x2": 1410, "y2": 1391},
        "C26": {"x1": 348,  "y1": 1613, "x2": 400,  "y2": 1653},
        "C27": {"x1": 590,  "y1": 1606, "x2": 661,  "y2": 1655},
        "C28": {"x1": 843,  "y1": 1613, "x2": 919,  "y2": 1656},
        "C29": {"x1": 1086, "y1": 1603, "x2": 1176, "y2": 1655},
        "C30": {"x1": 1326, "y1": 1600, "x2": 1408, "y2": 1651},
        "C31": {"x1": 1338, "y1": 2198, "x2": 1453, "y2": 2316},
    },
    3: {
        "C1": {"x1": 327,  "y1": 780,  "x2": 1385, "y2": 848},
        "C2": {"x1": 327,  "y1": 846,  "x2": 1386, "y2": 918},
        "C3": {"x1": 326,  "y1": 917,  "x2": 1384, "y2": 994},
        "C4": {"x1": 325,  "y1": 992,  "x2": 1386, "y2": 1060},
        "C5": {"x1": 321,  "y1": 1058, "x2": 1388, "y2": 1137},
        "C6": {"x1": 324,  "y1": 1138, "x2": 1390, "y2": 1210},
        "C7": {"x1": 322,  "y1": 1209, "x2": 1390, "y2": 1279},
        "C8": {"x1": 1338, "y1": 2198, "x2": 1454, "y2": 2320},
    },
}

# ─────────────────────────────────────────────────────────────
# SCHEMA: agrupa caixas em questões
# ─────────────────────────────────────────────────────────────

LIKERT_LABELS = [1, 2, 3, 4, 5]
PREFER_LABELS = ["A2", "A1", "N", "B1", "B2"]

# Cada entrada: field → lista de caixas na ordem das opções
# Para pág 1/2 cada caixa é uma bolinha; para pág 3 a caixa é uma linha inteira.

PAGE_SCHEMA: Dict[int, List[Dict]] = {
    1: [
        {"field": "ano_escola",  "boxes": ["C1","C2","C3","C4"],
         "labels": ["9ano","1ano","2ano","3ano"],   "mode": "single_boxes"},
        {"field": "tipo_escola", "boxes": ["C5","C6","C7","C8"],
         "labels": ["Municipal","Estadual","Federal","Privada"], "mode": "single_boxes"},
        {"field": "sexo",        "boxes": ["C9","C10"],
         "labels": ["Feminino","Masculino"],         "mode": "single_boxes"},
        {"field": "disciplinas", "boxes": ["C11","C12","C13"],
         "labels": ["Humanas","Biologicas","Exatas"], "mode": "single_boxes"},
        {"field": "likert_q1",   "boxes": ["C14","C15","C16","C17","C18"],
         "labels": LIKERT_LABELS,                    "mode": "single_boxes"},
        {"field": "likert_q2",   "boxes": ["C19","C20","C21","C22","C23"],
         "labels": LIKERT_LABELS,                    "mode": "single_boxes"},
        {"field": "likert_q3",   "boxes": ["C24","C25","C26","C27","C28"],
         "labels": LIKERT_LABELS,                    "mode": "single_boxes"},
        {"field": "likert_q4",   "boxes": ["C29","C30","C31","C32","C33"],
         "labels": LIKERT_LABELS,                    "mode": "single_boxes"},
        {"field": "_qr",         "boxes": ["C34"],   "labels": [], "mode": "qr"},
    ],
    2: [
        {"field": "likert_q5",   "boxes": ["C1","C2","C3","C4","C5"],
         "labels": LIKERT_LABELS, "mode": "single_boxes"},
        {"field": "likert_q6",   "boxes": ["C6","C7","C8","C9","C10"],
         "labels": LIKERT_LABELS, "mode": "single_boxes"},
        {"field": "likert_q7",   "boxes": ["C11","C12","C13","C14","C15"],
         "labels": LIKERT_LABELS, "mode": "single_boxes"},
        {"field": "likert_q8",   "boxes": ["C16","C17","C18","C19","C20"],
         "labels": LIKERT_LABELS, "mode": "single_boxes"},
        {"field": "likert_q9",   "boxes": ["C21","C22","C23","C24","C25"],
         "labels": LIKERT_LABELS, "mode": "single_boxes"},
        {"field": "likert_q10",  "boxes": ["C26","C27","C28","C29","C30"],
         "labels": LIKERT_LABELS, "mode": "single_boxes"},
        {"field": "_qr",         "boxes": ["C31"], "labels": [], "mode": "qr"},
    ],
    3: [
        {"field": "pref_q1", "boxes": ["C1"], "labels": PREFER_LABELS, "mode": "row_box"},
        {"field": "pref_q2", "boxes": ["C2"], "labels": PREFER_LABELS, "mode": "row_box"},
        {"field": "pref_q3", "boxes": ["C3"], "labels": PREFER_LABELS, "mode": "row_box"},
        {"field": "pref_q4", "boxes": ["C4"], "labels": PREFER_LABELS, "mode": "row_box"},
        {"field": "pref_q5", "boxes": ["C5"], "labels": PREFER_LABELS, "mode": "row_box"},
        {"field": "pref_q6", "boxes": ["C6"], "labels": PREFER_LABELS, "mode": "row_box"},
        {"field": "pref_q7", "boxes": ["C7"], "labels": PREFER_LABELS, "mode": "row_box"},
        {"field": "_qr",     "boxes": ["C8"], "labels": [],             "mode": "qr"},
    ],
}

# ─────────────────────────────────────────────────────────────
# UTILITÁRIOS DE IMAGEM
# ─────────────────────────────────────────────────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


def crop_box(img: np.ndarray, c: Dict) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, c["x1"]); y1 = max(0, c["y1"])
    x2 = min(w, c["x2"]); y2 = min(h, c["y2"])
    return img[y1:y2, x1:x2].copy()


def binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)


# ─────────────────────────────────────────────────────────────
# QR CODE
# ─────────────────────────────────────────────────────────────

def read_qr(img: np.ndarray) -> Optional[str]:
    if not PYZBAR_AVAILABLE:
        return None
    gray = to_gray(img)
    for proc in [
        gray,
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]:
        codes = pyzbar.decode(proc)
        if codes:
            return codes[0].data.decode("utf-8")
    return None


# ─────────────────────────────────────────────────────────────
# MODO 1: CAIXA INDIVIDUAL (uma bolinha por caixa)
# ─────────────────────────────────────────────────────────────

def is_filled(cell_img: np.ndarray, min_density: float = 0.10) -> Tuple[bool, float]:
    """
    Retorna (preenchida, densidade) para uma caixa que contém uma única bolinha.
    Mede a densidade de pixels escuros no centro da caixa.
    """
    # Valida se a imagem é válida
    if cell_img is None or cell_img.size == 0:
        return False, 0.0
    
    gray = to_gray(cell_img)
    if gray is None or gray.size == 0:
        return False, 0.0
    
    bw   = binarize(gray)
    H, W = bw.shape

    # Círculo interno centralizado, raio = 35% do menor lado
    cx, cy = W // 2, H // 2
    r = max(4, int(min(H, W) * 0.35))

    mask = np.zeros_like(bw)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    total  = cv2.countNonZero(mask)
    filled = cv2.countNonZero(cv2.bitwise_and(bw, mask))
    density = filled / total if total > 0 else 0.0

    return density >= min_density, round(density, 4)


def detect_single_boxes(
    img: np.ndarray,
    boxes: List[str],
    labels: List,
    coords: Dict[str, Dict],
    min_density: float = 0.10,
    debug_dir: Optional[str] = None,
    field: str = "",
    page: int = 0,
) -> Any:
    """
    Para questões onde cada bolinha tem sua própria caixa.
    Mede a densidade de cada caixa e retorna o label da mais preenchida
    (desde que ultrapasse min_density).
    """
    densities = []
    cells     = []

    for box_key in boxes:
        c    = coords[box_key]
        cell = crop_box(img, c)
        filled, density = is_filled(cell, min_density=0.0)  # sempre mede
        densities.append(density)
        cells.append((box_key, cell, density))

    best_idx  = int(np.argmax(densities))
    best_dens = densities[best_idx]

    # Confiança relativa
    sorted_d = sorted(densities, reverse=True)
    if len(sorted_d) >= 2 and sorted_d[0] > 0:
        conf = (sorted_d[0] - sorted_d[1]) / sorted_d[0]
    else:
        conf = 1.0 if best_dens > 0 else 0.0

    selected = labels[best_idx] if best_dens >= min_density else None

    # Debug: salva cada caixa com anotação
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        for i, (box_key, cell, dens) in enumerate(cells):
            vis = cv2.cvtColor(to_gray(cell), cv2.COLOR_GRAY2BGR)
            is_best = (i == best_idx and best_dens >= min_density)
            color = (0, 200, 0) if is_best else (160, 160, 160)
            H, W = vis.shape[:2]
            cx, cy_ = W // 2, H // 2
            r = max(4, int(min(H, W) * 0.35))
            cv2.circle(vis, (cx, cy_), r, color, 2)
            cv2.putText(vis, f"{dens:.3f}", (2, H - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.imwrite(
                str(Path(debug_dir) / f"p{page}_{field}_{box_key}.png"),
                vis,
            )

    return selected, round(conf, 3), densities


# ─────────────────────────────────────────────────────────────
# MODO 2: CAIXA DE LINHA (múltiplas bolinhas numa caixa larga)
# ─────────────────────────────────────────────────────────────

def detect_row_box(
    img: np.ndarray,
    box_key: str,
    labels: List,
    coords: Dict[str, Dict],
    min_density: float = 0.10,
    debug_dir: Optional[str] = None,
    field: str = "",
    page: int = 0,
) -> Any:
    """
    Para questões onde uma caixa larga contém N bolinhas lado a lado.
    Divide em N faixas verticais e mede a densidade central de cada uma.
    """
    c    = coords[box_key]
    cell = crop_box(img, c)
    gray = to_gray(cell)
    bw   = binarize(gray)
    H, W = bw.shape
    n    = len(labels)

    # Raio para medir: proporcional à altura da caixa
    r  = max(4, int(min(H, W // (n * 2)) * 0.4))
    cy = H // 2
    sw = W // n

    densities  = []
    centers_x  = []

    for i in range(n):
        xs = i * sw
        xe = xs + sw if i < n - 1 else W
        strip = bw[:, xs:xe]

        # Centro X: pico da projeção vertical dentro da faixa
        proj = strip.sum(axis=0).astype(np.float32)
        cx_local = int(np.argmax(proj)) if proj.max() > 0 else sw // 2
        cx = xs + cx_local
        centers_x.append(cx)

        mask = np.zeros_like(bw)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        total  = cv2.countNonZero(mask)
        filled = cv2.countNonZero(cv2.bitwise_and(bw, mask))
        densities.append(filled / total if total > 0 else 0.0)

    best_idx  = int(np.argmax(densities))
    best_dens = densities[best_idx]

    sorted_d = sorted(densities, reverse=True)
    if len(sorted_d) >= 2 and sorted_d[0] > 0:
        conf = (sorted_d[0] - sorted_d[1]) / sorted_d[0]
    else:
        conf = 1.0 if best_dens > 0 else 0.0

    selected = labels[best_idx] if best_dens >= min_density else None

    # Debug
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, (cx, dens, lbl) in enumerate(zip(centers_x, densities, labels)):
            is_best = (i == best_idx and best_dens >= min_density)
            color = (0, 200, 0) if is_best else (160, 160, 160)
            cv2.circle(vis, (cx, cy), r, color, 2)
            cv2.putText(vis, f"{dens:.2f}", (max(0, cx - 14), H - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
        label_str = str(selected) if selected is not None else "?"
        cv2.putText(vis, f"-> {label_str} (c={conf:.2f})",
                    (4, H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 220), 1)
        cv2.imwrite(
            str(Path(debug_dir) / f"p{page}_{field}_{box_key}_det.png"),
            vis,
        )

    return selected, round(conf, 3), [round(d, 4) for d in densities]


# ─────────────────────────────────────────────────────────────
# PROCESSAMENTO POR PÁGINA
# ─────────────────────────────────────────────────────────────

def process_page(
    img: np.ndarray,
    page_num: int,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:

    schema = PAGE_SCHEMA[page_num]
    coords = COORDS[page_num]
    results: Dict[str, Any] = {}

    for entry in schema:
        field  = entry["field"]
        boxes  = entry["boxes"]
        labels = entry["labels"]
        mode   = entry["mode"]

        # ── QR Code ──
        if mode == "qr":
            cell   = crop_box(img, coords[boxes[0]])
            qr_val = read_qr(cell)
            results["qr_code"] = qr_val
            print(f"  {'qr_code':20s} → {qr_val!r}")
            if debug_dir:
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(Path(debug_dir) / f"p{page_num}_qr.png"), cell)
            continue

        # ── Uma bolinha por caixa ──
        if mode == "single_boxes":
            selected, conf, densities = detect_single_boxes(
                img, boxes, labels, coords,
                debug_dir=debug_dir, field=field, page=page_num,
            )

        # ── Linha com múltiplas bolinhas ──
        elif mode == "row_box":
            selected, conf, densities = detect_row_box(
                img, boxes[0], labels, coords,
                debug_dir=debug_dir, field=field, page=page_num,
            )

        else:
            selected, conf, densities = None, 0.0, []

        results[field] = selected
        print(
            f"  {field:20s} → {str(selected):20s}"
            f"  conf={conf:.2f}  dens={[round(d,3) for d in densities]}"
        )

    return results


# ─────────────────────────────────────────────────────────────
# INTERFACE PÚBLICA
# ─────────────────────────────────────────────────────────────

def process_questionnaire(
    img_p1: np.ndarray,
    img_p2: np.ndarray,
    img_p3: np.ndarray,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Processa as três páginas e retorna um dicionário unificado.

    Chaves de saída:
        qr_code        → string do QR Code (letra do modelo, ex: "A")
        ano_escola     → "9ano" | "1ano" | "2ano" | "3ano"
        tipo_escola    → "Municipal" | "Estadual" | "Federal" | "Privada"
        sexo           → "Feminino" | "Masculino"
        disciplinas    → "Humanas" | "Biologicas" | "Exatas"
        likert_q1..10  → 1..5
        pref_q1..7     → "A2" | "A1" | "N" | "B1" | "B2"
        (None = bolinha não detectada / não preenchida)
    """
    print("\n=== Página 1 ===")
    r1 = process_page(img_p1, 1, debug_dir)
    print("\n=== Página 2 ===")
    r2 = process_page(img_p2, 2, debug_dir)
    print("\n=== Página 3 ===")
    r3 = process_page(img_p3, 3, debug_dir)

    # QR Code — deve ser idêntico nas 3 páginas; usa o primeiro não-None
    qr_codes = [r1.pop("qr_code", None), r2.pop("qr_code", None), r3.pop("qr_code", None)]
    qr_final = next((q for q in qr_codes if q), None)

    resultado: Dict[str, Any] = {"qr_code": qr_final}
    resultado.update(r1)
    resultado.update(r2)
    resultado.update(r3)
    return resultado


# ─────────────────────────────────────────────────────────────
# MAPEAMENTO PARA LABELS LEGÍVEIS NO CSV
# ─────────────────────────────────────────────────────────────

# Nomes das colunas no CSV (chave interna → nome legível)
COLUMN_NAMES: Dict[str, str] = {
    "arquivo":      "Arquivo",
    "qr_code":      "Modelo (QR)",
    "ano_escola":   "Ano da escola",
    "tipo_escola":  "Tipo de escola",
    "sexo":         "Sexo biológico",
    "disciplinas":  "Disciplinas preferidas",
    "likert_q1":    "Gosto de resolver novos problemas de matemática",
    "likert_q2":    "Gosto de mais matemática em meus estudos",
    "likert_q3":    "Eu realmente gosto de matemática",
    "likert_q4":    "Eu me sinto desconfortável ao pensar em matemática",
    "likert_q5":    "Aprender matemática me deixa nervoso",
    "likert_q6":    "Eu me sinto terrivelmente tenso utilizando matemática",
    "likert_q7":    "Eu tenho medo de matemática",
    "likert_q8":    "Resolvo problemas de matemática sem ter dificuldade",
    "likert_q9":    "Tenho um bom desempenho em disciplinas e conteúdos matemáticos",
    "likert_q10":   "Sinto-me seguro ao resolver problemas matemáticos",
    "pref_q1":      "Preferência questão 1",
    "pref_q2":      "Preferência questão 2",
    "pref_q3":      "Preferência questão 3",
    "pref_q4":      "Preferência questão 4",
    "pref_q5":      "Preferência questão 5",
    "pref_q6":      "Preferência questão 6",
    "pref_q7":      "Preferência questão 7",
}

# Mapeamento de valores internos → texto legível
VALUE_MAP: Dict[str, str] = {
    # Ano da escola
    "9ano": "9º ano",
    "1ano": "1º ano",
    "2ano": "2º ano",
    "3ano": "3º ano",
    # Tipo de escola
    "Municipal": "Municipal",
    "Estadual":  "Estadual",
    "Federal":   "Federal",
    "Privada":   "Privada",
    # Sexo
    "Feminino":  "Feminino",
    "Masculino": "Masculino",
    # Disciplinas
    "Humanas":    "Humanas",
    "Biologicas": "Biológicas",
    "Exatas":     "Exatas",
    # Likert 1–5
    1: "1 - Discordo fortemente",
    2: "2 - Discordo",
    3: "3 - Indiferente",
    4: "4 - Concordo",
    5: "5 - Concordo fortemente",
    # Preferência A/B
    "A2": "Prefiro A plenamente",
    "A1": "Prefiro A parcialmente",
    "N":  "Neutro",
    "B1": "Prefiro B parcialmente",
    "B2": "Prefiro B plenamente",
}


def humanize(result: Dict[str, Any], pdf_name: str) -> Dict[str, str]:
    """Converte um resultado bruto num dicionário com colunas e valores legíveis."""
    row: Dict[str, str] = {"Arquivo": pdf_name}
    for key, col_name in COLUMN_NAMES.items():
        if key == "arquivo":
            continue
        raw = result.get(key)
        if raw is None:
            row[col_name] = ""
        else:
            row[col_name] = str(VALUE_MAP.get(raw, raw))
    return row


# ─────────────────────────────────────────────────────────────
# BATCH: processa todos os PDFs de uma pasta
# ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: Path) -> List[np.ndarray]:
    """
    Converte um PDF de 3 páginas em lista de imagens OpenCV.
    Requer: pip install pdf2image  (e poppler instalado no sistema)
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image não encontrado. Instale com: pip install pdf2image\n"
            "Também é necessário o poppler: https://poppler.freedesktop.org/"
        )

    pil_images = convert_from_path(str(pdf_path))
    imgs = []
    for pil_img in pil_images:
        arr = np.array(pil_img.convert("RGB"))
        imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return imgs


def process_pdf(pdf_path: Path, debug_root: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Processa um único PDF de 3 páginas e retorna o resultado bruto.
    Retorna None em caso de erro.
    """
    print(f"\n{'─'*55}")
    print(f"Processando: {pdf_path.name}")
    print(f"{'─'*55}")

    try:
        imgs = pdf_to_images(pdf_path)
    except Exception as e:
        print(f"  [ERRO] Falha ao converter PDF: {e}")
        return None

    if len(imgs) < 3:
        print(f"  [ERRO] PDF tem apenas {len(imgs)} página(s); esperadas 3.")
        return None

    debug_dir = None
    if debug_root:
        debug_dir = str(Path(debug_root) / pdf_path.stem)

    try:
        resultado = process_questionnaire(imgs[0], imgs[1], imgs[2], debug_dir=debug_dir)
    except Exception as e:
        print(f"  [ERRO] Falha ao processar questionário: {e}")
        import traceback; traceback.print_exc()
        return None

    return resultado


def process_folder(
    input_folder: str,
    output_csv: str = "resultados.csv",
    debug_root: Optional[str] = "debug_batch",
    save_debug: bool = True,
) -> None:
    """
    Processa todos os PDFs de input_folder e salva os resultados em output_csv.

    Parâmetros:
        input_folder  – pasta com os PDFs
        output_csv    – caminho do CSV de saída
        debug_root    – pasta raiz para imagens de debug (None = sem debug)
        save_debug    – se False, desativa o debug mesmo que debug_root esteja definido
    """
    import csv

    folder = Path(input_folder)
    if not folder.exists():
        print(f"Pasta não encontrada: {input_folder}")
        return

    pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    if not pdfs:
        print(f"Nenhum PDF encontrado em: {input_folder}")
        return

    print(f"Encontrados {len(pdfs)} PDF(s) em '{input_folder}'")

    debug_dir = debug_root if save_debug else None

    rows      = []
    erros     = []
    col_order = ["Arquivo"] + [COLUMN_NAMES[k] for k in COLUMN_NAMES if k != "arquivo"]

    for pdf_path in pdfs:
        resultado = process_pdf(pdf_path, debug_root=debug_dir)

        if resultado is None:
            erros.append(pdf_path.name)
            # Linha em branco para não perder o registro no CSV
            row = {"Arquivo": pdf_path.name}
            for col in col_order[1:]:
                row[col] = "ERRO"
        else:
            row = humanize(resultado, pdf_path.name)

        rows.append(row)
        print(f"  ✓ {pdf_path.name}")

    # Salva CSV
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=col_order, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*55}")
    print(f"  Processados : {len(pdfs)} PDF(s)")
    print(f"  Com erro    : {len(erros)}")
    if erros:
        for e in erros:
            print(f"    ✗ {e}")
    print(f"  CSV salvo em: {output_csv}")
    if debug_dir:
        print(f"  Debugs em   : {debug_dir}/")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    """
    Modos de uso:

    1) Pasta de PDFs (modo principal):
       python extrator_questionario.py <pasta_pdfs> [saida.csv]

    2) Três imagens avulsas (modo de teste):
       python extrator_questionario.py pagina_1.jpg pagina_2.jpg pagina_3.jpg
    """
    if len(sys.argv) < 2:
        print(main.__doc__)
        sys.exit(1)

    first_arg = Path(sys.argv[1])

    # ── Modo pasta ──
    if first_arg.is_dir():
        output_csv = sys.argv[2] if len(sys.argv) >= 3 else "resultados.csv"
        process_folder(str(first_arg), output_csv=output_csv)
        return

    # ── Modo três imagens ──
    if len(sys.argv) < 4:
        print(main.__doc__)
        sys.exit(1)

    imgs = []
    for p in sys.argv[1:4]:
        img = cv2.imread(p)
        if img is None:
            print(f"Erro: não foi possível abrir '{p}'")
            sys.exit(1)
        imgs.append(img)

    debug_dir = "debug_questionario"
    resultado = process_questionnaire(*imgs, debug_dir=debug_dir)

    print("\n" + "=" * 55)
    print("RESULTADO FINAL")
    print("=" * 55)
    row = humanize(resultado, Path(sys.argv[1]).stem)
    for col, val in row.items():
        print(f"  {col}: {val}")

    out_json = "resultado_questionario.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    import csv
    col_order = ["Arquivo"] + [COLUMN_NAMES[k] for k in COLUMN_NAMES if k != "arquivo"]
    with open("resultado_questionario.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=col_order, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)

    print(f"\nJSON salvo em : resultado_questionario.json")
    print(f"CSV salvo em  : resultado_questionario.csv")
    print(f"Debugs em     : {debug_dir}/")


if __name__ == "__main__":
    main()