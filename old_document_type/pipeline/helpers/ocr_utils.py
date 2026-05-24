# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pytesseract
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .geometry import to_gray
from .logger import logger

OCR_LANG = "por+eng"

@dataclass
class OCRBox:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

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

def reconcile_question_headers(question_headers: List[str], expected_headers: Optional[List[str]] = None) -> List[str]:
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

                if score <= 8:
                    final_headers.append(observed_norm)
                else:
                    final_headers.append(expected)
            return final_headers

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
