# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Optional, Tuple, List
# pyrefly: ignore [missing-import]
from pyzbar import pyzbar
from .logger import logger
from .profiler import profile_time
from .geometry import crop

@profile_time("detect_qr_code")
def detect_qr_code(img: np.ndarray) -> Optional[str]:
    """
    Detecta e decodifica código QR do canto superior esquerdo da imagem.
    Usa múltiplas técnicas de pré-processamento para detecção robusta.
    """
    h, w = img.shape[:2]
    regions_to_try = [
        ("Top-Left 30%", img[0:int(h*0.3), 0:int(w*0.3)]),
        ("Top-Left 40%", img[0:int(h*0.4), 0:int(w*0.4)]),
        ("Top-Left 50%", img[0:int(h*0.5), 0:int(w*0.5)]),
    ]
    
    for region_name, qr_region in regions_to_try:
        if len(qr_region.shape) == 3:
            gray_region = cv2.cvtColor(qr_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = qr_region
            
        preprocessing_methods = [
            ("Original", qr_region),
            ("Grayscale", gray_region),
            ("Binary Threshold", cv2.threshold(gray_region, 127, 255, cv2.THRESH_BINARY)[1]),
            ("Adaptive Threshold", cv2.adaptiveThreshold(gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("Otsu Threshold", cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("Inverted", cv2.bitwise_not(gray_region)),
            ("Contrast Enhanced", cv2.equalizeHist(gray_region)),
        ]
    
        for method_name, processed in preprocessing_methods:
            qr_codes = pyzbar.decode(processed)
            if qr_codes:
                qr_data = qr_codes[0].data.decode('utf-8')
                logger.debug(f"[QR] Detected QR code in {region_name} using {method_name}: {qr_data}")
                return qr_data
                
    logger.debug("[QR] No QR code detected in any region (tried 30%, 40%, 50%)")
    return None

def parse_qr_data(qr_data: str) -> Tuple[List[str], List[str], List[str], dict]:
    """
    Analisa dados do código QR para extrair números de questões, IDs de alunos e metadados.
    Formato esperado (novo): "F;ID_PROF;ID_ESCOLA;ANO_ESCOLAR;BIMESTRE;DATA;QUESTÕES_DETALHADAS;IDS_ALUNOS;PÁGINA"
    """
    metadata = {
        'id_prof': None, 'id_escola': None, 'ano_escolar': None,
        'bimestre': None, 'data': None, 'pagina': None
    }
    
    try:
        if ';' in qr_data and ',' in qr_data:
            parts = qr_data.split(';')
            
            if len(parts) >= 8:
                metadata['id_prof'] = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
                metadata['id_escola'] = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
                metadata['ano_escolar'] = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
                metadata['bimestre'] = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
                metadata['data'] = parts[5].strip() if len(parts) > 5 and parts[5].strip() else None
                
                questions_str = parts[6].strip()
                questions = [q.strip() for q in questions_str.split(',') if q.strip()]
                
                ids_str = parts[7].strip()
                student_ids = [sid.strip() for sid in ids_str.split(',') if sid.strip()]
                
                if len(parts) > 8:
                    metadata['pagina'] = parts[8].strip() if parts[8].strip() else None
                
                logger.debug(f"[QR Parse] New format detected")
                logger.debug(f"[QR Parse] Extracted {len(questions)} questions")
                logger.debug(f"[QR Parse] Extracted {len(student_ids)} student IDs")
                
                return questions, student_ids, [], metadata
                
        logger.debug(f"[QR Parse] Unrecognized format: {qr_data}")
        return [], [], [], metadata
        
    except Exception as e:
        logger.error(f"[QR Parse] Error parsing QR data: {e}")
        return [], [], [], metadata

def get_qr_headers(img: np.ndarray, col_intervals: List[Tuple[int, int]], header_row: Tuple[int, int], qr_questions: List[str]) -> List[str]:
    headers = [""] # First is name
    for idx in range(1, len(col_intervals)):
        if idx - 1 < len(qr_questions):
            headers.append(qr_questions[idx - 1])
        else:
            headers.append(f"Q{idx}")
    logger.debug(f"[QR Headers] Generated {len(headers)} headers: {headers}")
    return headers

def get_qr_names(img: np.ndarray, name_col: Tuple[int, int], candidate_rows: List[Tuple[int, int]], qr_students: List[str]) -> List[str]:
    names = []
    for i, _ in enumerate(candidate_rows):
        if i < len(qr_students):
            names.append(qr_students[i])
        else:
            names.append(f"Student_{i+1}")
    logger.debug(f"[QR Names] Generated {len(names)} names: {names}")
    return names
