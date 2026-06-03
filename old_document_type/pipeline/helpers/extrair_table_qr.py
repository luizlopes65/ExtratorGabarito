# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path
from typing import Optional, List

# Local imports
from .logger import logger
from .profiler import profile_time
from .geometry import preprocess_document, crop
from .grid_detector import get_table_structure, identify_header_and_student_rows, choose_question_columns
from .qr_parser import detect_qr_code, parse_qr_data, get_qr_headers
from .bubble_analyzer import detect_filled_option_v4
from .ocr_utils import reconcile_question_headers, clean_name, ocr_text_block

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configuration constants
IMAGE_PATH = str(PROJECT_ROOT / "examples/page_002.png")
OUTPUT_CSV = str(PROJECT_ROOT / "resultados/resultado_gabarito_qr.csv")
DEBUG_DIR = str(PROJECT_ROOT / "debug/debug_gabarito_qr")
ENABLE_DEBUG_IMAGES = False

MIN_EXPECTED_QUESTION_COLS = 5
MIN_EXPECTED_STUDENT_ROWS = 3
EXPECTED_NUM_QUESTIONS = None
EXPECTED_QUESTION_HEADERS = []

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
        import shutil
        shutil.rmtree(debug_path)
    debug_path.mkdir(parents=True, exist_ok=True)

@profile_time("build_final_table")
def build_final_table(img: np.ndarray, qr_data: Optional[str] = None):
    """
    Build the final extraction table. Orchestrates the flow using specialized modules.
    """
    if qr_data is None:
        qr_data = detect_qr_code(img)
    
    qr_questions = []
    qr_ids = []
    use_qr_mode = False
    qr_metadata = {}
    expected_question_count = 0
    
    if qr_data:
        qr_questions, qr_ids, _, qr_metadata = parse_qr_data(qr_data)
        if qr_questions and qr_ids:
            use_qr_mode = True
            expected_question_count = len(qr_questions)
            logger.info(f"[QR Mode] Using QR code data: {len(qr_questions)} questions, {len(qr_ids)} student IDs")
            logger.info(f"[QR Metadata] ID_PROF={qr_metadata.get('id_prof')}, ID_ESCOLA={qr_metadata.get('id_escola')}")
        else:
            logger.info("[QR Mode] QR data parsing failed or legacy format ignored, falling back to OCR mode")
    else:
        logger.info("[QR Mode] No QR code detected, falling back to OCR mode")
    
    xs, ys, col_intervals, row_intervals = get_table_structure(img, expected_question_count)

    if len(col_intervals) < 2:
        raise RuntimeError("Não foi possível detectar colunas suficientes.")
    if len(row_intervals) < 2:
        raise RuntimeError("Não foi possível detectar linhas suficientes.")

    header_row, candidate_student_rows = identify_header_and_student_rows(img, row_intervals, use_qr_mode)
    if header_row is None:
        raise RuntimeError("Não foi possível identificar a linha de cabeçalho.")
    
    if use_qr_mode:
        headers = ["Nome/ID"] + qr_questions
        logger.debug(f"[QR Mode] Headers from QR code: {headers}")
    else:
        logger.error("[OCR Mode] OCR header extraction is not fully supported without QR. Using placeholders.")
        headers = ["Nome/ID"] + [f"Q{i}" for i in range(1, len(col_intervals))]
    
    name_col_idx, question_col_indices = choose_question_columns(headers, col_intervals)

    if not EXPECTED_QUESTION_HEADERS:
        all_non_name_cols = [i for i in range(len(col_intervals)) if i != name_col_idx]
        if len(question_col_indices) < len(all_non_name_cols):
            logger.debug(f"  [headers] Preserving invalid-OCR columns: using all {len(all_non_name_cols)} non-name columns")
            question_col_indices = all_non_name_cols

    if col_intervals:
        name_x1 = col_intervals[name_col_idx][0]
        valid_cols = []
        for i, (x1, x2) in enumerate(col_intervals):
            if x2 <= name_x1:
                logger.debug(f"  [Fix3] Dropping pre-name column idx={i} x=({x1},{x2})")
                continue
            valid_cols.append((x1, x2))

        if len(valid_cols) >= 2 and len(valid_cols) < len(col_intervals):
            old_to_new = {}
            new_i = 0
            for old_i, col in enumerate(col_intervals):
                if col in valid_cols:
                    old_to_new[old_i] = new_i
                    new_i += 1
            col_intervals = valid_cols
            name_col_idx  = old_to_new.get(name_col_idx, 0)
            question_col_indices = [old_to_new[i] for i in question_col_indices if i in old_to_new]
            if use_qr_mode:
                headers = ["Nome/ID"] + qr_questions[:len(col_intervals)-1]
            name_col_idx, question_col_indices = choose_question_columns(headers, col_intervals)
            if not EXPECTED_QUESTION_HEADERS:
                all_non_name_cols = [i for i in range(len(col_intervals)) if i != name_col_idx]
                if len(question_col_indices) < len(all_non_name_cols):
                    question_col_indices = all_non_name_cols

    if EXPECTED_NUM_QUESTIONS is not None:
        question_col_indices = question_col_indices[:EXPECTED_NUM_QUESTIONS]

    if len(question_col_indices) < MIN_EXPECTED_QUESTION_COLS:
        max_q = max(MIN_EXPECTED_QUESTION_COLS, EXPECTED_NUM_QUESTIONS or 0)
        question_col_indices = [i for i in range(len(col_intervals)) if i != name_col_idx][:max_q]

    name_col = col_intervals[name_col_idx]

    logger.debug(f"\n[build] candidate_student_rows ({len(candidate_student_rows)} rows):")

    student_rows, student_ids, student_names = [], [], []
    
    if use_qr_mode:
        num_students = len(qr_ids)
        student_rows = candidate_student_rows[:num_students]
        student_ids = qr_ids[:num_students]
        student_names = [""] * num_students
        logger.debug(f"[QR Mode] Using {len(student_ids)} student IDs from QR code")
    else:
        logger.error("[OCR Mode] OCR Names extraction not supported without QR. Using empty names.")
        student_rows = candidate_student_rows
        student_ids = [""] * len(candidate_student_rows)
        student_names = ["Aluno OCR"] * len(candidate_student_rows)

    question_headers = []
    for idx in question_col_indices:
        h = headers[idx] if idx < len(headers) else ""
        question_headers.append(h)

    final_question_headers = reconcile_question_headers(question_headers, EXPECTED_QUESTION_HEADERS)
    
    seen = {}
    unique_headers = []
    for h in final_question_headers:
        if h in seen:
            seen[h] += 1
            unique_h = f"{h}_{seen[h]}"
            unique_headers.append(unique_h)
            logger.debug(f"  [headers] Duplicate header '{h}' renamed to '{unique_h}'")
        else:
            seen[h] = 0
            unique_headers.append(h)
    final_question_headers = unique_headers

    records, conf_records, density_records = [], [], []

    logger.debug(f"\n[build] student_rows intervals:")
    for i, (y1, y2) in enumerate(student_rows):
        id_str = f"ID={student_ids[i]!r}" if (student_ids and i < len(student_ids) and student_ids[i]) else ""
        logger.debug(f"  row{i+1}: y=({y1},{y2}) h={y2-y1}  {id_str}")

    for row_i, ((y1, y2), student_id, student_name) in enumerate(zip(student_rows, student_ids, student_names)):
        if student_id:
            rec, c_rec, d_rec = {"ID": student_id}, {"ID": student_id}, {"ID": student_id}
        else:
            rec, c_rec, d_rec = {"Nome": student_name}, {"Nome": student_name}, {"Nome": student_name}

        if y2 <= y1:
            logger.debug(f"  [WARN] row{row_i+1} has degenerate interval y=({y1},{y2}), skipping")
            records.append(rec)
            conf_records.append(c_rec)
            density_records.append(d_rec)
            continue

        row_crop = crop(img, 0, y1, img.shape[1], y2, pad=0)
        if row_crop is not None:
            save_debug(f"row{row_i+1}_FULLROW.png", row_crop)

        for q_pos, col_idx in enumerate(question_col_indices):
            if col_idx >= len(col_intervals):
                continue
            x1, x2 = col_intervals[col_idx]
            q_name  = final_question_headers[q_pos]
            debug_name = f"row{row_i+1}_{q_name}"
            
            if row_i == 0:
                logger.debug(f"  [DEBUG row1] Processing q_pos={q_pos} col_idx={col_idx} q_name={q_name} x=({x1},{x2})")
            
            cell = crop(img, x1, y1, x2, y2, pad=4)

            if cell is None:
                logger.debug(f"  [WARN] crop returned None for {debug_name} col=({x1},{x2}) row=({y1},{y2})")
                placeholder = np.zeros((40, 60, 3), dtype=np.uint8)
                placeholder[:] = (0, 0, 180)
                cv2.putText(placeholder, "NONE", (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                save_debug(f"{debug_name}_cell.png", placeholder)
                rec[q_name]   = ""
                c_rec[q_name] = 0.0
                d_rec[q_name] = 0.0
                continue

            result = detect_filled_option_v4(cell, debug_name=debug_name)
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

@profile_time("process_single_image")
def process_single_image(image_path: str, output_csv: str, debug_dir: str) -> bool:
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"  ❌ Erro ao abrir imagem: {image_path}")
            return False

        global DEBUG_DIR
        original_debug_dir = DEBUG_DIR
        DEBUG_DIR = debug_dir
        clear_debug_dir(DEBUG_DIR)

        qr_data = detect_qr_code(img)
        pre = preprocess_document(img)
        df, df_conf, df_density, meta = build_final_table(pre, qr_data=qr_data)

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        conf_path = output_csv.replace(".csv", "_confianca.csv")
        density_path = output_csv.replace(".csv", "_densidade.csv")
        df_conf.to_csv(conf_path, index=False, encoding="utf-8-sig")
        df_density.to_csv(density_path, index=False, encoding="utf-8-sig")
        
        metadata_path = output_csv.replace(".csv", "_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta.get('qr_metadata', {}), f, indent=2, ensure_ascii=False)

        DEBUG_DIR = original_debug_dir

        logger.info(f"  ✓ Processado: {Path(image_path).name}")
        logger.info(f"    - {len(meta['student_names'])} alunos, {len(meta['question_headers_final'])} questões")
        return True

    except RuntimeError as e:
        # Erro específico de detecção de colunas
        error_msg = str(e)
        if "Não foi possível detectar colunas suficientes" in error_msg:
            logger.error(f"  ❌ Erro ao processar {image_path}: {error_msg}")
            logger.warning(f"     Arquivo: {image_path}")
            logger.warning(f"     Motivo: Falha na detecção de estrutura da tabela")
            logger.warning(f"     Sugestão: Verifique a qualidade da imagem e o alinhamento do documento")
        else:
            logger.error(f"  ❌ Erro ao processar {image_path}: {e}")
        
        # Log detalhado no arquivo
        import traceback
        logger.debug(f"Traceback completo:\n{''.join(traceback.format_exc())}")
        return False
    
    except Exception as e:
        logger.error(f"  ❌ Erro ao processar {image_path}: {e}")
        import traceback
        logger.debug(f"Traceback completo:\n{''.join(traceback.format_exc())}")
        traceback.print_exc()
        return False

def _worker_process_single(task_info):
    # pyrefly: ignore [missing-import]
    import cv2
    import traceback
    cv2.setNumThreads(1)
    try:
        success = process_single_image(task_info['image_path'], task_info['output_csv'], task_info['debug_dir'])
        if task_info.get('is_temp'):
            try:
                Path(task_info['image_path']).unlink()
            except Exception:
                pass
        return {
            'file': task_info['file_name'],
            'success': success,
            'output_dir': str(Path(task_info['output_csv']).parent)
        }
    except Exception as e:
        logger.error(f"Erro no worker ao processar {task_info['file_name']}: {e}")
        return {
            'file': task_info['file_name'],
            'success': False,
            'output_dir': 'N/A'
        }

@profile_time("process_batch")
def process_batch(input_folder: str, output_dir: str = "resultados/batch"):
    input_path = Path(input_folder)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    pdf_extension = {'.pdf'}
    
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    pdf_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in pdf_extension]

    if not image_files and not pdf_files:
        logger.warning(f"Nenhuma imagem ou PDF encontrado em: {input_folder}")
        return
    
    total_files = len(image_files) + len(pdf_files)
    logger.info(f"\n{'='*60}\nPROCESSAMENTO EM LOTE\n{'='*60}")
    logger.info(f"Total de arquivos: {total_files} ({len(image_files)} imagens, {len(pdf_files)} PDFs)")

    results = []
    successful = 0
    failed = 0
    tasks = []

    for image_file in image_files:
        image_stem = image_file.stem
        image_output_dir = output_path / image_stem
        image_output_dir.mkdir(exist_ok=True)
        tasks.append({
            'image_path': str(image_file),
            'output_csv': str(image_output_dir / "resultado.csv"),
            'debug_dir': str(image_output_dir / "debug"),
            'file_name': image_file.name,
            'is_temp': False
        })
    
    try:
        # pyrefly: ignore [missing-import]
        from helpers.pdf_utils import pdf_to_images
        pdf_support = True
    except ImportError:
        pdf_support = False
        if pdf_files:
            logger.warning("AVISO: pdf2image não disponível. PDFs serão ignorados.")
            
    if pdf_support:
        for pdf_file in pdf_files:
            try:
                images = pdf_to_images(str(pdf_file))
                logger.info(f"PDF {pdf_file.name}: extraídas {len(images)} páginas.")
                for i, img_array in enumerate(images):
                    page_stem = f"{pdf_file.stem}_page{i+1}"
                    page_output_dir = output_path / page_stem
                    page_output_dir.mkdir(exist_ok=True)
                    temp_img_path = str(output_path / f"temp_{page_stem}.png")
                    cv2.imwrite(temp_img_path, img_array)
                    tasks.append({
                        'image_path': temp_img_path,
                        'output_csv': str(page_output_dir / "resultado.csv"),
                        'debug_dir': str(page_output_dir / "debug"),
                        'file_name': f"{pdf_file.name} (Pág {i+1})",
                        'is_temp': True
                    })
            except Exception as e:
                logger.error(f"Erro ao processar PDF {pdf_file.name}: {e}")

    # Use multiprocessing
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    num_workers = min(multiprocessing.cpu_count(), len(tasks))
    logger.info(f"Iniciando processamento com {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_process_single, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if res['success']:
                successful += 1
            else:
                failed += 1
                
    logger.info(f"\n{'='*60}\nRESUMO DO BATCH\n{'='*60}")
    logger.info(f"Total processado: {len(results)}")
    logger.info(f"Sucesso: {successful}")
    logger.info(f"Falha: {failed}")
    
    # pyrefly: ignore [missing-import]
    from helpers.create_master_table import create_master_table
    logger.info("Gerando tabela mestre com todos os resultados...")
    create_master_table(output_dir)

def main(qr_data: Optional[str] = None):
    success = process_single_image(IMAGE_PATH, OUTPUT_CSV, DEBUG_DIR)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()