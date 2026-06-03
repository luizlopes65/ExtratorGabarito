import logging
from pathlib import Path
from datetime import datetime

def setup_logger(verbose: bool = False, log_file: str = None) -> logging.Logger:
    """
    Configura e retorna o logger principal para o pipeline OCR.
    
    Args:
        verbose: Se True, define nível DEBUG, caso contrário INFO
        log_file: Caminho para o arquivo de log. Se None, usa 'pipeline_errors.log' no diretório atual
    """
    logger = logging.getLogger("ocr_pipeline")
    
    # Se o logger já tem handlers, pode ter sido configurado anteriormente.
    # Apenas atualizamos seu nível.
    if logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger
        
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Criar handler de console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Criar formatador para console (sem timestamp)
    console_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(console_formatter)
    
    logger.addHandler(ch)
    
    # Criar handler de arquivo
    if log_file is None:
        log_file = 'pipeline_errors.log'
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.WARNING)  # Apenas warnings e erros no arquivo
    
    # Criar formatador para arquivo (com timestamp)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(file_formatter)
    
    logger.addHandler(fh)
    
    return logger

# Criar uma instância de logger padrão
logger = logging.getLogger("ocr_pipeline")
