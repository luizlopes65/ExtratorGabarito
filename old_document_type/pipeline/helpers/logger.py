import logging

def setup_logger(verbose: bool = False) -> logging.Logger:
    """
    Configura e retorna o logger principal para o pipeline OCR.
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
    
    # Criar formatador
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger

# Criar uma instância de logger padrão
logger = logging.getLogger("ocr_pipeline")
