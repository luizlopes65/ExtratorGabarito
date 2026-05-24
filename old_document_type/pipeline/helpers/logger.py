import logging

def setup_logger(verbose: bool = False) -> logging.Logger:
    """
    Configures and returns the main logger for the OCR pipeline.
    """
    logger = logging.getLogger("ocr_pipeline")
    
    # If the logger already has handlers, it might have been set up already.
    # We just update its level.
    if logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger
        
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger

# Create a default logger instance
logger = logging.getLogger("ocr_pipeline")
