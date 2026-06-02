#!/usr/bin/env python3
"""
Utilitários de PDF para Extrator de Gabaritos OCR

Este módulo fornece utilitários para converter arquivos PDF em imagens para processamento.
Cada página do PDF é convertida em uma imagem separada.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
from pathlib import Path
from typing import List, Optional
from .profiler import profile_time

@profile_time("pdf_to_images")
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    # pyrefly: ignore [missing-import]
    from pdf2image import convert_from_path

    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Converter páginas do PDF para imagens PIL
        pil_images = convert_from_path(str(pdf_file), dpi=dpi)
        
        # Converter imagens PIL para formato OpenCV (BGR)
        cv_images = []
        for pil_img in pil_images:
            # Converter PIL RGB para array numpy
            rgb_array = np.array(pil_img.convert("RGB"))
            # Converter RGB para BGR (formato OpenCV)
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv_images.append(bgr_array)
        
        return cv_images
    
    except Exception as e:
        raise Exception(f"Failed to convert PDF to images: {e}")


def save_pdf_pages_as_images(
    pdf_path: str,
    output_dir: str,
    prefix: str = None,
    dpi: int = 300,
    format: str = "png"
) -> List[str]:
    """
    Convert PDF pages to image files and save them.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the images
        prefix: Prefix for output filenames (default: None, uses PDF filename without extension)
        dpi: Resolution for conversion (default: 300)
        format: Image format (default: "png")
    
    Returns:
        List of paths to saved image files
    
    Example:
        >>> paths = save_pdf_pages_as_images("exam.pdf", "output/")
        >>> print(paths)
        ['output/exam_pagina_1.png', 'output/exam_pagina_2.png', 'output/exam_pagina_3.png']
    """
    images = pdf_to_images(pdf_path, dpi=dpi)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use PDF filename (without extension) as prefix if not provided
    if prefix is None:
        prefix = Path(pdf_path).stem
    
    saved_paths = []
    for i, img in enumerate(images, start=1):
        # New naming convention: pdf_name_pagina_X.png
        filename = f"{prefix}_pagina_{i}.{format}"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), img)
        saved_paths.append(str(filepath))
    
    return saved_paths


def is_pdf_file(filepath: str) -> bool:
    """
    Check if a file is a PDF based on its extension.
    
    Args:
        filepath: Path to the file
    
    Returns:
        True if file has .pdf extension (case-insensitive)
    """
    return Path(filepath).suffix.lower() == '.pdf'


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Get the number of pages in a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Number of pages in the PDF
    """
    try:
        # pyrefly: ignore [missing-import]
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        return info.get("Pages", 0)
    except ImportError:
        # Fallback: convert and count
        images = pdf_to_images(pdf_path)
        return len(images)
    except Exception:
        return 0


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_utils.py <pdf_file> [output_dir]")
        print("\nExample:")
        print("  python pdf_utils.py exam.pdf output/")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "pdf_output"
    
    try:
        print(f"Converting PDF: {pdf_file}")
        print(f"Output directory: {output_dir}")
        
        # Get page count
        page_count = get_pdf_page_count(pdf_file)
        print(f"PDF has {page_count} page(s)")
        
        # Convert and save
        saved_files = save_pdf_pages_as_images(pdf_file, output_dir)
        
        print(f"\n✓ Successfully converted {len(saved_files)} page(s):")
        for i, filepath in enumerate(saved_files, start=1):
            print(f"  {i}. {filepath}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

 
