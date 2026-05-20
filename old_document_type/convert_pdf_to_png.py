
from pathlib import Path
from pdf2image import convert_from_path
import sys

def convert_pdf_to_pngs(pdf_path: str, output_dir: str = None):
    """
    Convert a PDF file to separate PNG images, one per page.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save PNG files (default: same as PDF location)
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    # Set output directory
    if output_dir is None:
        output_dir = pdf_file.parent / f"{pdf_file.stem}_pages"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting PDF: {pdf_file.name}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Convert PDF to images
        # Using 300 DPI for good quality
        images = convert_from_path(pdf_path, dpi=300)
        
        print(f"Found {len(images)} pages")
        
        # Save each page as PNG
        for i, image in enumerate(images, start=1):
            output_path = output_dir / f"page_{i:03d}.png"
            image.save(output_path, 'PNG')
            print(f"  ✓ Saved: {output_path.name}")
        
        print(f"\n✓ Successfully converted {len(images)} pages to PNG")
        print(f"  Output location: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error converting PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Default PDF path
    pdf_path = "1º ANO - A - Jacqueline Goncalves Feliciano - Folhas de Correção - Gabarito.pdf"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    convert_pdf_to_pngs(pdf_path)

 
