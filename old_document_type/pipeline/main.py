import sys
import argparse
from pathlib import Path
# pyrefly: ignore [missing-import]
from helpers.pdf_utils import is_pdf_file, pdf_to_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Extract answers from scanned multiple-choice answer sheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run QR-based version on single image (default)
  python main.py
  
  # Process a PDF file (first page)
  python main.py --image exam.pdf
  
  # Process specific page from PDF
  python main.py --image exam.pdf --pdf-page 2
  
  # Process all images in a folder
  python main.py --batch examples/
  
  # Run with profiling enabled
  python main.py --profile --image examples/my_sheet.png
  
  # Batch process with profiling
  python main.py --profile --batch examples/
  
  # Specify custom output directory for batch
  python main.py --batch examples/ --output-dir resultados/batch_run/
  
  # Provide QR data directly (for testing)
  python main.py --qr-data "1;2;3.Student1;Student2;Student3"
  
  # Run the FULL pipeline on a folder
  python main.py --full examples/
        """
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Run profiling version with performance metrics'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image or PDF file (overrides default in script)'
    )
    
    parser.add_argument(
        '--pdf-page',
        type=int,
        default=1,
        help='Page number to extract from PDF (default: 1, first page)'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        metavar='FOLDER',
        help='Process all images in the specified folder'
    )
    
    parser.add_argument(
        '--full',
        type=str,
        metavar='FOLDER',
        help='Run the full pipeline (OCR, consolidate, update cloud) on the given folder'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output CSV file for single image (overrides default in script)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for batch processing results (default: resultados/batch/)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug image generation'
    )
    
    parser.add_argument(
        '--qr-data',
        type=str,
        help='QR code data string (format: "q1;q2;q3.student1;student2;student3")'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if sum(1 for x in [args.batch, args.image, args.full] if x) > 1:
        print("Error: Cannot specify more than one of --batch, --image, or --full", file=sys.stderr)
        sys.exit(1)
    
    # Always import the QR-based version
    # pyrefly: ignore [missing-import]
    from helpers import extrair_table_qr as extractor
    
    if args.profile:
        print("Running profiling version with performance metrics...")
        # pyrefly: ignore [missing-import]
        from helpers.profiler import TimeProfiler
        TimeProfiler.enable()
    else:
        print("Running QR-based version...")
        
    if args.debug:
        extractor.ENABLE_DEBUG_IMAGES = True
    
    # Run batch or single processing
    try:
        if args.batch or args.full:
            # Batch processing mode
            batch_folder = Path(args.batch if args.batch else args.full)
            if not batch_folder.exists():
                print(f"Error: Folder not found: {batch_folder}", file=sys.stderr)
                sys.exit(1)
            
            output_dir = args.output_dir or "resultados/batch"
            extractor.process_batch(str(batch_folder), output_dir)
            
            if args.full:
                print("\n" + "=" * 60)
                print("🔄 RUNNING FULL PIPELINE")
                print("=" * 60)
                
                # 1. Create master table
                # pyrefly: ignore [missing-import]
                from helpers.create_master_table import create_master_table
                master_csv = create_master_table(output_dir)
                
                if master_csv:
                    # 2. Update cloud statistics
                    # pyrefly: ignore [missing-import]
                    from helpers.update_cloud_statistics import run_update
                    MATRIX_CSV = str(PROJECT_ROOT / 'matriz_assuntos_subatributos_populated.csv')
                    CREDENTIALS = str(PROJECT_ROOT / 'credenciais.json')
                    SHEET_ID = '1v21Q3TKPkJuf08HvwpMmZ6IYwa6Wsht2S4OvlKmenfc'
                    
                    run_update(master_csv, MATRIX_CSV, CREDENTIALS, SHEET_ID)
            
        else:
            # Single image/PDF processing mode
            input_path = args.image if args.image else extractor.IMAGE_PATH
            
            # Check if input file exists
            if not Path(input_path).exists():
                print(f"Error: Input file not found: {input_path}", file=sys.stderr)
                sys.exit(1)
            
            # Handle PDF input
            if is_pdf_file(input_path):
                print(f"PDF detected: {input_path}")
                try:
                    images = pdf_to_images(input_path)
                    print(f"  Converted {len(images)} page(s) from PDF")
                    
                    # Use specified page or first page
                    page_num = args.pdf_page
                    if page_num < 1 or page_num > len(images):
                        print(f"Error: Page {page_num} out of range (PDF has {len(images)} pages)", file=sys.stderr)
                        sys.exit(1)
                    
                    # Save the selected page as temporary image
                    temp_image_path = Path(extractor.DEBUG_DIR) / f"pdf_page_{page_num}.png"
                    temp_image_path.parent.mkdir(parents=True, exist_ok=True)
                    # pyrefly: ignore [missing-import]
                    import cv2

                    cv2.imwrite(str(temp_image_path), images[page_num - 1])
                    
                    print(f"  Processing page {page_num} of {len(images)}")
                    extractor.IMAGE_PATH = str(temp_image_path)
                    
                except Exception as e:
                    print(f"Error converting PDF: {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                # Regular image file
                if args.image:
                    extractor.IMAGE_PATH = args.image
            
            if args.output:
                extractor.OUTPUT_CSV = args.output
            
            # Call main with qr_data parameter for QR version
            if args.qr_data:
                extractor.main(qr_data=args.qr_data)
            else:
                extractor.main()
            
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if args.profile:
            # pyrefly: ignore [missing-import]
            from helpers.profiler import TimeProfiler
            TimeProfiler.print_report()


if __name__ == "__main__":
    main()

 
