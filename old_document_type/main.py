import sys
import argparse
from pathlib import Path
from pdf_utils import is_pdf_file, pdf_to_images


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
        help='Enable debug image generation (profiling version only)'
    )
    
    parser.add_argument(
        '--qr-data',
        type=str,
        help='QR code data string (format: "q1;q2;q3.student1;student2;student3")'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch and args.image:
        print("Error: Cannot specify both --batch and --image", file=sys.stderr)
        sys.exit(1)
    
    # Import the appropriate version
    if args.profile:
        print("Running profiling version with performance metrics...")
        import extrair_table_profiling as extractor
        
        if args.debug:
            extractor.ENABLE_DEBUG_IMAGES = True
            extractor.PROFILE_DETAILED = True
    else:
        print("Running QR-based version...")
        import extrair_table_qr as extractor
    
    # Run batch or single processing
    try:
        if args.batch:
            # Batch processing mode
            batch_folder = Path(args.batch)
            if not batch_folder.exists():
                print(f"Error: Batch folder not found: {batch_folder}", file=sys.stderr)
                sys.exit(1)
            
            output_dir = args.output_dir or "resultados/batch"
            extractor.process_batch(str(batch_folder), output_dir)
            
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
            
            # Set QR data if provided (only for QR version)
            if args.qr_data and not args.profile:
                extractor.QR_DATA = args.qr_data
                print(f"Using provided QR data: {args.qr_data[:50]}...")
            
            # Call main with qr_data parameter for QR version
            if not args.profile and hasattr(extractor, 'QR_DATA'):
                extractor.main(qr_data=extractor.QR_DATA)
            else:
                extractor.main()
            
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

 
