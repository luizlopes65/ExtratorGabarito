#!/usr/bin/env python3
"""
Main entry point for OCR Answer Sheet Extractor

This script provides a CLI interface to run either the standard or profiling version
of the answer sheet extraction pipeline.
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract answers from scanned multiple-choice answer sheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run standard version on single image (default)
  python main.py
  
  # Process all images in a folder
  python main.py --batch examples/
  
  # Run with profiling enabled
  python main.py --profile --image examples/my_sheet.png
  
  # Batch process with profiling
  python main.py --profile --batch examples/
  
  # Specify custom output directory for batch
  python main.py --batch examples/ --output-dir resultados/batch_run/
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
        help='Path to input image (overrides default in script)'
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
        print("Running standard version...")
        import extrair_table_fixed as extractor
    
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
            # Single image processing mode
            if args.image:
                extractor.IMAGE_PATH = args.image
            if args.output:
                extractor.OUTPUT_CSV = args.output
            
            # Verify input file exists
            if not Path(extractor.IMAGE_PATH).exists():
                print(f"Error: Image file not found: {extractor.IMAGE_PATH}", file=sys.stderr)
                sys.exit(1)
            
            extractor.main()
            
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
