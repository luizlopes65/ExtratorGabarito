#!/usr/bin/env python3
"""
Test script to scan and detect QR codes in answer sheet images.

This script attempts to detect QR codes in various regions of the image
and displays the results for debugging purposes.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from pyzbar import pyzbar

def detect_qr_in_region(img: np.ndarray, region_name: str, x1: int, y1: int, x2: int, y2: int):
    """
    Attempt to detect QR code in a specific region of the image.
    
    Args:
        img: Input image
        region_name: Name of the region for display
        x1, y1, x2, y2: Region coordinates
    """
    print(f"\n{'='*80}")
    print(f"Scanning region: {region_name}")
    print(f"Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"{'='*80}")
    
    # Extract region
    region = img[y1:y2, x1:x2]
    
    if region.size == 0:
        print("❌ Invalid region (empty)")
        return None
    
    # Convert to grayscale if needed
    if len(region.shape) == 3:
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray_region = region
    
    # Try multiple preprocessing techniques
    preprocessing_methods = [
        ("Original", gray_region),
        ("Binary Threshold", cv2.threshold(gray_region, 127, 255, cv2.THRESH_BINARY)[1]),
        ("Adaptive Threshold", cv2.adaptiveThreshold(gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ("Otsu Threshold", cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("Inverted", cv2.bitwise_not(gray_region)),
        ("Contrast Enhanced", cv2.equalizeHist(gray_region)),
    ]
    
    qr_codes = None
    successful_method = None
    
    for method_name, processed in preprocessing_methods:
        qr_codes = pyzbar.decode(processed)
        if qr_codes:
            successful_method = method_name
            print(f"  ✓ Detection successful with: {method_name}")
            break
    
    if not qr_codes:
        qr_codes = []
    
    if qr_codes:
        print(f"✓ Found {len(qr_codes)} QR code(s)!")
        for i, qr in enumerate(qr_codes):
            qr_data = qr.data.decode('utf-8')
            print(f"\n  QR Code #{i+1}:")
            print(f"  Type: {qr.type}")
            print(f"  Data: {qr_data}")
            print(f"  Position: {qr.rect}")
            
            # Draw bounding box on region
            points = qr.polygon
            if len(points) == 4:
                pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
                region_vis = cv2.cvtColor(gray_region.copy(), cv2.COLOR_GRAY2BGR)
                cv2.polylines(region_vis, [pts], True, (0, 255, 0), 3)
                
                # Add text showing detection method
                cv2.putText(region_vis, f"Method: {successful_method}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save visualization
                output_path = f"debug/qr_detected_{region_name.replace(' ', '_').lower()}.png"
                cv2.imwrite(output_path, region_vis)
                print(f"  Saved visualization: {output_path}")
                print(f"  Detection method: {successful_method}")
        
        return qr_codes[0].data.decode('utf-8')
    else:
        print("❌ No QR code detected in this region")
        
        # Save the region for inspection
        output_path = f"debug/qr_notfound_{region_name.replace(' ', '_').lower()}.png"
        cv2.imwrite(output_path, gray_region)
        print(f"  Saved region for inspection: {output_path}")
        
        return None


def test_qr_detection(image_path: str):
    """
    Test QR code detection on an answer sheet image.
    
    Args:
        image_path: Path to the input image
    """
    print("\n" + "="*80)
    print("QR CODE DETECTION TEST")
    print("="*80)
    print(f"Image: {image_path}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return 1
    
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h} pixels")
    
    # Create debug directory
    Path("debug").mkdir(exist_ok=True)
    
    # Test different regions
    regions = [
        ("Top-Left 30%", 0, 0, int(w*0.3), int(h*0.3)),
        ("Top-Left 40%", 0, 0, int(w*0.4), int(h*0.4)),
        ("Top-Left 50%", 0, 0, int(w*0.5), int(h*0.5)),
        ("Top-Left Quarter", 0, 0, w//4, h//4),
        ("Top-Left Half", 0, 0, w//2, h//2),
        ("Full Top Strip", 0, 0, w, int(h*0.2)),
        ("Full Image", 0, 0, w, h),
    ]
    
    detected_qr_data = []
    
    for region_name, x1, y1, x2, y2 in regions:
        qr_data = detect_qr_in_region(img, region_name, x1, y1, x2, y2)
        if qr_data:
            detected_qr_data.append((region_name, qr_data))
    
    # Summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    
    if detected_qr_data:
        print(f"\n✓ Successfully detected QR code(s) in {len(detected_qr_data)} region(s):")
        for region_name, qr_data in detected_qr_data:
            print(f"\n  Region: {region_name}")
            print(f"  Data: {qr_data}")
    else:
        print("\n❌ No QR codes detected in any region")
        print("\nTroubleshooting tips:")
        print("  1. Check if the image actually contains a QR code")
        print("  2. Ensure the QR code is not too small or blurry")
        print("  3. Try adjusting the image contrast/brightness")
        print("  4. Check the saved region images in debug/ folder")
    
    return 0 if detected_qr_data else 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "examples/subtest/image.png"
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found at {image_path}")
        print(f"\nUsage: python {sys.argv[0]} [image_path]")
        print(f"Default: examples/subtest/image.png")
        return 1
    
    return test_qr_detection(image_path)


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
