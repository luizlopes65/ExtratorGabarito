#!/usr/bin/env python3
"""
Compare different QR code detection methods.
"""

import cv2
import numpy as np
from pyzbar import pyzbar

def test_opencv_detector(image_path: str):
    """Test OpenCV's built-in QR code detector."""
    print("\n" + "="*80)
    print("METHOD 1: OpenCV QRCodeDetector")
    print("="*80)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Initialize the detector
    detector = cv2.QRCodeDetector()
    
    # Detect and decode
    data, points, straight_qrcode = detector.detectAndDecode(img)
    
    if data:
        print(f"✓ QR Code detected!")
        print(f"  Data: {data}")
        if points is not None:
            print(f"  Points: {points}")
        return data
    else:
        print("❌ QR Code not detected")
        return None


def test_pyzbar_detector(image_path: str):
    """Test pyzbar QR code detector with preprocessing."""
    print("\n" + "="*80)
    print("METHOD 2: pyzbar with Adaptive Threshold")
    print("="*80)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply adaptive threshold (this worked in our test)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Detect QR codes
    qr_codes = pyzbar.decode(processed)
    
    if qr_codes:
        print(f"✓ Found {len(qr_codes)} QR code(s)!")
        for i, qr in enumerate(qr_codes):
            qr_data = qr.data.decode('utf-8')
            print(f"\n  QR Code #{i+1}:")
            print(f"  Type: {qr.type}")
            print(f"  Data: {qr_data}")
            print(f"  Position: {qr.rect}")
        return qr_codes[0].data.decode('utf-8')
    else:
        print("❌ No QR code detected")
        return None


def test_opencv_on_region(image_path: str):
    """Test OpenCV detector on top-left region."""
    print("\n" + "="*80)
    print("METHOD 3: OpenCV QRCodeDetector on Top-Left 30%")
    print("="*80)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    region = img[0:int(h*0.3), 0:int(w*0.3)]
    
    # Initialize the detector
    detector = cv2.QRCodeDetector()
    
    # Detect and decode
    data, points, straight_qrcode = detector.detectAndDecode(region)
    
    if data:
        print(f"✓ QR Code detected in region!")
        print(f"  Data: {data}")
        return data
    else:
        print("❌ QR Code not detected in region")
        return None


def main():
    image_path = "examples/subtest/image.png"
    
    print("\n" + "="*80)
    print("QR CODE DETECTION METHOD COMPARISON")
    print("="*80)
    print(f"Image: {image_path}\n")
    
    # Test all methods
    result1 = test_opencv_detector(image_path)
    result2 = test_pyzbar_detector(image_path)
    result3 = test_opencv_on_region(image_path)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOpenCV (full image):     {'✓ ' + result1 if result1 else '❌ Failed'}")
    print(f"pyzbar (adaptive):       {'✓ ' + result2 if result2 else '❌ Failed'}")
    print(f"OpenCV (top-left 30%):   {'✓ ' + result3 if result3 else '❌ Failed'}")
    
    if result1 or result2 or result3:
        print("\n✓ At least one method succeeded!")
    else:
        print("\n❌ All methods failed")


if __name__ == "__main__":
    main()

# Made with Bob
