# OCR Project - Answer Sheet Digitizer

This project provides tools to automate the extraction of data from multiple-choice answer sheets (gabaritos) using Computer Vision (OpenCV) and Optical Character Recognition (OCR).

## 🚀 Features

The project implements two different approaches for data extraction:

1.  **Dynamic Grid Detection (`codigo_julia.py`)**:
    *   **Automatic Preprocessing**: Corrects image rotation (skew) and performs perspective transformation to flatten the document.
    *   **Adaptive Grid Detection**: Automatically detects table lines to identify columns (questions) and rows (students).
    *   **Dynamic OCR**: Extracts question headers and student names automatically from the detected grid.
    *   **Advanced Bubble Detection**: Uses circularity and intensity analysis to determine if a bubble is filled, providing a confidence score for each selection.
    *   **CSV Output**: Exports results and confidence scores to CSV files.

2.  **Template-Based Extraction (`extrair_param_personalizado.py`)**:
    *   **Fixed Coordinates**: Uses predefined bounding boxes for high-precision extraction from a specific form layout.
    *   **Fast Processing**: Ideal for standardized forms where the layout never changes.

## 🛠️ Prerequisites

### Tesseract OCR
This project relies on Tesseract for text recognition.
*   **Windows**: Install Tesseract and update the `TESSERACT_CMD` path in `codigo_julia.py`.
*   **Linux/macOS**: `sudo apt install tesseract-ocr` or `brew install tesseract`.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd OCR_Project
    ```

2.  **Install dependencies**:
    Using Poetry (recommended):
    ```bash
    poetry install
    ```
    Or using pip:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

### Automatic Grid Processing
Run `codigo_julia.py` to process a generic image with a table:
```bash
python codigo_julia.py
```
*   Update `IMAGE_PATH` at the top of the file to point to your input image.
*   Check the `debug_gabarito_v2` folder for visual debugging stages.

### Template-Based Processing
Run `extrair_param_personalizado.py` for specific fixed-layout forms:
```bash
python extrair_param_personalizado.py
```

## 📂 Project Structure

*   `codigo_julia.py`: Main pipeline with dynamic grid detection and OMR.
*   `extrair_param_personalizado.py`: Template-based extraction script.
*   `pyproject.toml` / `requirements.txt`: Project dependencies.
*   `coordenadas_*.json`: Coordinate mappings for template-based extraction.
*   `image.png`: Sample answer sheet image.
