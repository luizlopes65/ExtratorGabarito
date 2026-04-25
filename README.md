# OCR Project - Answer Sheet Digitizer

This project provides tools to automate the extraction of data from multiple-choice answer sheets (gabaritos) using Computer Vision (OpenCV) and Optical Character Recognition (OCR).

## 🚀 Features

The project implements two versions of the extraction pipeline:

1.  **Standard Version (`extrair_table_fixed.py`)**:
    *   **Automatic Preprocessing**: Corrects image rotation (skew) and performs perspective transformation to flatten the document.
    *   **Adaptive Grid Detection**: Automatically detects table lines to identify columns (questions) and rows (students).
    *   **Smart OCR with Pattern Inference**: Extracts question headers and student names with intelligent error correction for multi-part questions (e.g., 35-A, 35-B, 36-A, 36-B).
    *   **Advanced Bubble Detection**: Uses circularity and intensity analysis to determine if a bubble is filled, providing confidence and density scores.
    *   **CSV Output**: Exports results, confidence scores, and density metrics to CSV files.

2.  **Profiling Version (`extrair_table_profiling.py`)**:
    *   All features from the standard version
    *   **Parallel OCR Processing**: Uses ThreadPoolExecutor for faster header and name extraction
    *   **Performance Metrics**: Detailed profiling reports showing execution time per function and layer
    *   **cProfile Integration**: Generates detailed performance profiles for optimization

## ✨ Recent Improvements

- **Fixed Question Pairing**: Enhanced OCR to correctly handle multi-part questions (35-A, 35-B, 36-A, 36-B, etc.)
- **Pattern-Based Inference**: Automatically corrects OCR errors by detecting question number sequences
- **Improved Header Detection**: Better extraction of question identifiers with letter suffixes
- **Unified Main Entry Point**: Simple CLI interface to run either version

## 🛠️ Prerequisites

### Tesseract OCR
This project relies on Tesseract for text recognition.
*   **macOS**: `brew install tesseract`
*   **Linux**: `sudo apt install tesseract-ocr`
*   **Windows**: Install Tesseract and update the `TESSERACT_CMD` path in the scripts.

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

3.  **Configure environment** (optional):
    ```bash
    cp .env.example .env
    # Edit .env to customize paths and parameters
    ```
    
    The `.env` file allows you to configure:
    - Input/output paths
    - Tesseract location
    - Grid detection parameters
    - Bubble detection thresholds
    - Profiling settings

## 🖥️ Usage

### Quick Start with main.py

The easiest way to run the extractor:

#### Single Image Processing

```bash
# Run standard version (default)
poetry run python main.py

# Run with profiling enabled
poetry run python main.py --profile

# Specify custom image path
poetry run python main.py --image examples/my_sheet.png

# Run profiling with custom output
poetry run python main.py --profile --output resultados/custom_output.csv

# Enable debug images (profiling version only)
poetry run python main.py --profile --debug
```

#### Batch Processing (Multiple Images)

Process all images in a folder at once:

```bash
# Process all images in examples/ folder
poetry run python main.py --batch examples/

# Batch process with profiling
poetry run python main.py --profile --batch examples/

# Specify custom output directory
poetry run python main.py --batch examples/ --output-dir resultados/my_batch/
```

**Batch Output Structure:**
```
resultados/batch/
├── image1/
│   ├── resultado.csv
│   ├── resultado_confianca.csv
│   ├── resultado_densidade.csv
│   └── debug/
├── image2/
│   ├── resultado.csv
│   ├── resultado_confianca.csv
│   ├── resultado_densidade.csv
│   └── debug/
└── summary.txt              # Processing summary with statistics
```

The `summary.txt` file contains:
- Total images processed
- Success/failure counts
- Processing time per image (profiling version)
- Detailed status for each image

### Direct Script Execution

You can also run the scripts directly:

```bash
# Standard version
poetry run python extrair_table_fixed.py

# Profiling version
poetry run python extrair_table_profiling.py
```

*   Update `IMAGE_PATH` at the top of the file to point to your input image.
*   Check the `debug/` folder for visual debugging stages.

## 📂 Project Structure

```
OCR_Project/
├── main.py                      # CLI entry point
├── extrair_table_fixed.py       # Standard extraction pipeline
├── extrair_table_profiling.py   # Profiling version with performance metrics
├── pyproject.toml               # Poetry dependencies
├── requirements.txt             # Pip dependencies
├── examples/                    # Sample answer sheet images
│   └── image.png
├── resultados/                  # Output CSV files
├── debug/                       # Debug images (grid detection, OCR, etc.)
├── archive/                     # Archived/unused code
└── AGENTS.md                    # Development guidelines for AI agents
```

## 📊 Output Files

The extractor generates three CSV files:

1. **Main Results** (`resultado_gabarito_v3.csv`): Student names and their selected answers
2. **Confidence Scores** (`resultado_gabarito_v3_confianca.csv`): Confidence level for each answer detection
3. **Density Metrics** (`resultado_gabarito_v3_densidade.csv`): Fill density for each bubble

## 🐛 Debugging

Debug images are saved to the `debug/` directory showing:
- Preprocessed document (rotation correction, perspective transform)
- Grid line detection (vertical and horizontal lines)
- Header cell OCR results
- Name cell OCR results
- Bubble detection for each answer cell

## 🔧 Configuration

### Using .env File (Recommended)

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Edit `.env` to customize:

**Paths:**
- `IMAGE_PATH`: Input answer sheet image
- `OUTPUT_CSV`: Output CSV file (generates 3 files: main, confidence, density)
- `DEBUG_DIR`: Directory for debug images
- `TESSERACT_CMD`: Path to Tesseract executable

**Grid Detection:**
- `ROW_HEIGHT_MIN`: Minimum row height in pixels (default: 30)
- `COL_WIDTH_MIN`: Minimum column width in pixels (default: 25)
- `GRID_CLUSTER_TOLERANCE`: Tolerance for clustering grid lines (default: 12)

**Bubble Detection:**
- `MIN_FILL_DENSITY`: Threshold for bubble fill detection, 0.03-0.08 (default: 0.05)
- `MIN_INNER_DIFF`: Minimum intensity difference for answer selection (default: 5)
- `MAX_SECOND_RATIO`: Maximum ratio for double marking detection (default: 0.65)

**Profiling (profiling version only):**
- `OCR_MAX_WORKERS`: Number of parallel OCR workers (default: 4)
- `ENABLE_DEBUG_IMAGES`: Generate debug images (true/false)
- `PROFILE_DETAILED`: Enable detailed profiling metrics (true/false)

### Direct Script Configuration

Alternatively, you can edit parameters directly at the top of each script file.

## 📝 Supported Question Formats

The extractor handles various question numbering formats:
- Simple sequential: `1, 2, 3, 4, ...`
- Multi-part questions: `35-A, 35-B, 36-A, 36-B, ...`
- Mixed formats: `1, 2, 3, 47, 5, 64-A, 64-B, ...`

## 🤝 Contributing

See `AGENTS.md` for development guidelines and architecture details.

## 📄 License

[Add your license information here]
