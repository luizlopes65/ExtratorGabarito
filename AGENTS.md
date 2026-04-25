# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

This repository contains Python scripts for digitizing multiple-choice answer sheets using OpenCV-based image processing plus OCR with Tesseract. The codebase supports two versions of the extraction pipeline:

- **Standard version** (`extrair_table_fixed.py`): Full-featured extraction with debug image generation
- **Profiling version** (`extrair_table_profiling.py`): Same features plus parallel OCR processing and detailed performance metrics

Both versions share the same core pipeline but differ in execution strategy and diagnostic output.

The project is script-oriented rather than package-oriented: there is no importable application package yet, and execution happens by running Python files directly or through the `main.py` CLI entry point.

Core technologies:

- Python 3.10+
- OpenCV (`cv2`) for image preprocessing, geometry correction, thresholding, contour analysis, and bubble detection
- Tesseract OCR via `pytesseract`
- NumPy for numeric operations
- Pandas for tabular result generation and CSV export
- Poetry for dependency management, with `requirements.txt` also present for pip-based installs

## High-Level Architecture

The extraction pipeline follows this flow:

1. Load an input image from a configured `IMAGE_PATH`.
2. Preprocess the document:
   - grayscale conversion
   - contour-based document detection
   - perspective correction
   - skew estimation and rotation correction
3. Detect table/grid lines with adaptive thresholding and morphology.
4. Derive column and row intervals from clustered line positions.
5. Use OCR to extract headers and student names:
   - Enhanced pattern matching for multi-part questions (35-A, 35-B, etc.)
   - Intelligent error correction using sequence inference
6. For each answer cell, detect the selected bubble using density and intensity heuristics.
7. Build `pandas.DataFrame` outputs for answers, confidence scores, and fill densities.
8. Save CSV outputs and intermediate debug images.

## Recent Improvements (2026-04)

### Question Pairing Fix

The pipeline now correctly handles multi-part questions with letter suffixes:

1. **Enhanced `clean_question_header()`**: Extracts the rightmost complete pattern when OCR returns multiple matches (e.g., "36 36-B" → "36-B")
2. **Improved `ocr_text_block()`**: Added fallback to raw grayscale when adaptive thresholding fails
3. **Pattern-based inference**: Automatically corrects OCR errors by detecting question number sequences:
   - "36-A" followed by "5" (OCR error) → inferred as "36-B"
   - "38-B" followed by "38" → inferred as "38-C"
4. **Removed deduplication logic**: Headers with letter suffixes are already unique

These changes ensure correct extraction for answer sheets with formats like:
- `35-A, 35-B, 36-A, 36-B, 37, 38-A, 38-B, 38-C, 39, 40`
- `1, 2, 3, 47, 5, 64-A, 64-B`

## Important Files

### Core Scripts
- `main.py`: CLI entry point with argument parsing for running either version
- `extrair_table_fixed.py`: Standard OCR/OMR pipeline with debug artifact generation
- `extrair_table_profiling.py`: Profiling version with parallel OCR and performance metrics

### Configuration & Dependencies
- `pyproject.toml`: Python version and Poetry dependencies (authoritative)
- `requirements.txt`: Pip install path (should be kept in sync with pyproject.toml)

### Documentation
- `README.md`: User-facing documentation with usage examples
- `AGENTS.md`: This file - development guidelines for AI agents
- `MILESTONES.md`: Project milestones and progress tracking

### Directories
- `examples/`: Sample answer sheet images
- `resultados/`: Output CSV files (answers, confidence, density)
- `debug/`: Diagnostic image outputs for troubleshooting
- `archive/`: Archived/unused code (formerly `chest/`)

## Building and Running

### Environment setup

Preferred with Poetry:

```bash
poetry install
```

Alternative with pip:

```bash
pip install -r requirements.txt
```

### External dependency

Tesseract OCR must be installed on the machine. On macOS:

```bash
brew install tesseract
```

The scripts expect Tesseract at `/opt/homebrew/bin/tesseract`. Update `TESSERACT_CMD` if installed elsewhere.

### Running the extractor

**Recommended: Use main.py CLI**

```bash
# Standard version
poetry run python main.py

# Profiling version
poetry run python main.py --profile

# Custom image
poetry run python main.py --image examples/my_sheet.png
```

**Direct script execution**

```bash
# Standard version
poetry run python extrair_table_fixed.py

# Profiling version
poetry run python extrair_table_profiling.py
```

Before running directly, verify the constants near the top of the file:
- `IMAGE_PATH`
- `OUTPUT_CSV`
- `DEBUG_DIR`
- `TESSERACT_CMD`
- Threshold/tolerance values if tuning is needed

### Testing

There is no automated test suite configured at this time.

**TODO**: Add repeatable regression tests using a small fixture set of input images plus expected CSV outputs.

## Development Conventions

### Code organization

- The repository is organized around standalone scripts, not reusable packages.
- Configuration is kept as module-level constants near the top of each script.
- The pipeline is decomposed into focused helper functions for preprocessing, OCR, grid detection, and bubble classification.
- Data passed between stages uses `dataclass` structures: `OCRBox` and `CellResult`.
- Both versions share nearly identical logic; profiling version adds `@profile` decorators and parallel execution.

### Output and diagnostics

- **Debug-first workflow**: Both scripts write intermediate images to the configured debug directory for troubleshooting OCR/grid-detection failures.
- When adjusting image-processing thresholds, keep debug artifact generation intact so changes can be validated visually.
- **CSV outputs** (3 files per run):
  - Main extracted answers
  - Confidence values (0.0-1.0)
  - Density values (0.0-1.0)

### Configuration expectations

- Paths are hard-coded in scripts but can be overridden via `main.py` CLI arguments.
- If extending the project, prefer enhancing the CLI interface over scattering new constants across files.
- Keep Tesseract path handling explicit; the code sets `pytesseract.pytesseract.tesseract_cmd` when configured.

### Dependency management

- `pyproject.toml` is the authoritative dependency definition.
- `requirements.txt` should be kept in sync. If you add or update dependencies, update both files.

### Language and naming

- Code and inline comments are primarily in Portuguese.
- Keep naming and comments consistent with the surrounding file instead of translating only part of a module.
- User-facing printed output is also in Portuguese; maintain that convention unless the repository is being intentionally internationalized.

## Repository-Specific Notes for Agents

### Critical Considerations

- **OCR/bubble-detection heuristics**: Small threshold changes can materially affect extraction accuracy. Test thoroughly when modifying:
  - `MIN_FILL_DENSITY` (0.03-0.08)
  - `MIN_INNER_DIFF` (intensity difference threshold)
  - `ROW_HEIGHT_MIN` / `COL_WIDTH_MIN` (grid detection)
  
- **Pattern inference logic**: The question header inference (lines 719-764 in both scripts) is carefully tuned to handle OCR errors without over-correcting. Changes should preserve the logic for:
  - Detecting "N-X" followed by "N" → "N-Y" pattern
  - Single-digit misreads in multi-part sequences
  - Avoiding false positives on simple sequential numbers

- **Parallel execution**: The profiling version uses `ThreadPoolExecutor` for OCR operations. Maintain thread-safety when modifying OCR functions.

### Best Practices

- **Favor small, targeted edits** over broad rewrites in the detection pipeline, because the current behavior is tightly coupled to scanned form characteristics.
- **When modifying output paths**, ensure required directories exist or are created before writing.
- **If introducing new automation**, prioritize reproducibility around image fixtures, calibration data, and expected CSV outputs.
- **Debug images are essential**: Always generate them when troubleshooting extraction issues.

### Known Limitations

- No automated tests yet
- Configuration is script-level constants (no config file)
- No batch processing support (processes one image at a time)
- Template-based extraction code is archived but not integrated with main pipeline

### Future Improvements

Consider these when extending the codebase:
- Add pytest-based regression tests with fixture images
- Implement batch processing mode
- Create a configuration file system (YAML/JSON)
- Package the code as an installable module
- Add support for different answer sheet layouts
- Implement a web interface for easier usage