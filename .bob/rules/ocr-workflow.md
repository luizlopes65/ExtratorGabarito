# OCR Project Workflow Rules

## Development Workflow

### Primary Development File
- **Always work on `extrair_table_fixed.py` first**
- This is the main development file for testing and implementing new features
- Only after confirming changes work correctly in the fixed version, propagate them to `extrair_table_profiling.py`

### Testing Protocol
- **Ground truth test image**: `examples/image.png`
- Always test changes against this reference image before considering the work complete
- This image should be used as the baseline for validating extraction accuracy

### File Synchronization
1. Make changes to `extrair_table_fixed.py`
2. Test with `examples/image.png`
3. Verify results (CSV outputs, debug images)
4. Only after successful validation, apply the same changes to `extrair_table_profiling.py`
5. The profiling version should maintain identical logic, only adding `@profile` decorators

### Testing Command
```bash
poetry run python main.py --image examples/subset/image.png
```

### Validation Checklist
Before propagating changes to the profiling version:
- [ ] Code runs without errors
- [ ] Header row is correctly identified
- [ ] Question numbers are properly extracted
- [ ] Student names are recognized
- [ ] Answer bubbles are accurately detected
- [ ] CSV outputs are generated correctly
- [ ] Debug images show expected results

## Rationale
This workflow ensures:
- Faster iteration (no need to maintain two files simultaneously)
- Cleaner testing (single source of truth during development)
- Reduced errors (changes are validated before duplication)
- Consistent behavior between standard and profiling versions