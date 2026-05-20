#!/usr/bin/env python3
"""
Create master table from batch extraction results.

Combines all individual resultado.csv files with their QR metadata
into a single master table.

Output columns:
    arquivo_de_origem, turma, escola, bimestre, prof, pagina, ID_aluno, Q1, Q2, ..., Q103
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import sys


def load_page_data(page_dir: Path) -> Dict:
    """
    Load data from a single page directory.
    
    Returns dict with:
        - metadata: QR metadata from JSON
        - results: DataFrame with student answers
        - source: source PDF name
    """
    # Load metadata
    metadata_file = page_dir / "resultado_metadata.json"
    if not metadata_file.exists():
        print(f"  [WARN] No metadata found: {page_dir.name}")
        metadata = {}
    else:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Load results
    results_file = page_dir / "resultado.csv"
    if not results_file.exists():
        print(f"  [WARN] No results found: {page_dir.name}")
        return None
    
    df = pd.read_csv(results_file)
    
    # Extract source PDF name from directory name
    # Format: "pdf_name_pageN" -> "pdf_name"
    dir_name = page_dir.name
    if '_page' in dir_name:
        source = dir_name.rsplit('_page', 1)[0]
    else:
        source = dir_name
    
    return {
        'metadata': metadata,
        'results': df,
        'source': source
    }


def create_master_table(results_dir: str, output_file: str = None) -> str:
    """
    Create master table from all batch results.
    
    Args:
        results_dir: Directory containing batch results
        output_file: Output CSV path (default: results_dir/master_table.csv)
    
    Returns:
        Path to created master table
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return None
    
    # Find all page directories
    page_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    if not page_dirs:
        print(f"Error: No page directories found in {results_dir}")
        return None
    
    print(f"Found {len(page_dirs)} page directories")
    
    # Collect all rows
    all_rows = []
    
    for page_dir in sorted(page_dirs):
        data = load_page_data(page_dir)
        if data is None:
            continue
        
        metadata = data['metadata']
        df = data['results']
        source = data['source']
        
        # Get student ID column name (could be 'ID' or 'Nome')
        id_col = 'ID' if 'ID' in df.columns else 'Nome'
        
        # For each student in this page
        for _, row in df.iterrows():
            # Create master row
            master_row = {
                'arquivo_de_origem': source,
                'turma': metadata.get('ano_escolar', ''),
                'escola': metadata.get('id_escola', ''),
                'bimestre': metadata.get('bimestre', ''),
                'prof': metadata.get('id_prof', ''),
                'pagina': metadata.get('pagina', ''),
                'ID_aluno': row[id_col]
            }
            
            # Add all question columns (Q1, Q2, etc.)
            for col in df.columns:
                if col != id_col:  # Skip ID/Nome column
                    # Rename column to Q format if needed
                    if col.startswith('Q') or col.isdigit() or '-' in col:
                        master_row[col] = row[col]
                    else:
                        master_row[col] = row[col]
            
            all_rows.append(master_row)
    
    if not all_rows:
        print("Error: No data found to compile")
        return None
    
    # Create master DataFrame
    master_df = pd.DataFrame(all_rows)
    
    # Sort columns: metadata first, then questions in order
    metadata_cols = ['arquivo_de_origem', 'turma', 'escola', 'bimestre', 'prof', 'pagina', 'ID_aluno']
    question_cols = sorted([c for c in master_df.columns if c not in metadata_cols])
    
    master_df = master_df[metadata_cols + question_cols]
    
    # Save master table
    if output_file is None:
        output_file = results_path / "master_table.csv"
    else:
        output_file = Path(output_file)
    
    master_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Master table created: {output_file}")
    print(f"   Total rows: {len(master_df)}")
    print(f"   Total columns: {len(master_df.columns)}")
    print(f"   Metadata columns: {len(metadata_cols)}")
    print(f"   Question columns: {len(question_cols)}")
    
    return str(output_file)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python create_master_table.py <results_dir> [output_file]")
        print("\nExample:")
        print("  python create_master_table.py my_results/")
        print("  python create_master_table.py my_results/ master.csv")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    master_file = create_master_table(results_dir, output_file)
    
    if master_file is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

 
