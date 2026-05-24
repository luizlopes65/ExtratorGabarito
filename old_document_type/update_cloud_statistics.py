#!/usr/bin/env python3
"""
Update Google Sheets with aggregated student performance statistics.

Reads master_table.csv and matriz_assuntos_subatributos_populated.csv,
calculates statistics, and updates Google Sheets.

Scoring: B=0, 1=0, 2 or 3=+1
"""

import pandas as pd
import csv
import gspread
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Import functions from google_sheets_utils.py
from google_sheets_utils import (
    obter_coordenada,
    resolver_ancora,
    col_letra_para_indice,
    indice_para_col_letra
)


def parse_question_mappings(csv_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse matriz CSV. Map questions to (subject, attribute) pairs.
    
    Returns: {"2": [("Adição", "Un.")], "93": [("Adição", "Un.")], ...}
    """
    mappings = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            subject = row['Assunto']
            
            for attribute, value in row.items():
                if attribute == 'Assunto' or not value.strip():
                    continue
                
                # Parse comma-separated question numbers
                questions = [q.strip() for q in value.split(',')]
                
                for q in questions:
                    if q:
                        mappings[q].append((subject, attribute))
    
    print(f"📊 Mapped {len(mappings)} questions to cells")
    return dict(mappings)


def calculate_question_statistics(master_path: str) -> Dict[str, int]:
    """
    Calculate correct answers per question.
    Scoring: B=0, 1=0, 2 or 3=+1
    
    Returns: {"1": 15, "2": 12, ...}
    """
    df = pd.read_csv(master_path, encoding='utf-8-sig')
    
    # Get question columns (skip metadata)
    metadata_cols = ['arquivo_de_origem', 'turma', 'escola', 'bimestre', 'prof', 'pagina', 'ID_aluno']
    question_cols = [col for col in df.columns if col not in metadata_cols]
    
    stats = {}
    
    for col in question_cols:
        # Count correct answers (2 or 3)
        correct = df[col].isin(['2', '3', 2, 3]).sum()
        stats[col] = int(correct)
    
    print(f"📈 Calculated stats for {len(stats)} questions")
    return stats


def aggregate_by_cell(
    question_stats: Dict[str, int],
    question_mappings: Dict[str, List[Tuple[str, str]]]
) -> Dict[Tuple[str, str], int]:
    """
    Aggregate scores by (subject, attribute) cell.
    
    Returns: {("Adição", "Un."): 27, ...}
    """
    cell_stats = defaultdict(int)
    unmapped = []
    
    for question, score in question_stats.items():
        if question in question_mappings:
            for subject, attribute in question_mappings[question]:
                cell_stats[(subject, attribute)] += score
        else:
            unmapped.append(question)
    
    if unmapped:
        print(f"⚠️  {len(unmapped)} unmapped questions: {unmapped[:5]}...")
    
    print(f"✅ Aggregated into {len(cell_stats)} cells")
    return dict(cell_stats)


def batch_update_sheets(
    aba,
    cell_statistics: Dict[Tuple[str, str], int],
    batch_size: int = 100
):
    """
    Update Google Sheets using batch API. Single request.
    """
    total = len(cell_statistics)
    failed = []
    
    print(f"\n🔄 Preparing {total} cells for batch update...")
    
    # Prepare batch data
    batch_data = []
    
    for (subject, attribute), value in cell_statistics.items():
        try:
            # Get coordinate
            coord = obter_coordenada(subject, attribute)
            
            # Resolve merged cell anchor
            coord = resolver_ancora(aba, coord)
            
            batch_data.append({
                'range': coord,
                'values': [[value]]
            })
            
            print(f"  📝 {coord} ({subject} -> {attribute}): {value}")
            
        except Exception as e:
            print(f"  ❌ Failed prep ({subject}, {attribute}): {e}")
            failed.append((subject, attribute, str(e)))
    
    # Execute batch update (single API call)
    if batch_data:
        print(f"\n🚀 Executing batch update ({len(batch_data)} cells)...")
        try:
            aba.batch_update(batch_data, value_input_option='RAW')
            success = len(batch_data)
            print(f"  ✅ Batch update complete!")
        except Exception as e:
            print(f"  ❌ Batch update failed: {e}")
            return 0, failed + [("BATCH", "UPDATE", str(e))]
    else:
        success = 0
    
    print(f"\n📊 Summary:")
    print(f"  Success: {success}/{total}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\n❌ Failed updates:")
        for subj, attr, err in failed[:10]:
            print(f"  - ({subj}, {attr}): {err}")
    
    return success, failed


def run_update(master_table_path: str, matrix_csv_path: str, credentials_path: str, sheet_id: str, worksheet_index: int = 2):
    """Run the cloud statistics update pipeline."""
    print("=" * 60)
    print("📊 UPDATE CLOUD STATISTICS")
    print("=" * 60)
    
    # Step 1: Parse question mappings
    print("\n1️⃣  Parsing question mappings...")
    question_mappings = parse_question_mappings(matrix_csv_path)
    
    # Step 2: Calculate statistics
    print("\n2️⃣  Calculating question statistics...")
    question_stats = calculate_question_statistics(master_table_path)
    
    # Step 3: Aggregate by cell
    print("\n3️⃣  Aggregating by cell...")
    cell_stats = aggregate_by_cell(question_stats, question_mappings)
    
    # Step 4: Connect to Google Sheets
    print("\n4️⃣  Connecting to Google Sheets...")
    try:
        gc = gspread.service_account(filename=credentials_path)
        sheet = gc.open_by_key(sheet_id)
        aba = sheet.get_worksheet(worksheet_index)
        print("  ✅ Connected!")
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        return
    
    # Step 5: Update sheets (single batch request)
    print("\n5️⃣  Updating Google Sheets (single batch API call)...")
    success, failed = batch_update_sheets(aba, cell_stats)
    
    print("\n" + "=" * 60)
    print("✅ UPDATE COMPLETE!")
    print("=" * 60)


def main():
    """CLI execution."""
    # Default Paths
    MASTER_TABLE = 'my_results/master_table.csv'
    MATRIX_CSV = 'matriz_assuntos_subatributos_populated.csv'
    CREDENTIALS = 'credenciais.json'
    SHEET_ID = '1v21Q3TKPkJuf08HvwpMmZ6IYwa6Wsht2S4OvlKmenfc'
    WORKSHEET_INDEX = 2
    
    run_update(MASTER_TABLE, MATRIX_CSV, CREDENTIALS, SHEET_ID, WORKSHEET_INDEX)


if __name__ == '__main__':
    main()

# Made with Bob
