#!/usr/bin/env python3
"""
Generate average tables from master_table.csv.

Two aggregation modes:
  - by_turma : one row per (turma, bimestre, prof)
  - by_escola: one row per (escola, bimestre)

For every question column (subatributo):
  - Pontuação por resposta: B=0, 1=0, 2=1, 3=2  (máx = 2 por aluno)
  - numerador   = soma dos pontos obtidos pelo grupo nesse subatributo
  - denominador = n_alunos_considerados × 2
  - média       = numerador / denominador   (entre 0.0 e 1.0)

Output CSV columns (example for by_turma):
  turma, escola, bimestre, prof,
  Q1_num, Q1_den, Q1_avg,
  Q2_num, Q2_den, Q2_avg, ...
"""

import pandas as pd
from pathlib import Path
from typing import Literal

METADATA_COLS = ['arquivo_de_origem', 'turma', 'escola', 'bimestre', 'prof', 'pagina', 'ID_aluno']
MAX_SCORE_PER_ITEM = 2  # pontuação máxima por questão por aluno


def score_value(val) -> int:
    """Convert a raw answer value to its score (B=0, 1=0, 2=1, 3=2)."""
    val_str = str(val).strip()
    if val_str == '2':
        return 1
    if val_str == '3':
        return 2
    return 0  # 'B', '1', NaN, empty → 0


def _compute_group_stats(group: pd.DataFrame, question_cols: list) -> dict:
    """
    For a group of rows (same turma or same escola), compute per-question stats.

    Returns a flat dict:
      { 'Qx_num': int, 'Qx_den': int, 'Qx_avg': float, ... }
    """
    n_alunos = len(group)
    row = {}
    for col in question_cols:
        numerador = group[col].apply(score_value).sum()
        denominador = n_alunos * MAX_SCORE_PER_ITEM
        media = round(numerador / denominador, 4) if denominador > 0 else None
        row[f"{col}_num"] = int(numerador)
        row[f"{col}_den"] = int(denominador)
        row[f"{col}_avg"] = media
    return row


def generate_averages_by_turma(master_path: str, output_file: str = None) -> str:
    """
    Generate a CSV with one row per (turma, escola, bimestre, prof).

    Args:
        master_path : path to master_table.csv
        output_file : output CSV path (defaults to same dir as master_table)

    Returns:
        Path of the generated CSV.
    """
    df = pd.read_csv(master_path, encoding='utf-8-sig')

    question_cols = [c for c in df.columns if c not in METADATA_COLS]
    if not question_cols:
        print("❌ Nenhuma coluna de questão encontrada na master table.")
        return None

    group_keys = [k for k in ['turma', 'escola', 'bimestre', 'prof'] if k in df.columns]

    rows = []
    for keys, group in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_keys, keys))
        meta['n_alunos'] = len(group)
        meta.update(_compute_group_stats(group, question_cols))
        rows.append(meta)

    result_df = pd.DataFrame(rows)

    # Reorder: group keys + n_alunos + question stats
    stat_cols = []
    for col in question_cols:
        stat_cols += [f"{col}_num", f"{col}_den", f"{col}_avg"]
    ordered_cols = group_keys + ['n_alunos'] + stat_cols
    result_df = result_df[[c for c in ordered_cols if c in result_df.columns]]

    if output_file is None:
        output_file = str(Path(master_path).parent / "media_por_turma.csv")

    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ Média por turma gerada: {output_file}")
    print(f"   Turmas encontradas : {len(result_df)}")
    print(f"   Subatributos       : {len(question_cols)}")
    print(f"   Colunas geradas    : {len(result_df.columns)} "
          f"({len(question_cols)} × 3 + {len(group_keys) + 1} meta)")
    return output_file


def generate_averages_by_escola(master_path: str, output_file: str = None) -> str:
    """
    Generate a CSV with one row per (escola, bimestre).

    Args:
        master_path : path to master_table.csv
        output_file : output CSV path (defaults to same dir as master_table)

    Returns:
        Path of the generated CSV.
    """
    df = pd.read_csv(master_path, encoding='utf-8-sig')

    question_cols = [c for c in df.columns if c not in METADATA_COLS]
    if not question_cols:
        print("❌ Nenhuma coluna de questão encontrada na master table.")
        return None

    group_keys = [k for k in ['escola', 'bimestre'] if k in df.columns]

    rows = []
    for keys, group in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_keys, keys))
        meta['n_alunos'] = len(group)
        meta['n_turmas'] = group['turma'].nunique() if 'turma' in group.columns else None
        meta.update(_compute_group_stats(group, question_cols))
        rows.append(meta)

    result_df = pd.DataFrame(rows)

    # Reorder: group keys + n_alunos + n_turmas + question stats
    stat_cols = []
    for col in question_cols:
        stat_cols += [f"{col}_num", f"{col}_den", f"{col}_avg"]
    ordered_cols = group_keys + ['n_alunos', 'n_turmas'] + stat_cols
    result_df = result_df[[c for c in ordered_cols if c in result_df.columns]]

    if output_file is None:
        output_file = str(Path(master_path).parent / "media_por_escola.csv")

    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ Média por escola gerada: {output_file}")
    print(f"   Escolas encontradas : {len(result_df)}")
    print(f"   Subatributos        : {len(question_cols)}")
    print(f"   Colunas geradas     : {len(result_df.columns)} "
          f"({len(question_cols)} × 3 + {len(group_keys) + 2} meta)")
    return output_file


def generate_both(master_path: str) -> tuple:
    """Convenience wrapper that generates both aggregation CSVs."""
    turma_csv  = generate_averages_by_turma(master_path)
    escola_csv = generate_averages_by_escola(master_path)
    return turma_csv, escola_csv
