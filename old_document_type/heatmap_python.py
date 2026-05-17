"""
Student performance heatmap generator.

Usage:
    python heatmap_performance.py banco.csv resultado.csv [output.png]

Scoring rules:
    2 or 3  → hit  (student got it right)
    1 or B  → miss (student got it wrong)
    blank   → skip (not counted)

banco.csv structure:
    - "Questão"     : question filename, e.g. "06.pdf"
    - "Itens"       : optional item letters, e.g. "A,B,C"
    - "Assunto"     : category (top-level grouping)
    - subcategory columns: marked with 1.0 when applicable

resultado.csv structure:
    - "ID"          : student identifier
    - question cols : named as "6", "6-A", "6-B", "47", etc.
"""

import sys
import re
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

warnings.filterwarnings("ignore")

# ── Subcategory columns present in banco ─────────────────────────────────────

SUBCAT_COLS = [
    "Un.", "Dez.", "Cent", "Milhar", "Dezena de milhar",
    "Significado primário", "Significado secundário",
    "Algoritmo intermediário", "Algoritmo final",
    "Fatos básicos e dedutivos", "Cálculo mental",
    "Sim", "Não",
    "Material dourado", "Representação concreta", "Reta numérica",
    "Gráficos", "Tabelas", "Sistema decimal",
    "Compor e decompor número", "Reconhecimento de padrões",
    "Adição", "Subtração", "Multiplicação", "Divisão",
    "Representação Visual", "Escrita por Extenso", "Reta Numérica",
    "Comparação", "Problema com Contexto",
    "Frações Equivalentes", "Números Mistos", "Adição/Subtração",
    "Multiplicação.1", "Divisão.1", "Conversão fração ↔ decimal",
    "Fração", "Decimal",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_q_num(questao: str):
    m = re.search(r"\d+", str(questao))
    return int(m.group()) if m else None


def parse_result_col(col: str):
    col = str(col)
    if "-" in col:
        num_part, item = col.split("-", 1)
        return int(num_part), item.strip().upper()
    return int(col), None


def score_value(val):
    """Return the numeric score (2 or 3), 0 for a miss (1/B), None to skip."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    v = str(val).strip().upper()
    if v == "3":
        return 3
    if v == "2":
        return 2
    if v in ("1", "B"):
        return 0
    return None


# ── Core pipeline ─────────────────────────────────────────────────────────────

def build_score_table(banco_path: str, resultado_path: str):
    banco = pd.read_csv(banco_path)
    resultado = pd.read_csv(resultado_path)

    present_subcats = [c for c in SUBCAT_COLS if c in banco.columns]
    banco["_qnum"] = banco["Questão"].apply(extract_q_num)

    result_cols = [c for c in resultado.columns if c != "ID"]
    students = resultado["ID"].tolist()

    skipped = []
    col_meta = {}

    for col in result_cols:
        qnum, item = parse_result_col(col)
        matches = banco[banco["_qnum"] == qnum]
        if item:
            matches = matches[matches["Itens"].fillna("").str.contains(item)]
        if matches.empty:
            skipped.append({"col": col, "reason": "question not found in banco"})
            col_meta[col] = None
            continue
        row = matches.iloc[0]
        active = [sc for sc in present_subcats if pd.notna(row.get(sc))]
        if not active:
            skipped.append({"col": col, "reason": "no subcategory tags"})
            col_meta[col] = None
            continue
        col_meta[col] = {"assunto": row["Assunto"].strip(), "subcats": active}

    # accumulate — store [score_sum, answer_count] per (assunto, subcat)
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_student = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))

    for _, srow in resultado.iterrows():
        sid = srow["ID"]
        for col in result_cols:
            meta = col_meta.get(col)
            if meta is None:
                continue
            score = score_value(srow[col])
            if score is None:
                continue
            assunto = meta["assunto"]
            for sc in meta["subcats"]:
                agg[assunto][sc][0] += score
                agg[assunto][sc][1] += 1
                per_student[sid][assunto][sc][0] += score
                per_student[sid][assunto][sc][1] += 1

    # agg_df: MultiIndex (assunto, subcat), columns score/total
    agg_records = []
    for assunto, sdict in agg.items():
        for sc, (score_sum, total) in sdict.items():
            agg_records.append({
                "assunto": assunto, "subcat": sc,
                "score": score_sum, "total": total,
            })
    agg_df = pd.DataFrame(agg_records).set_index(["assunto", "subcat"])

    # student_df: MultiIndex (assunto, subcat), columns = student IDs, values = score sums
    student_records = defaultdict(dict)
    for sid in students:
        sd = per_student.get(sid, {})
        for assunto, sdict in sd.items():
            for sc, (score_sum, total) in sdict.items():
                student_records[(assunto, sc)][sid] = score_sum

    student_df = pd.DataFrame(student_records).T
    student_df.index = pd.MultiIndex.from_tuples(student_df.index, names=["assunto", "subcat"])
    student_df = student_df.reindex(columns=students)

    return agg_df, student_df, skipped, students


# ── Colour helpers ────────────────────────────────────────────────────────────

CMAP = LinearSegmentedColormap.from_list(
    "perf", ["#D85A30", "#FAC775", "#1D9E75"], N=256
)


def _score_label(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "–"
    return str(int(round(val)))


def _text_color(norm_val) -> str:
    """norm_val is 0–1 for colour-map position."""
    if norm_val is None or (isinstance(norm_val, float) and np.isnan(norm_val)):
        return "#aaa"
    return "white" if norm_val < 0.35 or norm_val > 0.75 else "#3a2000"


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_section(ax, subcats, matrix, col_labels, show_col_labels, font_size, vmin, vmax):
    """Draw one assunto block onto ax. matrix values are raw score sums."""
    n_sub = len(subcats)
    n_col = len(col_labels)
    score_range = max(vmax - vmin, 1)

    ax.set_xlim(0, n_col)
    ax.set_ylim(0, n_sub)

    for r, sc in enumerate(subcats):
        for c in range(n_col):
            val = float(matrix[r, c])
            if np.isnan(val):
                facecolor = "#f0efea"
                norm = None
            else:
                norm = (val - vmin) / score_range
                facecolor = CMAP(norm)
            rect = mpatches.FancyBboxPatch(
                (c + 0.04, n_sub - 1 - r + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02",
                linewidth=0,
                facecolor=facecolor,
                transform=ax.transData,
                clip_on=True,
            )
            ax.add_patch(rect)
            lbl = _score_label(None if np.isnan(val) else val)
            ax.text(
                c + 0.5, n_sub - 0.5 - r, lbl,
                ha="center", va="center",
                fontsize=font_size - 1,
                color=_text_color(norm),
                fontweight="bold",
            )

    ax.set_xticks([i + 0.5 for i in range(n_col)])
    if show_col_labels:
        ax.set_xticklabels(col_labels, fontsize=font_size - 1, rotation=35, ha="right", color="#555")
        ax.xaxis.set_tick_params(length=0)
    else:
        ax.set_xticklabels([])
        ax.xaxis.set_tick_params(length=0)

    ax.set_yticks([n_sub - 0.5 - r for r in range(n_sub)])
    ax.set_yticklabels(subcats, fontsize=font_size - 1, color="#555")
    ax.yaxis.set_tick_params(length=0)
    ax.set_frame_on(False)


def plot_heatmap(
    agg_df: pd.DataFrame,
    student_df: pd.DataFrame,
    skipped: list,
    students: list,
    output_path: str = "heatmap_performance.png",
    mode: str = "aggregate",
    font_size: int = 9,
) -> None:
    assuntos = agg_df.index.get_level_values("assunto").unique().tolist()

    if mode == "aggregate":
        sections = []
        for assunto in assuntos:
            sub = agg_df.loc[assunto]
            subcats = sub.index.tolist()
            matrix = sub["score"].values.reshape(-1, 1).astype(float)
            sections.append((assunto, subcats, matrix, [assunto]))
        col_width = 1.4
    else:
        sections = []
        for assunto in assuntos:
            try:
                sub = student_df.loc[assunto]
            except KeyError:
                continue
            subcats = sub.index.tolist()
            matrix = sub.reindex(columns=students).values.astype(float)
            sections.append((assunto, subcats, matrix, students))
        col_width = max(0.5, min(1.0, 5.0 / max(len(students), 1)))

    if not sections:
        print("No data to plot.")
        return

    # global score range for a consistent colour scale across sections
    all_vals = np.concatenate([s[2].ravel() for s in sections])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = float(all_vals.min()) if len(all_vals) else 0
    vmax = float(all_vals.max()) if len(all_vals) else 1

    # figure sizing
    max_cols = max(len(s[3]) for s in sections)
    label_w = 2.8
    fig_w = label_w + max_cols * col_width + 0.6
    row_h = 0.44
    total_rows = sum(len(s[1]) for s in sections)
    hspace_total = len(sections) * 0.9
    fig_h = total_rows * row_h + hspace_total + 1.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    title_str = "class total score" if mode == "aggregate" else "per-student total score"
    fig.suptitle(
        f"Student performance heatmap — {title_str}",
        fontsize=font_size + 3, fontweight="bold",
        y=1.0, va="top", color="#222",
    )

    height_ratios = [len(s[1]) for s in sections]
    gs = fig.add_gridspec(len(sections), 1, hspace=1.1, height_ratios=height_ratios)

    for idx, (assunto, subcats, matrix, col_labels) in enumerate(sections):
        ax = fig.add_subplot(gs[idx])
        show_top = (mode == "students") or (idx == 0)
        _draw_section(ax, subcats, matrix, col_labels, show_top, font_size, vmin, vmax)
        ax.set_title(assunto, fontsize=font_size, fontweight="bold",
                     loc="left", color="#333", pad=3)

    # colour-bar legend — show actual score range
    legend_ax = fig.add_axes([0.15, -0.025, 0.7, 0.015])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    legend_ax.imshow(gradient, aspect="auto", cmap=CMAP)
    legend_ax.set_yticks([])
    tick_scores = np.linspace(vmin, vmax, 5)
    legend_ax.set_xticks(np.linspace(0, 255, 5))
    legend_ax.set_xticklabels([str(int(round(v))) for v in tick_scores],
                               fontsize=font_size - 1, color="#555")
    legend_ax.tick_params(length=0)
    legend_ax.set_frame_on(False)

    if skipped:
        note = "Skipped (no tags): " + ", ".join(s["col"] for s in skipped)
        fig.text(0.5, -0.05, note, ha="center",
                 fontsize=font_size - 2, color="#aaa", style="italic")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output_path}")
    plt.close(fig)


# ── CLI / importable entry point ──────────────────────────────────────────────

def main(banco_path: str, resultado_path: str, output_path: str = "heatmap_performance.png") -> None:
    print(f"Reading banco:     {banco_path}")
    print(f"Reading resultado: {resultado_path}\n")

    agg_df, student_df, skipped, students = build_score_table(banco_path, resultado_path)

    if skipped:
        print(f"⚠  {len(skipped)} question(s) skipped:")
        for s in skipped:
            print(f"   col={s['col']!r:10s}  reason: {s['reason']}")
        print()

    assuntos = agg_df.index.get_level_values("assunto").unique().tolist()
    subcats  = agg_df.index.get_level_values("subcat").unique().tolist()
    total_score = int(agg_df["score"].sum())
    print(f"Subjects found : {assuntos}")
    print(f"Subcategories  : {subcats}")
    print(f"Total score    : {total_score}\n")

    stem   = Path(output_path).stem
    suffix = Path(output_path).suffix or ".png"
    parent = Path(output_path).parent

    plot_heatmap(agg_df, student_df, skipped, students,
                 output_path=str(parent / f"{stem}_aggregate{suffix}"),
                 mode="aggregate")
    plot_heatmap(agg_df, student_df, skipped, students,
                 output_path=str(parent / f"{stem}_students{suffix}"),
                 mode="students")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python heatmap_performance.py banco.csv resultado.csv [output.png]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "heatmap_performance.png")