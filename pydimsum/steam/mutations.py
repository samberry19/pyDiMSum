"""Identify and annotate single and double mutation variants.

Mirrors:
  R/dimsum__identify_single_aa_mutations.R
  R/dimsum__identify_double_aa_mutations.R
  R/dimsum__identify_single_nt_mutations.R
  R/dimsum__identify_double_nt_mutations.R

Key improvement: instead of per-row strsplit in R, we use the NumPy
mutation_positions function to find differing positions vectorized.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from pydimsum.steam.sequences import encode, mutation_positions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_singles(
    df: pl.DataFrame,
    sequence_type: str,
    wt_nt_seq: str,
    wt_aa_seq: str,
) -> pl.DataFrame:
    """Return a DataFrame of single-mutant variants with mutation annotations.

    For coding sequences: annotates on AA level (Nham_aa == 1).
    For noncoding: annotates on NT level (Nham_nt == 1).

    Columns added: Pos, WT_AA (or WT_nt), Mut.
    fitness/sigma columns are renamed: drop the ``_uncorr`` suffix.

    Mirrors:
      dimsum__identify_single_aa_mutations.R
      dimsum__identify_single_nt_mutations.R
    """
    if sequence_type == "coding":
        return _singles_aa(df, wt_aa_seq)
    else:
        return _singles_nt(df, wt_nt_seq)


def identify_doubles(
    df: pl.DataFrame,
    singles_df: pl.DataFrame,
    sequence_type: str,
    wt_nt_seq: str,
    wt_aa_seq: str,
) -> pl.DataFrame:
    """Return a DataFrame of double-mutant variants with mutation annotations.

    Columns added: Pos1, Pos2, WT_AA1/WT_AA2 (or WT_nt1/WT_nt2), Mut1, Mut2,
    s1_mean_count, s2_mean_count.

    Mirrors:
      dimsum__identify_double_aa_mutations.R
      dimsum__identify_double_nt_mutations.R
    """
    if sequence_type == "coding":
        return _doubles_aa(df, singles_df, wt_aa_seq)
    else:
        return _doubles_nt(df, singles_df, wt_nt_seq)


# ---------------------------------------------------------------------------
# AA-level singles
# ---------------------------------------------------------------------------

def _singles_aa(df: pl.DataFrame, wt_aa_seq: str) -> pl.DataFrame:
    singles = df.filter(pl.col("Nham_aa") == 1)
    if len(singles) == 0:
        return pl.DataFrame()

    aa_seqs = singles["aa_seq"].to_list()
    wt_aa_arr = np.frombuffer(wt_aa_seq.encode("ascii"), dtype=np.uint8)
    mat = encode(aa_seqs)

    pos_lists = mutation_positions(mat, wt_aa_arr)

    positions = [int(p[0]) for p in pos_lists]  # 1-based
    muts = [chr(int(mat[i, positions[i] - 1])) for i in range(len(singles))]
    wt_aas = [wt_aa_seq[positions[i] - 1] for i in range(len(singles))]

    singles = singles.with_columns([
        pl.Series("Pos", positions, dtype=pl.Int32),
        pl.Series("Mut", muts, dtype=pl.Utf8),
        pl.Series("WT_AA", wt_aas, dtype=pl.Utf8),
    ])

    # Rename fitness/sigma columns: drop _uncorr suffix
    singles = _drop_uncorr_suffix(singles)
    return singles


# ---------------------------------------------------------------------------
# NT-level singles
# ---------------------------------------------------------------------------

def _singles_nt(df: pl.DataFrame, wt_nt_seq: str) -> pl.DataFrame:
    singles = df.filter(pl.col("Nham_nt") == 1)
    if len(singles) == 0:
        return pl.DataFrame()

    seqs = singles["merge_seq"].to_list()
    wt_arr = np.frombuffer(wt_nt_seq.encode("ascii"), dtype=np.uint8)
    mat = encode(seqs)

    pos_lists = mutation_positions(mat, wt_arr)

    positions = [int(p[0]) for p in pos_lists]
    muts = [chr(int(mat[i, positions[i] - 1])) for i in range(len(singles))]
    wt_nts = [wt_nt_seq[positions[i] - 1] for i in range(len(singles))]

    singles = singles.with_columns([
        pl.Series("Pos", positions, dtype=pl.Int32),
        pl.Series("Mut", muts, dtype=pl.Utf8),
        pl.Series("WT_AA", wt_nts, dtype=pl.Utf8),  # WT_AA column name kept for nt too (R uses WT_AA then renames)
    ])

    singles = _drop_uncorr_suffix(singles)
    return singles


# ---------------------------------------------------------------------------
# AA-level doubles
# ---------------------------------------------------------------------------

def _doubles_aa(
    df: pl.DataFrame,
    singles_df: pl.DataFrame,
    wt_aa_seq: str,
) -> pl.DataFrame:
    doubles = df.filter(pl.col("Nham_aa") == 2)
    if len(doubles) == 0:
        return pl.DataFrame()

    aa_seqs = doubles["aa_seq"].to_list()
    wt_aa_arr = np.frombuffer(wt_aa_seq.encode("ascii"), dtype=np.uint8)
    mat = encode(aa_seqs)

    pos_lists = mutation_positions(mat, wt_aa_arr)

    pos1 = [int(p[0]) for p in pos_lists]
    pos2 = [int(p[1]) for p in pos_lists]
    mut1 = [chr(int(mat[i, pos1[i] - 1])) for i in range(len(doubles))]
    mut2 = [chr(int(mat[i, pos2[i] - 1])) for i in range(len(doubles))]
    wt1 = [wt_aa_seq[pos1[i] - 1] for i in range(len(doubles))]
    wt2 = [wt_aa_seq[pos2[i] - 1] for i in range(len(doubles))]

    doubles = doubles.with_columns([
        pl.Series("Pos1", pos1, dtype=pl.Int32),
        pl.Series("Pos2", pos2, dtype=pl.Int32),
        pl.Series("Mut1", mut1, dtype=pl.Utf8),
        pl.Series("Mut2", mut2, dtype=pl.Utf8),
        pl.Series("WT_AA1", wt1, dtype=pl.Utf8),
        pl.Series("WT_AA2", wt2, dtype=pl.Utf8),
    ])

    # Join mean_count from singles
    doubles = _join_singles_mean_count(doubles, singles_df)
    return doubles


# ---------------------------------------------------------------------------
# NT-level doubles
# ---------------------------------------------------------------------------

def _doubles_nt(
    df: pl.DataFrame,
    singles_df: pl.DataFrame,
    wt_nt_seq: str,
) -> pl.DataFrame:
    doubles = df.filter(pl.col("Nham_nt") == 2)
    if len(doubles) == 0:
        return pl.DataFrame()

    seqs = doubles["merge_seq"].to_list()
    wt_arr = np.frombuffer(wt_nt_seq.encode("ascii"), dtype=np.uint8)
    mat = encode(seqs)

    pos_lists = mutation_positions(mat, wt_arr)

    pos1 = [int(p[0]) for p in pos_lists]
    pos2 = [int(p[1]) for p in pos_lists]
    mut1 = [chr(int(mat[i, pos1[i] - 1])) for i in range(len(doubles))]
    mut2 = [chr(int(mat[i, pos2[i] - 1])) for i in range(len(doubles))]
    wt1 = [wt_nt_seq[pos1[i] - 1] for i in range(len(doubles))]
    wt2 = [wt_nt_seq[pos2[i] - 1] for i in range(len(doubles))]

    doubles = doubles.with_columns([
        pl.Series("Pos1", pos1, dtype=pl.Int32),
        pl.Series("Pos2", pos2, dtype=pl.Int32),
        pl.Series("Mut1", mut1, dtype=pl.Utf8),
        pl.Series("Mut2", mut2, dtype=pl.Utf8),
        pl.Series("WT_AA1", wt1, dtype=pl.Utf8),
        pl.Series("WT_AA2", wt2, dtype=pl.Utf8),
    ])

    doubles = _join_singles_mean_count(doubles, singles_df)
    return doubles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join_singles_mean_count(
    doubles: pl.DataFrame,
    singles_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join mean_count from singles for Pos1/Mut1 and Pos2/Mut2."""
    if singles_df is None or len(singles_df) == 0:
        doubles = doubles.with_columns([
            pl.lit(None).cast(pl.Float64).alias("s1_mean_count"),
            pl.lit(None).cast(pl.Float64).alias("s2_mean_count"),
        ])
        return doubles

    if "mean_count" not in singles_df.columns:
        doubles = doubles.with_columns([
            pl.lit(None).cast(pl.Float64).alias("s1_mean_count"),
            pl.lit(None).cast(pl.Float64).alias("s2_mean_count"),
        ])
        return doubles

    s_lookup = singles_df.select(["Pos", "Mut", "mean_count"])

    # Join for position 1
    doubles = doubles.join(
        s_lookup.rename({"Pos": "Pos1", "Mut": "Mut1", "mean_count": "s1_mean_count"}),
        on=["Pos1", "Mut1"],
        how="left",
    )
    # Join for position 2
    doubles = doubles.join(
        s_lookup.rename({"Pos": "Pos2", "Mut": "Mut2", "mean_count": "s2_mean_count"}),
        on=["Pos2", "Mut2"],
        how="left",
    )
    return doubles


def _drop_uncorr_suffix(df: pl.DataFrame) -> pl.DataFrame:
    """Rename fitness{E}_uncorr / sigma{E}_uncorr → fitness{E} / sigma{E}."""
    rename_map = {}
    for col in df.columns:
        if col.endswith("_uncorr"):
            new_name = col[:-7]  # remove "_uncorr"
            rename_map[col] = new_name
    if rename_map:
        df = df.rename(rename_map)
    return df
