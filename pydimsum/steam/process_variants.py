"""Stage 4 (STEAM) — process, filter, and annotate variants.

Processes the wide variant count table to:
  - Detect and handle indel variants
  - Remove internal constant regions
  - Calculate Nham_nt, Nham_aa, Nmut_codons
  - Translate nt → aa sequences
  - Identify WT sequence and flag STOP codons
  - Filter out forbidden, too-many-substitution, and mixed-substitution variants
  - Compute the ``merge_seq`` column

Returns separate DataFrames for:
  retained variants, indel variants, rejected variants, no-barcode variants.

Mirrors:
  R/dimsum__process_merged_variants.R
  R/dimsum__remove_internal_constant_region.R
  R/dimsum__identify_STOP_mutations.R
  R/dimsum__identify_permitted_mutations.R
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from pydimsum.config import RunConfig
from pydimsum.steam.sequences import (
    constant_region_matches_wt,
    decode_row,
    detect_stop,
    encode,
    hamming,
    n_mut_codons,
    permitted_mask,
    strip_constant_regions,
    translate_sequences_fast,
    variable_position_mask,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProcessedVariants:
    retained: pl.DataFrame
    """Variants passing all filters.  Contains nt_seq, aa_seq, WT, STOP,
    STOP_readthrough, Nham_nt, Nham_aa, Nmut_codons, indel, merge_seq,
    error_model, and all count_e* columns."""

    indel: pl.DataFrame
    """Indel variants that were discarded (length != WT and not retained)."""

    rejected: pl.DataFrame
    """Non-indel variants removed by constant-region / permitted / too-many-subs
    / mixed-subs filters."""

    # Mutation count statistics (dict of sample→count)
    stats: dict[str, dict]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_variants(
    df: pl.DataFrame,
    config: RunConfig,
) -> ProcessedVariants:
    """Filter and annotate variants, returning separated DataFrames.

    Parameters
    ----------
    df:
        Wide variant table from merge.py: columns ``nt_seq`` + count_e* columns.
    config:
        Validated pipeline configuration.
    """
    count_cols = [c for c in df.columns if c.startswith("count_e")]
    wt_seq = config.wt_nt_seq                # full sequence, lower-case
    wt_coded = config.wildtype_sequence      # case-coded (upper=variable)
    has_const = config.has_constant_region
    seq_type = config.sequence_type_resolved
    permitted_seqs = config.permitted_sequences  # IUPAC, variable pos only

    # ------------------------------------------------------------------ #
    # 1. Reverse complement (optional)                                    #
    # ------------------------------------------------------------------ #
    if config.reverse_complement:
        from Bio.Seq import Seq
        df = df.with_columns(
            pl.col("nt_seq").map_elements(
                lambda s: str(Seq(s).reverse_complement()),
                return_dtype=pl.Utf8,
            )
        )

    # ------------------------------------------------------------------ #
    # 2. No-barcode filtering (barcode_identity_path support)             #
    # In STEAM-only mode there are no barcodes to filter.                 #
    # ------------------------------------------------------------------ #
    nobarcode_df = pl.DataFrame({"nt_seq": pl.Series([], dtype=pl.Utf8)})

    # ------------------------------------------------------------------ #
    # 3. Mark indels and split                                            #
    # ------------------------------------------------------------------ #
    wt_len = len(wt_seq)
    df = df.with_columns(
        (pl.col("nt_seq").str.len_chars() != wt_len).alias("indel")
    )

    indel_lengths_cfg = config._indel_lengths  # None=none, []=all, [int...]=specific

    if indel_lengths_cfg is None:
        # Discard all indels
        indel_disc_mask = df["indel"]
    elif len(indel_lengths_cfg) == 0:
        # Retain all indels
        indel_disc_mask = pl.Series([False] * len(df))
    else:
        # Retain specified lengths
        indel_disc_mask = df["indel"] & ~df["nt_seq"].str.len_chars().is_in(
            indel_lengths_cfg
        )

    indel_df = df.filter(indel_disc_mask).select(
        ["nt_seq"] + count_cols + ["indel"]
    )
    # Remaining: substitution variants + retained indels
    df = df.filter(~indel_disc_mask)

    if len(df) == 0:
        raise RuntimeError(
            "No variants remaining after indel filtering. "
            "Check wildtypeSequence and indels parameters."
        )

    # ------------------------------------------------------------------ #
    # 4. Vectorized nt operations on non-indel variants                   #
    # ------------------------------------------------------------------ #
    subst_mask = ~df["indel"]
    subst_df = df.filter(subst_mask)
    indel_retained_df = df.filter(~subst_mask)

    nt_seqs = subst_df["nt_seq"].to_list()
    if nt_seqs:
        mat = encode(nt_seqs)                        # (n, L) uint8, lower-case
        wt_arr = np.frombuffer(wt_seq.encode("ascii"), dtype=np.uint8)
        nham_nt = hamming(mat, wt_arr)               # (n,) int32
    else:
        mat = np.empty((0, wt_len), dtype=np.uint8)
        nham_nt = np.empty(0, dtype=np.int32)

    # ---- Translate nt → aa ----
    aa_seqs = translate_sequences_fast(nt_seqs)

    # ---- WT aa sequence ----
    wt_aa = translate_sequences_fast([wt_seq])[0]
    wt_aa_arr = np.frombuffer(wt_aa.encode("ascii"), dtype=np.uint8)

    aa_mat = encode(aa_seqs)                         # (n, n_aa) uint8
    nham_aa = hamming(aa_mat, wt_aa_arr) if len(aa_seqs) else np.empty(0, np.int32)

    # ---- Identify WT ----
    wt_flags = np.array([s == wt_seq for s in nt_seqs], dtype=bool)

    if not wt_flags.any():
        logger.warning("WT sequence not found in variant table.")

    # ---- STOP codon detection ----
    wt_has_term_stop = wt_aa.endswith("*")
    if len(aa_seqs):
        stop_flags, readthrough_flags = detect_stop(aa_mat, wt_has_term_stop)
    else:
        stop_flags = np.empty(0, dtype=bool)
        readthrough_flags = np.empty(0, dtype=bool)

    # ---- Assemble subst_df with annotations ----
    subst_df = subst_df.with_columns([
        pl.Series("aa_seq", aa_seqs, dtype=pl.Utf8),
        pl.Series("Nham_nt", nham_nt, dtype=pl.Int32),
        pl.Series("Nham_aa", nham_aa, dtype=pl.Int32),
        pl.Series("WT", wt_flags, dtype=pl.Boolean),
        pl.Series("STOP", stop_flags.tolist(), dtype=pl.Boolean),
        pl.Series("STOP_readthrough", readthrough_flags.tolist(), dtype=pl.Boolean),
    ])

    # ------------------------------------------------------------------ #
    # 5. Constant-region filtering                                        #
    # ------------------------------------------------------------------ #
    rejected_parts: list[pl.DataFrame] = []

    if has_const:
        # Mask: True = WT constant region (sequence passes)
        const_ok = constant_region_matches_wt(mat, wt_coded)
        const_rejected_df = subst_df.filter(~pl.Series(const_ok.tolist()))
        const_rejected_df = const_rejected_df.with_columns(
            pl.lit(False).alias("constant_region")
        )
        rejected_parts.append(const_rejected_df)

        subst_df = subst_df.filter(pl.Series(const_ok.tolist()))
        # Recompute nt_seq (variable positions only) after stripping const
        vmask = variable_position_mask(wt_coded)
        if len(subst_df):
            # Re-encode filtered subset
            nt_seqs_ok = subst_df["nt_seq"].to_list()
            mat_ok = encode(nt_seqs_ok)
            mat_var = strip_constant_regions(mat_ok, vmask)
            wt_mat_var = strip_constant_regions(
                np.frombuffer(wt_seq.encode("ascii"), dtype=np.uint8).reshape(1, -1),
                vmask,
            )[0]
            # New nt_seq (variable only)
            new_nt_seqs = [decode_row(mat_var[i]) for i in range(len(mat_var))]
            # Retranslate
            new_aa_seqs = translate_sequences_fast(new_nt_seqs)
            new_aa_mat = encode(new_aa_seqs) if new_aa_seqs else np.empty((0, len(wt_aa_arr)), dtype=np.uint8)
            new_nham_nt = hamming(mat_var, wt_mat_var)
            new_nham_aa = hamming(new_aa_mat, wt_aa_arr) if len(new_aa_seqs) else np.empty(0, np.int32)
            # Update wt_has_term_stop using stripped WT
            wt_var_aa = translate_sequences_fast([decode_row(wt_mat_var)])[0]
            wt_has_term_stop = wt_var_aa.endswith("*")
            new_stop, new_rt = detect_stop(new_aa_mat, wt_has_term_stop) if len(new_aa_seqs) else (
                np.empty(0, dtype=bool), np.empty(0, dtype=bool)
            )
            new_wt = np.array([s == decode_row(wt_mat_var) for s in new_nt_seqs], dtype=bool)

            subst_df = subst_df.with_columns([
                pl.Series("nt_seq", new_nt_seqs, dtype=pl.Utf8),
                pl.Series("aa_seq", new_aa_seqs, dtype=pl.Utf8),
                pl.Series("Nham_nt", new_nham_nt, dtype=pl.Int32),
                pl.Series("Nham_aa", new_nham_aa, dtype=pl.Int32),
                pl.Series("WT", new_wt, dtype=pl.Boolean),
                pl.Series("STOP", new_stop.tolist(), dtype=pl.Boolean),
                pl.Series("STOP_readthrough", new_rt.tolist(), dtype=pl.Boolean),
            ])
            # Update mat and wt for subsequent steps
            mat = mat_var
            wt_arr = wt_mat_var
            wt_aa_arr = np.frombuffer(wt_var_aa.encode("ascii"), dtype=np.uint8)
    else:
        subst_df = subst_df.with_columns(pl.lit(True).alias("constant_region"))

    # ------------------------------------------------------------------ #
    # 6. Permitted mutation filtering                                     #
    # ------------------------------------------------------------------ #
    if len(subst_df):
        nt_seqs_cur = subst_df["nt_seq"].to_list()
        mat_cur = encode(nt_seqs_cur)
        perm_ok = permitted_mask(mat_cur, permitted_seqs, wt_coded)
        perm_rejected = subst_df.filter(~pl.Series(perm_ok.tolist()))
        perm_rejected = perm_rejected.with_columns(pl.lit(False).alias("permitted"))
        rejected_parts.append(perm_rejected)
        subst_df = subst_df.filter(pl.Series(perm_ok.tolist()))
        mat_cur = mat_cur[perm_ok]

    if len(subst_df) == 0:
        raise RuntimeError(
            "No variants remaining after permitted-mutations filtering."
        )

    # ------------------------------------------------------------------ #
    # 7. Too-many-substitutions filter                                    #
    # ------------------------------------------------------------------ #
    max_subs = config.max_substitutions
    if len(subst_df):
        if seq_type == "coding":
            too_many = subst_df["Nham_aa"].fill_null(0) > max_subs
        else:
            too_many = subst_df["Nham_nt"].fill_null(0) > max_subs
        tmsub_rejected = subst_df.filter(too_many).with_columns(
            pl.lit(True).alias("too_many_substitutions")
        )
        rejected_parts.append(tmsub_rejected)
        subst_df = subst_df.filter(~too_many)
        if len(subst_df):
            mat_cur = encode(subst_df["nt_seq"].to_list())

    if len(subst_df) == 0:
        raise RuntimeError(
            "No variants remaining after too-many-substitutions filtering."
        )

    # ------------------------------------------------------------------ #
    # 8. Nmut_codons + mixed-substitutions filter                         #
    # ------------------------------------------------------------------ #
    if len(subst_df):
        nt_seqs_cur = subst_df["nt_seq"].to_list()
        mat_cur = encode(nt_seqs_cur)
        wt_row = np.frombuffer(
            subst_df.filter(pl.col("WT"))["nt_seq"].to_list()[0].encode("ascii"),
            dtype=np.uint8,
        )

        if seq_type == "coding" and len(wt_row) % 3 == 0:
            nmut_c = n_mut_codons(mat_cur, wt_row)
        else:
            nmut_c = np.zeros(len(subst_df), dtype=np.int32)

        subst_df = subst_df.with_columns(
            pl.Series("Nmut_codons", nmut_c, dtype=pl.Int32)
        )

        # Mixed substitutions: codons_with_mutations != Nham_aa AND Nham_aa != 0
        if seq_type == "coding" and not config.mixed_substitutions:
            nham_aa_arr = np.array(subst_df["Nham_aa"].fill_null(0).to_list(), dtype=np.int32)
            mixed = (nmut_c - nham_aa_arr != 0) & (nham_aa_arr != 0)
            mixed_rejected = subst_df.filter(pl.Series(mixed.tolist())).with_columns(
                pl.lit(True).alias("mixed_substitutions")
            )
            rejected_parts.append(mixed_rejected)
            subst_df = subst_df.filter(~pl.Series(mixed.tolist()))
        else:
            subst_df = subst_df.with_columns(
                pl.lit(False).alias("mixed_substitutions")
            )
    else:
        subst_df = subst_df.with_columns(
            pl.lit(pl.Series([], dtype=pl.Int32)).alias("Nmut_codons")
        )

    if len(subst_df) == 0:
        raise RuntimeError(
            "No variants remaining after mixed-substitutions filtering."
        )

    # ------------------------------------------------------------------ #
    # 9. Set merge_seq and error_model columns                            #
    # ------------------------------------------------------------------ #
    subst_df = subst_df.with_columns([
        pl.col("nt_seq").alias("merge_seq"),
        pl.lit(True).alias("error_model"),
    ])

    # Combine substitution variants with retained indels
    # Retained indels get NaN for annotation columns
    if len(indel_retained_df) > 0:
        indel_retained_df = indel_retained_df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("aa_seq"),
            pl.lit(None).cast(pl.Int32).alias("Nham_nt"),
            pl.lit(None).cast(pl.Int32).alias("Nham_aa"),
            pl.lit(None).cast(pl.Int32).alias("Nmut_codons"),
            pl.lit(None).cast(pl.Boolean).alias("WT"),
            pl.lit(False).alias("STOP"),
            pl.lit(False).alias("STOP_readthrough"),
            pl.lit(False).alias("mixed_substitutions"),
            pl.lit(True).alias("constant_region"),
            pl.lit(True).alias("permitted"),
            pl.lit(False).alias("too_many_substitutions"),
            pl.col("nt_seq").alias("merge_seq"),
            pl.lit(True).alias("error_model"),
        ])
        retained_df = pl.concat([subst_df, indel_retained_df], how="diagonal")
    else:
        retained_df = subst_df

    # ------------------------------------------------------------------ #
    # 10. Build rejected DataFrame                                        #
    # ------------------------------------------------------------------ #
    if rejected_parts:
        rejected_df = pl.concat(rejected_parts, how="diagonal_relaxed")
    else:
        rejected_df = pl.DataFrame()

    # ------------------------------------------------------------------ #
    # 11. Mutation statistics                                             #
    # ------------------------------------------------------------------ #
    stats = _compute_stats(
        retained_df=retained_df,
        rejected_df=rejected_df,
        indel_df=indel_df,
        count_cols=count_cols,
        seq_type=seq_type,
    )

    logger.info(
        "Processed variants: %d retained, %d indels discarded, %d rejected",
        len(retained_df), len(indel_df), len(rejected_df) if isinstance(rejected_df, pl.DataFrame) else 0,
    )

    # Reorder output columns for clarity
    base_cols = [
        "nt_seq", "aa_seq", "WT", "STOP", "STOP_readthrough",
        "Nham_nt", "Nham_aa", "Nmut_codons", "indel",
        "merge_seq", "error_model",
    ]
    present = [c for c in base_cols if c in retained_df.columns]
    retained_df = retained_df.select(
        present + [c for c in retained_df.columns if c not in present]
    )

    return ProcessedVariants(
        retained=retained_df,
        indel=indel_df,
        rejected=rejected_df,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _compute_stats(
    retained_df: pl.DataFrame,
    rejected_df: pl.DataFrame,
    indel_df: pl.DataFrame,
    count_cols: list[str],
    seq_type: str,
) -> dict:
    """Compute mutation count statistics per sample for the report."""
    stats: dict[str, dict] = {}

    def _col_sums(df: pl.DataFrame) -> dict[str, int]:
        if len(df) == 0:
            return {c: 0 for c in count_cols}
        return {
            c: int(df[c].fill_null(0).sum())
            for c in count_cols
            if c in df.columns
        }

    stats["nuc_indel_dict"] = _col_sums(indel_df)
    stats["nuc_const_dict"] = _col_sums(
        rejected_df.filter(pl.col("constant_region") == False) if "constant_region" in rejected_df.columns else pl.DataFrame()
    )

    # Substitution distributions in retained variants
    if seq_type == "coding" and "Nham_aa" in retained_df.columns:
        for c in count_cols:
            if c in retained_df.columns:
                grp = (
                    retained_df
                    .with_columns(pl.col(c).fill_null(0))
                    .group_by("Nham_aa")
                    .agg(pl.col(c).sum().alias("sum_count"))
                    .sort("Nham_aa")
                )
                stats.setdefault("aa_subst_dict", {})[c] = dict(
                    zip(grp["Nham_aa"].to_list(), grp["sum_count"].to_list())
                )

    if "Nham_nt" in retained_df.columns:
        for c in count_cols:
            if c in retained_df.columns:
                grp = (
                    retained_df
                    .with_columns(pl.col(c).fill_null(0))
                    .group_by("Nham_nt")
                    .agg(pl.col(c).sum().alias("sum_count"))
                    .sort("Nham_nt")
                )
                stats.setdefault("nuc_subst_dict", {})[c] = dict(
                    zip(grp["Nham_nt"].to_list(), grp["sum_count"].to_list())
                )

    return stats
