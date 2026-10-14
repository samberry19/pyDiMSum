"""Stage 4 (STEAM) — process, filter, and annotate variants.

Processes the wide variant count table to:
  - Detect and handle indel variants
  - Remove internal constant regions
  - Calculate Nham_nt, Nham_aa, Nmut_codons
  - Translate nt → aa sequences
  - Assign each variant to its nearest WT (when several WTs are given)
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
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from pydimsum.config import RunConfig, WTEntry
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

    wt_refs: dict[str, tuple[str, str]] = field(default_factory=dict)
    """WT name -> (variable-position nt sequence lower-case, aa sequence ('' if
    noncoding)).  Used downstream for mutation calling per WT."""


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
    wildtypes = config.wildtypes             # primary WT first
    multi = len(wildtypes) > 1
    seq_type = config.sequence_type_resolved

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
    # 2. Barcode identity debarcoding (optional)                          #
    # ------------------------------------------------------------------ #
    if config.barcode_identity_path is not None:
        df, nobarcode_df = _debarcode_variants(df, config, count_cols)
    else:
        nobarcode_df = pl.DataFrame({"nt_seq": pl.Series([], dtype=pl.Utf8)})

    # ------------------------------------------------------------------ #
    # 3. Mark indels and split                                            #
    # ------------------------------------------------------------------ #
    wt_lengths = sorted({len(wt.seq) for wt in wildtypes})
    df = df.with_columns(
        (~pl.col("nt_seq").str.len_chars().is_in(wt_lengths)).alias("indel")
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

    subst_all = df.filter(~df["indel"])
    indel_retained_df = df.filter(df["indel"])

    # ------------------------------------------------------------------ #
    # 3b. Assign each substitution variant to its nearest WT              #
    # ------------------------------------------------------------------ #
    rejected_parts: list[pl.DataFrame] = []
    if multi:
        subst_all, ambiguous_df = _assign_wildtype(subst_all, wildtypes)
        if len(ambiguous_df):
            logger.warning(
                "%d variants are equidistant from two or more WTs and were rejected",
                len(ambiguous_df),
            )
            rejected_parts.append(ambiguous_df)
        for wt in wildtypes:
            n = int((subst_all["WT_id"] == wt.name).sum()) if len(subst_all) else 0
            logger.info("WT %s: %d variants assigned", wt.name, n)

    # ------------------------------------------------------------------ #
    # 4-8. Per-WT processing                                              #
    # ------------------------------------------------------------------ #
    group_dfs: list[pl.DataFrame] = []
    wt_refs: dict[str, tuple[str, str]] = {}
    for i, wt in enumerate(wildtypes):
        is_primary = i == 0
        if multi:
            group = subst_all.filter(pl.col("WT_id") == wt.name)
        else:
            group = subst_all
        wt_var = wt.variable_seq.lower()
        wt_refs[wt.name] = (
            wt_var,
            translate_sequences_fast([wt_var])[0] if seq_type == "coding" else "",
        )
        if len(group) == 0:
            if is_primary:
                raise RuntimeError(
                    "No variants assigned to the primary WT. "
                    "Check wildtypeSequence is correct."
                )
            logger.warning("No variants assigned to WT %s; skipping", wt.name)
            continue
        group, group_rejected = _process_wt_group(
            group, wt, config, count_cols, is_primary, multi
        )
        rejected_parts.extend(group_rejected)
        if len(group):
            group_dfs.append(group)

    subst_df = pl.concat(group_dfs, how="diagonal_relaxed")

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
        nobarcode_df=nobarcode_df,
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
        wt_refs=wt_refs,
    )


# ---------------------------------------------------------------------------
# Per-WT processing (steps 4-8)
# ---------------------------------------------------------------------------

def _process_wt_group(
    subst_df: pl.DataFrame,
    wt: WTEntry,
    config: RunConfig,
    count_cols: list[str],
    is_primary: bool,
    multi: bool,
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Annotate and filter the substitution variants assigned to one WT.

    Raises for the primary WT if nothing remains after a filter; a secondary
    WT that is exhausted returns an empty frame instead.
    """
    wt_seq = wt.seq                          # full sequence, lower-case
    wt_coded = wt.coded_seq                  # case-coded (upper=variable)
    has_const = wt.has_constant_region
    seq_type = config.sequence_type_resolved
    permitted_seqs = wt.permitted            # IUPAC, variable pos only
    wt_len = len(wt_seq)
    rejected_parts: list[pl.DataFrame] = []

    def _exhausted(msg: str) -> None:
        if is_primary:
            raise RuntimeError(msg)
        logger.warning("WT %s: %s", wt.name, msg)

    if multi:
        subst_df = subst_df.with_columns(pl.lit(wt.name).alias("WT_id"))

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
        logger.warning("WT sequence %s not found in variant table.", wt.name)

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
            # Recompute WT aa from variable positions before translating variants
            wt_var_aa = translate_sequences_fast([decode_row(wt_mat_var)])[0]
            wt_aa_arr = np.frombuffer(wt_var_aa.encode("ascii"), dtype=np.uint8)
            wt_has_term_stop = wt_var_aa.endswith("*")
            # Retranslate
            new_aa_seqs = translate_sequences_fast(new_nt_seqs)
            new_aa_mat = encode(new_aa_seqs) if new_aa_seqs else np.empty((0, len(wt_aa_arr)), dtype=np.uint8)
            new_nham_nt = hamming(mat_var, wt_mat_var)
            new_nham_aa = hamming(new_aa_mat, wt_aa_arr) if len(new_aa_seqs) else np.empty(0, np.int32)
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
        _exhausted("No variants remaining after permitted-mutations filtering.")
        return subst_df, rejected_parts

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
        _exhausted("No variants remaining after too-many-substitutions filtering.")
        return subst_df, rejected_parts

    # ------------------------------------------------------------------ #
    # 8. Nmut_codons + mixed-substitutions filter                         #
    # ------------------------------------------------------------------ #
    if is_primary:
        _check_variants(subst_df, count_cols)

    if len(subst_df):
        nt_seqs_cur = subst_df["nt_seq"].to_list()
        mat_cur = encode(nt_seqs_cur)
        wt_row = np.frombuffer(wt.variable_seq.lower().encode("ascii"), dtype=np.uint8)

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
        _exhausted("No variants remaining after mixed-substitutions filtering.")
        return subst_df, rejected_parts

    if multi:
        subst_df = subst_df.with_columns(pl.col("WT").alias("WT_ref"))
        if not is_primary:
            subst_df = subst_df.with_columns(pl.lit(False).alias("WT"))
    return subst_df, rejected_parts


# ---------------------------------------------------------------------------
# WT assignment (multiple WTs)
# ---------------------------------------------------------------------------

def _assign_wildtype(
    df: pl.DataFrame,
    wildtypes: list[WTEntry],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assign each variant to the nearest WT (by nt Hamming distance).

    Only WTs of the same length as the variant are candidates.  Variants that
    are equidistant from two or more candidate WTs are returned separately
    (flagged ``ambiguous_wt``) and are not assigned.

    Returns (assigned_df with ``WT_id`` column, ambiguous_df).
    """
    by_len: dict[int, list[WTEntry]] = {}
    for wt in wildtypes:
        by_len.setdefault(len(wt.seq), []).append(wt)

    lengths = df["nt_seq"].str.len_chars()
    assigned: list[pl.DataFrame] = []
    ambiguous: list[pl.DataFrame] = []
    for length, wts in by_len.items():
        sub = df.filter(lengths == length)
        if len(sub) == 0:
            continue
        if len(wts) == 1:
            assigned.append(sub.with_columns(pl.lit(wts[0].name).alias("WT_id")))
            continue
        mat = encode(sub["nt_seq"].to_list())
        dists = np.stack([hamming(mat, wt.seq) for wt in wts], axis=1)
        dmin = dists.min(axis=1)
        tie = (dists == dmin[:, None]).sum(axis=1) > 1
        names = np.array([wt.name for wt in wts])[dists.argmin(axis=1)]
        sub = sub.with_columns(pl.Series("WT_id", names.tolist(), dtype=pl.Utf8))
        tie_mask = pl.Series(tie.tolist(), dtype=pl.Boolean)
        assigned.append(sub.filter(~tie_mask))
        ambiguous.append(
            sub.filter(tie_mask).with_columns(pl.lit(True).alias("ambiguous_wt"))
        )

    if assigned:
        assigned_df = pl.concat(assigned, how="diagonal_relaxed")
    else:
        assigned_df = df.head(0).with_columns(pl.lit(None).cast(pl.Utf8).alias("WT_id"))
    if ambiguous:
        ambiguous_df = pl.concat(ambiguous, how="diagonal_relaxed")
    else:
        ambiguous_df = df.head(0)
    return assigned_df, ambiguous_df


# ---------------------------------------------------------------------------
# WT presence guard (mirrors R/dimsum__check_variants.R)
# ---------------------------------------------------------------------------

def _check_variants(subst_df: pl.DataFrame, count_cols: list[str]) -> None:
    """Raise RuntimeError if WT is absent or fewer than 2 substitution variants remain."""
    wt_present = subst_df["WT"].fill_null(False).any()
    if not wt_present:
        # Show top-5 candidates by mean count to aid debugging
        if count_cols:
            mean_count = subst_df.select(
                [pl.col(c).fill_null(0).cast(pl.Float64) for c in count_cols if c in subst_df.columns]
            ).mean_horizontal()
            top5 = (
                subst_df
                .with_columns(mean_count.alias("_mean_count"))
                .sort("_mean_count", descending=True)
                .head(5)["nt_seq"]
                .to_list()
            )
            candidates = ", ".join(top5)
        else:
            candidates = "(no count columns)"
        raise RuntimeError(
            f"WT variant not found in variant table after filtering. "
            f"Check wildtypeSequence is correct. "
            f"Top 5 sequences by mean count: {candidates}"
        )
    n_non_wt = subst_df.filter(~pl.col("WT").fill_null(False)).height
    if n_non_wt < 2:
        raise RuntimeError(
            f"Fewer than 2 non-WT variants remain after filtering (found {n_non_wt}). "
            "Check filters are not too strict."
        )


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _compute_stats(
    retained_df: pl.DataFrame,
    rejected_df: pl.DataFrame,
    indel_df: pl.DataFrame,
    nobarcode_df: pl.DataFrame,
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

    def _filtered_sums(df: pl.DataFrame, col: str, val) -> dict[str, int]:
        if not isinstance(df, pl.DataFrame) or len(df) == 0 or col not in df.columns:
            return {c: 0 for c in count_cols}
        return _col_sums(df.filter(pl.col(col) == val))

    stats["nuc_indel_dict"] = _col_sums(indel_df)
    stats["nuc_nbarc_dict"] = _col_sums(nobarcode_df)
    stats["nuc_const_dict"] = _filtered_sums(rejected_df, "constant_region", False)
    stats["nuc_frbdn_dict"] = _filtered_sums(rejected_df, "permitted", False)
    stats["nuc_tmsub_dict"] = _filtered_sums(rejected_df, "too_many_substitutions", True)
    stats["nuc_mxsub_dict"] = _filtered_sums(rejected_df, "mixed_substitutions", True)
    stats["nuc_ambig_dict"] = _filtered_sums(rejected_df, "ambiguous_wt", True)

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


# ---------------------------------------------------------------------------
# Barcode debarcoding helper
# ---------------------------------------------------------------------------

def _debarcode_variants(
    df: pl.DataFrame,
    config: "RunConfig",
    count_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replace barcode sequences with their mapped variant sequences.

    Mirrors: R/dimsum__debarcode_variants.R

    Parameters
    ----------
    df:
        Wide count table with ``nt_seq`` column (barcode sequences).
    config:
        RunConfig with ``barcode_identity_path`` set.
    count_cols:
        List of count column names (``count_e*``).

    Returns
    -------
    (debarcoded_df, nobarcode_df):
        debarcoded_df: rows with valid barcodes, ``nt_seq`` replaced with
            the mapped variant, counts aggregated per unique variant.
        nobarcode_df: rows with invalid (unknown) barcodes.
    """
    from pydimsum.io.designs import load_barcode_identity

    barcode_map = load_barcode_identity(config.barcode_identity_path)
    logger.info("Loaded %d barcode→variant mappings", len(barcode_map))

    # Flag valid vs. invalid barcodes
    seqs = df["nt_seq"].to_list()
    valid_flags = [s in barcode_map for s in seqs]
    df = df.with_columns(
        pl.Series("barcode_valid", valid_flags, dtype=pl.Boolean)
    )

    n_valid = sum(valid_flags)
    n_invalid = len(valid_flags) - n_valid
    logger.info(
        "Barcode debarcoding: %d valid, %d invalid (unknown barcode)",
        n_valid, n_invalid,
    )

    if n_valid == 0:
        raise RuntimeError(
            "Cannot proceed with variant processing: "
            "No valid barcodes found. Check barcodeIdentityPath."
        )

    nobarcode_df = df.filter(~pl.col("barcode_valid")).select(
        ["nt_seq"] + count_cols
    )

    valid_df = df.filter(pl.col("barcode_valid"))

    # Replace nt_seq with mapped variant
    new_seqs = [barcode_map[s] for s in valid_df["nt_seq"].to_list()]
    valid_df = valid_df.with_columns(
        pl.Series("nt_seq", new_seqs, dtype=pl.Utf8)
    )

    # Aggregate counts for identical variants
    agg_exprs = [
        pl.col(c).fill_null(0).sum().alias(c) for c in count_cols
    ]
    debarcoded_df = (
        valid_df.group_by("nt_seq")
        .agg(agg_exprs)
    )

    logger.info(
        "After debarcoding: %d unique variants (from %d valid-barcode rows)",
        len(debarcoded_df), n_valid,
    )

    return debarcoded_df, nobarcode_df
