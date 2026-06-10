"""AA-level count aggregation.

Mirrors:
  R/dimsum__aggregate_AA_variants.R
  R/dimsum__aggregate_AA_variants_fitness.R
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def aggregate_aa_variants(
    df: pl.DataFrame,
    replicates: list[int],
    synonym_sequences: list[str] | None = None,
) -> pl.DataFrame:
    """Aggregate variant counts to the amino acid level.

    Called when ``sequence_type == 'coding'`` and ``mixed_substitutions == True``.

    Mirrors: R/dimsum__aggregate_AA_variants.R

    Sets ``merge_seq`` to ``aa_seq``.
    """
    reps = replicates
    all_count_cols = (
        [f"count_e{E}_s0" for E in reps] +
        [f"count_e{E}_s1" for E in reps]
    )
    count_cols = [c for c in all_count_cols if c in df.columns]

    df = df.with_columns(pl.col("aa_seq").alias("merge_seq"))

    # Reference AA sequences for synonymous variants
    wt_aa = df.filter(pl.col("WT"))["aa_seq"].to_list()
    if wt_aa:
        synseq_refs = set(wt_aa)
    else:
        synseq_refs = set()
    if synonym_sequences:
        # Translate synonym sequences if they are nt sequences
        from pydimsum.steam.sequences import translate_sequences_fast
        for s in synonym_sequences:
            aa = translate_sequences_fast([s.lower()])
            synseq_refs.add(aa[0])

    # Extract and mark synonymous variants (WT-like at AA level)
    syn_mask = df["aa_seq"].is_in(list(synseq_refs))
    syn_df = df.filter(syn_mask).with_columns([
        pl.col("nt_seq").alias("merge_seq"),
        pl.lit(False).alias("error_model"),
    ])

    # For non-synonymous: set merge_seq = aa_seq, sum counts per AA group
    for c in count_cols:
        df = df.with_columns(
            pl.col(c)
            .fill_null(0)
            .sum()
            .over("merge_seq")
            .alias(f"{c}_agg")
        )

    # Retain one row per unique merge_seq (AA sequence)
    agg_cols = [f"{c}_agg" for c in count_cols]
    meta_cols = [
        "merge_seq", "nt_seq", "aa_seq", "Nham_nt", "Nham_aa",
        "Nmut_codons", "WT", "indel", "STOP", "STOP_readthrough", "error_model",
    ]
    present_meta = [c for c in meta_cols if c in df.columns]

    output_df = (
        df.unique(subset=["merge_seq"])
        .select(present_meta + agg_cols)
        .rename({f"{c}_agg": c for c in count_cols})
    )

    # AA-level WT: Nham_aa == 0
    output_df = output_df.with_columns(
        pl.when(pl.col("Nham_aa") == 0).then(True).otherwise(pl.col("WT")).alias("WT")
    )
    # Nham_nt and Nmut_codons are meaningless after AA aggregation
    output_df = output_df.with_columns([
        pl.lit(None).cast(pl.Int32).alias("Nham_nt"),
        pl.lit(None).cast(pl.Int32).alias("Nmut_codons"),
        pl.lit(None).cast(pl.Utf8).alias("nt_seq"),
    ])

    # Combine with synonymous variants
    output_df = pl.concat([syn_df, output_df], how="diagonal_relaxed")
    return output_df


def aggregate_aa_variants_fitness(
    df: pl.DataFrame,
    replicates: list[int],
) -> pl.DataFrame:
    """Aggregate fitness/sigma at the AA level (mixedSubstitutions=False).

    Multiple NT variants that encode the same AA change are combined via
    inverse-variance weighting for fitness/sigma, and by summation for counts.
    Synonymous variants (Nham_aa==0) keep their NT identity and are excluded
    from the error model (error_model=False).

    Mirrors: R/dimsum__aggregate_AA_variants_fitness.R
    """
    reps = replicates
    input_cols = [f"count_e{E}_s0" for E in reps if f"count_e{E}_s0" in df.columns]
    output_cols = [f"count_e{E}_s1" for E in reps if f"count_e{E}_s1" in df.columns]
    all_count_cols = input_cols + output_cols

    # Fitness and sigma column names (per replicate, _uncorr suffix)
    fitness_cols = [c for c in df.columns if c.startswith("fitness") and c.endswith("_uncorr")]
    sigma_cols   = [c.replace("fitness", "sigma") for c in fitness_cols]
    # Only keep sigma cols that actually exist
    sigma_cols = [c for c in sigma_cols if c in df.columns]

    # ---- Set merge_seq ----
    # Nonsynonymous error_model variants: merge_seq = aa_seq
    # WT/synonymous (Nham_aa==0) or indel: keep nt_seq
    df = df.with_columns(
        pl.when(
            pl.col("error_model").fill_null(False) &
            (pl.col("Nham_aa").fill_null(0) > 0) &
            ~pl.col("indel").fill_null(False)
        )
        .then(pl.col("aa_seq"))
        .otherwise(pl.col("nt_seq"))
        .alias("merge_seq")
    )

    # ---- Synonymous variants (Nham_aa == 0, not indel) ----
    # Mirrors R: syn_dt keeps original rows with error_model=F (for synonymous.txt),
    # AND the same rows are kept in output_dt with original error_model=T (for wildtype.txt).
    # So we need two copies: syn_df (error_model=F) and syn_truth_df (original error_model).
    syn_mask = (
        (pl.col("Nham_aa").fill_null(0) == 0) &
        ~pl.col("indel").fill_null(False)
    )
    syn_df = df.filter(syn_mask).with_columns(pl.lit(False).alias("error_model"))
    # Keep original (error_model=True) version of synonymous/WT rows for all_variants/wildtype output
    syn_truth_df = df.filter(syn_mask)

    # ---- Nonsynonymous + indel variants only ----
    nonsyn_df = df.filter(~syn_mask)

    if len(nonsyn_df) == 0:
        return pl.concat([syn_df, syn_truth_df], how="diagonal_relaxed")

    # ---- Aggregate counts: sum per merge_seq ----
    for c in all_count_cols:
        if c in nonsyn_df.columns:
            nonsyn_df = nonsyn_df.with_columns(
                pl.col(c).fill_null(0).sum().over("merge_seq").alias(f"{c}_agg")
            )

    # ---- Aggregate fitness: inverse-variance weighted mean per merge_seq ----
    for fc, sc in zip(fitness_cols, sigma_cols):
        if fc not in nonsyn_df.columns or sc not in nonsyn_df.columns:
            continue
        # weighted sum: f/s² → per group sum, then divide by sum(1/s²)
        nonsyn_df = nonsyn_df.with_columns([
            (
                pl.when(pl.col(fc).is_not_null() & pl.col(sc).is_not_null())
                .then(pl.col(fc) / (pl.col(sc) ** 2))
                .otherwise(None)
                .sum().over("merge_seq")
            ).alias(f"{fc}_wsum"),
            (
                pl.when(pl.col(sc).is_not_null())
                .then(1.0 / (pl.col(sc) ** 2))
                .otherwise(None)
                .sum().over("merge_seq")
            ).alias(f"{sc}_wsum"),
        ])
        nonsyn_df = nonsyn_df.with_columns([
            pl.when(pl.col(f"{sc}_wsum") > 0)
            .then(pl.col(f"{fc}_wsum") / pl.col(f"{sc}_wsum"))
            .otherwise(None)
            .alias(f"{fc}_agg"),
            pl.when(pl.col(f"{sc}_wsum") > 0)
            .then(1.0 / pl.col(f"{sc}_wsum").sqrt())
            .otherwise(None)
            .alias(f"{sc}_agg"),
        ])

    # ---- Deduplicate to one row per merge_seq ----
    agg_rename = {}
    for c in all_count_cols:
        if f"{c}_agg" in nonsyn_df.columns:
            agg_rename[f"{c}_agg"] = c
    for fc, sc in zip(fitness_cols, sigma_cols):
        if f"{fc}_agg" in nonsyn_df.columns:
            agg_rename[f"{fc}_agg"] = fc
        if f"{sc}_agg" in nonsyn_df.columns:
            agg_rename[f"{sc}_agg"] = sc

    # Columns to keep (drop the intermediate _wsum / original if overwritten)
    drop_cols = (
        [f"{c}_wsum" for c in sigma_cols if f"{c}_wsum" in nonsyn_df.columns] +
        [f"{c}_wsum" for c in fitness_cols if f"{c}_wsum" in nonsyn_df.columns]
    )
    # Also drop original count/fitness/sigma cols that were aggregated (they'll be renamed from _agg)
    original_to_drop = list(set(list(agg_rename.values())) & set(nonsyn_df.columns))

    unique_nonsyn = (
        nonsyn_df
        .drop(drop_cols + original_to_drop, strict=False)
        .unique(subset=["merge_seq"], keep="first")
        .rename(agg_rename)
    )

    # Recompute mean_count from aggregated input counts
    present_in_cols = [c for c in input_cols if c in unique_nonsyn.columns]
    if present_in_cols:
        unique_nonsyn = unique_nonsyn.with_columns(
            pl.mean_horizontal(*[pl.col(c).cast(pl.Float64) for c in present_in_cols]).alias("mean_count")
        )

    # Nham_nt / Nmut_codons are meaningless for nonsynonymous after AA aggregation
    unique_nonsyn = unique_nonsyn.with_columns([
        pl.when(pl.col("Nham_aa").fill_null(0) != 0)
        .then(None).otherwise(pl.col("Nham_nt"))
        .cast(pl.Int32).alias("Nham_nt"),
        pl.when(pl.col("Nham_aa").fill_null(0) != 0)
        .then(None).otherwise(pl.col("Nmut_codons"))
        .cast(pl.Int32).alias("Nmut_codons"),
    ])

    # Return: syn_df (error_model=F, for synonymous.txt)
    #       + syn_truth_df (original error_model=T, for wildtype.txt / all_variants)
    #       + unique_nonsyn (aggregated nonsynonymous, error_model=T)
    return pl.concat([syn_df, syn_truth_df, unique_nonsyn], how="diagonal_relaxed")
