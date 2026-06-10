"""Enrichment mode — per-sequence library analysis.

Used when ``config.enrichment_mode`` is True.  The library contains distinct
sequences that are not variants of a common wildtype (e.g. different IDRs fused to
a shared domain).

Replaces the mutation-centric ``process_variants`` → ``calculate_fitness`` →
``mutations`` → ``merge_fitness`` chain with a sequence-agnostic analogue:

  1. ``process_library_variants``  — annotate sequences; no WT-based filters.
  2. ``calculate_enrichment``      — log(out/in) per replicate; non-WT centering.
  3. ``write_enrichment_outputs``  — flat TSV + Parquet bundle.

Reused unchanged from the mutation path:
  - ``steam/fitness.py``:  ``filter_low_counts``, ``add_dropout_pseudocount``,
                            ``normalise_fitness_by_generations``
  - ``steam/error_model.py``: ``_fit_normalisation`` (inter-rep scale/shift fit)
  - ``steam/merge_fitness.py``: ``_inverse_variance_merge``,
                                 ``_write_parquet_bundle``
  - ``steam/sequences.py``: ``translate_sequences_fast``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 4 equivalent: annotate library variants
# ---------------------------------------------------------------------------

def process_library_variants(
    df: pl.DataFrame,
    config: "RunConfig",
) -> pl.DataFrame:
    """Annotate a library count table without any WT-based filtering.

    Parameters
    ----------
    df:
        Wide count table from ``build_variant_table``:
        ``nt_seq`` + ``count_e{E}_s0`` / ``count_e{E}_s1`` columns.
    config:
        Validated pipeline configuration (``enrichment_mode`` must be True).

    Returns
    -------
    Annotated DataFrame with columns:
      ``nt_seq``, ``length``, ``merge_seq``, ``WT`` (always False),
      ``error_model`` (always True), ``Nham_nt`` (always 0, for compatibility
      with ``filter_low_counts``), optionally ``aa_seq`` (coding mode),
      ``is_reference``, ``is_spikein``, and all count_e* columns.
    """
    if not config.enrichment_mode:
        raise ValueError("process_library_variants called outside enrichment_mode")

    seq_type = config.sequence_type_resolved
    logger.info(
        "Library mode: %d unique sequences, sequence_type=%s",
        len(df), seq_type,
    )

    # Base annotations
    df = df.with_columns([
        pl.col("nt_seq").str.len_chars().alias("length"),
        pl.col("nt_seq").alias("merge_seq"),
        pl.lit(False).alias("WT"),
        pl.lit(True).alias("error_model"),
        pl.lit(0).cast(pl.Int32).alias("Nham_nt"),
    ])

    # Optional AA translation (coding mode)
    if seq_type == "coding":
        lengths = df["length"].to_list()
        if all(l % 3 == 0 for l in lengths if l is not None):
            from pydimsum.steam.sequences import translate_sequences_fast
            aa_seqs = translate_sequences_fast(df["nt_seq"].to_list())
            df = df.with_columns(
                pl.Series("aa_seq", aa_seqs, dtype=pl.Utf8)
            )
            logger.info("Coding mode: added aa_seq column")
        else:
            logger.warning(
                "Some sequences have lengths not divisible by 3 — "
                "skipping translation despite sequence_type='coding'."
            )

    # Reference and spike-in flags
    ref_id = config.enrichment_reference_id
    df = df.with_columns(
        (pl.col("nt_seq") == pl.lit(ref_id or "")).alias("is_reference")
    )

    spikein_ids: set[str] = set()
    if config.enrichment_spikein_ids:
        spikein_ids = {s.strip() for s in config.enrichment_spikein_ids.split(",")}
    df = df.with_columns(
        pl.col("nt_seq").is_in(list(spikein_ids)).alias("is_spikein")
    )

    # Warn if reference not found
    if config.enrichment_normalise == "reference" and ref_id:
        if not df["is_reference"].any():
            logger.warning(
                "enrichment_reference_id %r not found in the count table. "
                "Normalisation will fall back to zero offset.",
                ref_id,
            )
    if config.enrichment_normalise == "spikein" and spikein_ids:
        n_spikein = int(df["is_spikein"].sum())
        if n_spikein == 0:
            logger.warning(
                "No spike-in sequences found in the count table. "
                "Normalisation will fall back to zero offset."
            )

    logger.info("Library variant table: %d sequences annotated", len(df))
    return df


# ---------------------------------------------------------------------------
# Stage 5 equivalent: compute per-sequence enrichment
# ---------------------------------------------------------------------------

def calculate_enrichment(
    df: pl.DataFrame,
    config: "RunConfig",
    replicates: list[int],
    norm_model_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Compute per-replicate log(out/in) enrichment with non-WT centering.

    Parameters
    ----------
    df:
        Annotated library table from ``process_library_variants`` (after count
        filtering and pseudocounts have been applied).
    config:
        Validated pipeline configuration.
    replicates:
        List of replicate integers.
    norm_model_df:
        Replicate scale/shift model from ``_fit_normalisation``, or None.
        When supplied, scale/shift is applied before offset centering.

    Returns
    -------
    DataFrame extended with per-replicate ``enrichment{E}_uncorr``,
    ``sigma{E}_uncorr``, and ``mean_count`` columns.  Rows with no enrichment
    estimate in any replicate are dropped.
    """
    reps = replicates

    # ---- Step 1: raw log(out/in) per replicate ----
    for E in reps:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        df = df.with_columns(
            pl.when(
                (pl.col(s0) > 0) & (pl.col(s1) > 0)
            ).then(
                (pl.col(s1).cast(pl.Float64) / pl.col(s0).cast(pl.Float64)).log()
            ).otherwise(None)
            .alias(f"enrichment{E}_uncorr")
        )

    # ---- Step 2: apply replicate scale/shift if available ----
    if norm_model_df is not None:
        scale_vals = {E: float(norm_model_df[f"scale_{E}"][0]) for E in reps}
        shift_vals = {E: float(norm_model_df[f"shift_{E}"][0]) for E in reps}
        for E in reps:
            df = df.with_columns(
                pl.when(pl.col(f"enrichment{E}_uncorr").is_not_null())
                .then(
                    (pl.col(f"enrichment{E}_uncorr") + shift_vals[E]) * scale_vals[E]
                )
                .otherwise(None)
                .alias(f"enrichment{E}_uncorr")
            )
    else:
        scale_vals = {E: 1.0 for E in reps}
        shift_vals = {E: 0.0 for E in reps}

    # ---- Step 3: compute normalisation offset per replicate ----
    strategy = config.enrichment_normalise
    offsets: dict[int, float] = {}

    for E in reps:
        col = f"enrichment{E}_uncorr"
        enr_vals = df[col].drop_nulls()

        if strategy == "none":
            offsets[E] = 0.0

        elif strategy == "median":
            offsets[E] = float(enr_vals.median()) if len(enr_vals) > 0 else 0.0

        elif strategy == "total":
            s0 = f"count_e{E}_s0"
            s1 = f"count_e{E}_s1"
            total_in = float(df[s0].fill_null(0).sum())
            total_out = float(df[s1].fill_null(0).sum())
            if total_in > 0 and total_out > 0:
                import math
                offsets[E] = math.log(total_out / total_in)
                # Apply scale/shift to the total-read offset too
                offsets[E] = (offsets[E] + shift_vals[E]) * scale_vals[E]
            else:
                offsets[E] = 0.0

        elif strategy == "reference":
            ref_id = config.enrichment_reference_id or ""
            ref_rows = df.filter(pl.col("nt_seq") == ref_id)
            if len(ref_rows) > 0:
                val = ref_rows[col].drop_nulls()
                offsets[E] = float(val[0]) if len(val) > 0 else 0.0
            else:
                offsets[E] = 0.0

        elif strategy == "spikein":
            spikein_rows = df.filter(pl.col("is_spikein"))
            if len(spikein_rows) > 0:
                vals = spikein_rows[col].drop_nulls()
                offsets[E] = float(vals.mean()) if len(vals) > 0 else 0.0
            else:
                offsets[E] = 0.0
        else:
            offsets[E] = 0.0

    # ---- Step 4: subtract offset ----
    for E in reps:
        col = f"enrichment{E}_uncorr"
        offset = offsets[E]
        df = df.with_columns(
            pl.when(pl.col(col).is_not_null())
            .then(pl.col(col) - offset)
            .otherwise(None)
            .alias(col)
        )
        logger.debug("Replicate %d: normalisation offset = %.4f", E, offset)

    # ---- Step 5: count-based sigma (Poisson, no WT correction term) ----
    for E in reps:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        col = f"enrichment{E}_uncorr"
        Corr = abs(scale_vals[E])
        df = df.with_columns(
            pl.when(pl.col(col).is_not_null())
            .then(
                (
                    Corr * (
                        1.0 / pl.col(s1).cast(pl.Float64) +
                        1.0 / pl.col(s0).cast(pl.Float64)
                    )
                ).sqrt()
            )
            .otherwise(None)
            .alias(f"sigma{E}_uncorr")
        )

    # ---- Step 6: mean input count, drop rows with no estimate ----
    df = df.with_columns(
        pl.mean_horizontal(*[f"count_e{E}_s0" for E in reps]).alias("mean_count")
    )
    any_enrichment = pl.lit(False)
    for E in reps:
        any_enrichment = any_enrichment | pl.col(f"enrichment{E}_uncorr").is_not_null()
    df = df.filter(any_enrichment)

    logger.info(
        "Enrichment calculated for %d sequences (%d replicates)",
        len(df), len(reps),
    )
    return df


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_enrichment_outputs(
    df: pl.DataFrame,
    replicates: list[int],
    config: "RunConfig",
) -> None:
    """Merge replicate enrichments and write enrichment output files.

    Parameters
    ----------
    df:
        DataFrame from ``calculate_enrichment`` (with ``enrichment{E}_uncorr``
        and ``sigma{E}_uncorr`` columns).
    replicates:
        List of replicate integers.
    config:
        Validated pipeline configuration.
    """
    from pydimsum.steam.merge_fitness import (
        _inverse_variance_merge,
        _write_parquet_bundle,
    )

    reps = replicates
    output_path = config.project_path
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Rename enrichment → fitness columns for the generic merge helper ----
    # _inverse_variance_merge looks for fitness{E}{suffix}/sigma{E}{suffix}
    # Our columns are named enrichment{E}_uncorr / sigma{E}_uncorr, so
    # temporarily alias them.
    rename_map_to = {f"enrichment{E}_uncorr": f"fitness{E}_uncorr" for E in reps}
    rename_map_back = {v: k for k, v in rename_map_to.items()}

    df_tmp = df.rename(rename_map_to)
    df_tmp = _inverse_variance_merge(
        df_tmp, reps,
        fitness_suffix="_uncorr",
        sigma_suffix="_uncorr",
        out_fitness="fitness",
        out_sigma="sigma",
    )
    # Rename back enrichment columns
    df_tmp = df_tmp.rename(rename_map_back)

    # ---- Select and order output columns ----
    base_cols = ["nt_seq"]
    if "aa_seq" in df_tmp.columns:
        base_cols.append("aa_seq")
    base_cols += ["length", "mean_count"]
    enr_sig_cols = []
    for E in reps:
        ecol = f"enrichment{E}_uncorr"
        scol = f"sigma{E}_uncorr"
        # Rename to drop _uncorr suffix for the output table
        if ecol in df_tmp.columns:
            df_tmp = df_tmp.rename({ecol: f"enrichment{E}"})
            enr_sig_cols.append(f"enrichment{E}")
        if scol in df_tmp.columns:
            df_tmp = df_tmp.rename({scol: f"sigma{E}"})
            enr_sig_cols.append(f"sigma{E}")
    merged_cols = ["fitness", "sigma"]

    all_out_cols = base_cols + enr_sig_cols + merged_cols
    present = [c for c in all_out_cols if c in df_tmp.columns]
    out_df = df_tmp.select(present)

    # ---- Write tab-separated text ----
    tsv_path = output_path / "enrichment_variant_data.txt"
    out_df.write_csv(str(tsv_path), separator="\t", null_value="NA")
    logger.info("Written enrichment table: %s (%d rows)", tsv_path, len(out_df))

    # ---- Write Parquet bundle ----
    parquet_path = output_path / f"{config.project_name}_enrichment.parquet"
    _write_parquet_bundle({"variants": out_df}, parquet_path)
    logger.info("Saved Parquet enrichment bundle: %s", parquet_path)
