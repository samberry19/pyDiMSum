"""Stage 5 (STEAM) — fitness calculation, count filtering, pseudocounts.

Mirrors:
  R/dimsum__filter_nuc_variants.R
  R/dimsum__calculate_fitness.R
  R/dimsum__add_dropout_pseudocount.R
  R/dimsum__normalise_fitness.R
"""

from __future__ import annotations

import logging
from typing import Union

import polars as pl

from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Count filtering
# ---------------------------------------------------------------------------

def filter_low_counts(
    df: pl.DataFrame,
    config: RunConfig,
    replicates: list[int],
) -> pl.DataFrame:
    """Remove and NaN-out low-count variants.

    Mirrors: R/dimsum__filter_nuc_variants.R

    For "All" thresholds: drop rows where any replicate is below threshold.
    For "Any" thresholds: drop rows where ALL replicates are below threshold
    (i.e. require at least one above), and NaN individual columns below threshold.
    """
    reps = replicates

    # Identify input/output count column names (bio-rep aggregated: count_e{E}_s0/s1)
    in_cols = [f"count_e{E}_s0" for E in reps if f"count_e{E}_s0" in df.columns]
    out_cols = [f"count_e{E}_s1" for E in reps if f"count_e{E}_s1" in df.columns]

    min_in_any = config._fitness_min_input_count_any_parsed
    min_in_all = config._fitness_min_input_count_all_parsed
    min_out_any = config._fitness_min_output_count_any_parsed
    min_out_all = config._fitness_min_output_count_all_parsed

    nham_col = "Nham_nt"  # use nt for filtering regardless of seq type

    df = _apply_any_filter(df, in_cols, min_in_any, nham_col, keep_if_any=True)
    df = _apply_any_filter(df, out_cols, min_out_any, nham_col, keep_if_any=True)
    df = _apply_all_filter(df, in_cols, min_in_all, nham_col)
    df = _apply_all_filter(df, out_cols, min_out_all, nham_col)

    # Set individual below-threshold cells to null
    df = _null_below_threshold(df, in_cols, min_in_any, nham_col)
    df = _null_below_threshold(df, out_cols, min_out_any, nham_col)

    return df


def _threshold_for_nham(threshold: Union[int, dict], nham: int | None) -> int:
    """Return the threshold applicable for a given Nham value."""
    if isinstance(threshold, int):
        return threshold
    if nham is None:
        return 0
    return threshold.get(nham, 0)


def _apply_any_filter(
    df: pl.DataFrame,
    cols: list[str],
    threshold: Union[int, dict],
    nham_col: str,
    keep_if_any: bool = True,
) -> pl.DataFrame:
    """Keep rows where at least one col >= threshold (or drop if none qualify)."""
    if isinstance(threshold, int):
        if threshold == 0:
            return df
        keep_expr = pl.lit(False)
        for c in cols:
            keep_expr = keep_expr | (pl.col(c).fill_null(0) >= threshold)
        return df.filter(keep_expr)
    else:
        # Dict: apply per Nham_nt group
        result_parts = []
        for nham_val, thresh in threshold.items():
            sub = df.filter(pl.col(nham_col) == nham_val)
            if thresh > 0 and len(cols) > 0:
                keep_expr = pl.lit(False)
                for c in cols:
                    keep_expr = keep_expr | (pl.col(c).fill_null(0) >= thresh)
                sub = sub.filter(keep_expr)
            result_parts.append(sub)
        # Rows not matching any Nham_nt threshold key are kept
        all_nham_vals = list(threshold.keys())
        rest = df.filter(~pl.col(nham_col).is_in(all_nham_vals))
        result_parts.append(rest)
        return pl.concat(result_parts)


def _apply_all_filter(
    df: pl.DataFrame,
    cols: list[str],
    threshold: Union[int, dict],
    nham_col: str,
) -> pl.DataFrame:
    """Drop rows where any col < threshold (require ALL to be >= threshold)."""
    if isinstance(threshold, int):
        if threshold == 0:
            return df
        keep_expr = pl.lit(True)
        for c in cols:
            keep_expr = keep_expr & (pl.col(c).fill_null(0) >= threshold)
        return df.filter(keep_expr)
    else:
        result_parts = []
        for nham_val, thresh in threshold.items():
            sub = df.filter(pl.col(nham_col) == nham_val)
            if thresh > 0 and len(cols) > 0:
                keep_expr = pl.lit(True)
                for c in cols:
                    keep_expr = keep_expr & (pl.col(c).fill_null(0) >= thresh)
                sub = sub.filter(keep_expr)
            result_parts.append(sub)
        all_nham_vals = list(threshold.keys())
        rest = df.filter(~pl.col(nham_col).is_in(all_nham_vals))
        result_parts.append(rest)
        return pl.concat(result_parts)


def _null_below_threshold(
    df: pl.DataFrame,
    cols: list[str],
    threshold: Union[int, dict],
    nham_col: str,
) -> pl.DataFrame:
    """Set individual cells to null where count < threshold."""
    if isinstance(threshold, int):
        if threshold == 0:
            return df
        for c in cols:
            df = df.with_columns(
                pl.when(pl.col(c).fill_null(0) < threshold)
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
            )
    else:
        for nham_val, thresh in threshold.items():
            if thresh > 0:
                for c in cols:
                    df = df.with_columns(
                        pl.when(
                            (pl.col(nham_col) == nham_val) &
                            (pl.col(c).fill_null(0) < thresh)
                        )
                        .then(None)
                        .otherwise(pl.col(c))
                        .alias(c)
                    )
    return df


# ---------------------------------------------------------------------------
# Pseudocounts
# ---------------------------------------------------------------------------

def add_dropout_pseudocount(
    df: pl.DataFrame,
    config: RunConfig,
    replicates: list[int],
) -> pl.DataFrame:
    """Add pseudocount to output samples that have dropout (count=0 with input > 0).

    Mirrors: R/dimsum__add_dropout_pseudocount.R
    """
    if config.fitness_dropout_pseudocount == 0:
        return df
    pseudo = config.fitness_dropout_pseudocount
    for E in replicates:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        df = df.with_columns(
            pl.when(
                (pl.col(s0) > 0) & (pl.col(s1) == 0)
            )
            .then(pl.col(s1) + pseudo)
            .otherwise(pl.col(s1))
            .alias(s1)
        )
    return df


# ---------------------------------------------------------------------------
# Fitness calculation
# ---------------------------------------------------------------------------

def calculate_fitness(
    df: pl.DataFrame,
    config: RunConfig,
    replicates: list[int],
    error_model_df: "pl.DataFrame | None",
    norm_model_df: "pl.DataFrame | None",
) -> pl.DataFrame:
    """Compute per-replicate fitness and sigma, remove no-fitness variants.

    Fitness_j = log(out_j / in_j) - log(WT_out_j / WT_in_j)

    Mirrors: R/dimsum__calculate_fitness.R
    """
    from pydimsum.steam.error_model import compute_sigma

    reps = replicates

    if norm_model_df is not None:
        scale_vals = {E: float(norm_model_df[f"scale_{E}"][0]) for E in reps}
        shift_vals = {E: float(norm_model_df[f"shift_{E}"][0]) for E in reps}
    else:
        scale_vals = {E: 1.0 for E in reps}
        shift_vals = {E: 0.0 for E in reps}

    # ---- Calculate raw fitness ----
    for E in reps:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        # WT correction
        wt_rows = df.filter(pl.col("WT") & pl.col("error_model"))
        wt_s0 = float(wt_rows[s0].drop_nulls()[0])
        wt_s1 = float(wt_rows[s1].drop_nulls()[0])
        wt_corr = float(pl.Series([float(wt_s1)]).log()[0]) - float(pl.Series([float(wt_s0)]).log()[0])

        import math
        df = df.with_columns(
            pl.when(
                (pl.col(s0) > 0) & (pl.col(s1) > 0)
            ).then(
                (pl.col(s1).cast(pl.Float64) / pl.col(s0).cast(pl.Float64)).log() - wt_corr
            ).otherwise(None)
            .alias(f"fitness{E}_uncorr")
        )

    # ---- Apply normalisation ----
    if config.fitness_normalise and norm_model_df is not None:
        # Compute WT correction after normalisation
        fitness_uncorr_cols = [f"fitness{E}_uncorr" for E in reps]
        wt_rows = df.filter(pl.col("WT") & pl.col("error_model"))
        shifts_arr = pl.Series([shift_vals[E] for E in reps])
        scales_arr = pl.Series([scale_vals[E] for E in reps])

        # wt_corr = mean((wt_fitness + shift) * scale) across replicates
        wt_fits = pl.Series([
            float(wt_rows[f"fitness{E}_uncorr"].drop_nulls()[0]) for E in reps
        ])
        wt_corr = float(((wt_fits + shifts_arr) * scales_arr).mean())

        for E in reps:
            df = df.with_columns(
                (
                    (pl.col(f"fitness{E}_uncorr") + shift_vals[E]) * scale_vals[E] - wt_corr
                ).alias(f"fitness{E}_uncorr")
            )

    # ---- Compute sigma ----
    df = compute_sigma(
        df=df,
        replicates=reps,
        error_model_df=error_model_df,
        norm_model_df=norm_model_df,
        fitness_normalise=config.fitness_normalise,
        fitness_error_model=config.fitness_error_model,
    )

    # ---- Remove variants without any fitness estimate ----
    any_fitness = pl.lit(False)
    for E in reps:
        any_fitness = any_fitness | pl.col(f"fitness{E}_uncorr").is_not_null()
    df = df.filter(any_fitness)

    # ---- Mean input count ----
    df = df.with_columns(
        pl.mean_horizontal(*[f"count_e{E}_s0" for E in reps]).alias("mean_count")
    )

    return df


# ---------------------------------------------------------------------------
# Generation normalisation
# ---------------------------------------------------------------------------

def normalise_fitness_by_generations(
    df: pl.DataFrame,
    exp_design_df: "pl.DataFrame",
    replicates: list[int],
    fitness_suffix: str = "",
) -> pl.DataFrame:
    """Normalise fitness and sigma by number of generations.

    fitness_norm = log2(exp(fitness / mean_generations))

    Mirrors: R/dimsum__normalise_fitness.R
    """
    if df.is_empty():
        return df

    # Mean generations per replicate
    gen_data = (
        exp_design_df
        .filter(pl.col("selection_id") == 1)
        .select(["experiment", "generations"])
        .filter(pl.col("generations").is_not_null())
        .group_by("experiment")
        .agg(pl.col("generations").mean().alias("mean_gen"))
    )
    gen_dict = dict(zip(
        gen_data["experiment"].to_list(),
        gen_data["mean_gen"].to_list(),
    ))

    import math
    for E in replicates:
        if E not in gen_dict:
            logger.warning("No generations data for replicate %d; skipping normalisation", E)
            continue
        g = gen_dict[E]
        for col_prefix in ["fitness", "sigma"]:
            col = f"{col_prefix}{E}{fitness_suffix}"
            if col in df.columns:
                # log2(exp(x / g)) = x / g * log2(e) = x / (g * ln(2))
                df = df.with_columns(
                    (pl.col(col) / (g * math.log(2))).alias(col)
                )
    return df
