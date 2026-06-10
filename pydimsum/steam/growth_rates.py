"""Stage 5 (STEAM) — infer growth rates from fitness.

Mirrors: R/dimsum__infer_growth_rates.R

The growth rate for a variant is estimated by:
  1. Computing per-replicate growth rates from raw counts and cell-density data.
  2. Fitting a linear model growthrate_E ~ fitness_E_uncorr (error_model rows only).
  3. Using that model to predict growth rates for ALL variants from their fitness.
  4. Merging replicate growth rates via inverse-variance weighting.
  5. Fitting a final linear model on the merged values to get the canonical
     `growthrate` and `growthrate_sigma` columns.

Only called when all selection_id==1 rows in exp_design have non-null
cell_density and selection_time values.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.stats import linregress

logger = logging.getLogger(__name__)

_EPS = 1e-30  # guard against log(0)


def infer_growth_rates(
    df: pl.DataFrame,
    exp_design_df: pl.DataFrame,
    replicates: list[int],
) -> pl.DataFrame:
    """Add ``growthrate`` and ``growthrate_sigma`` columns to *df*.

    Parameters
    ----------
    df:
        Variant table with per-replicate ``count_e{E}_s0``, ``count_e{E}_s1``,
        ``fitness{E}_uncorr``, ``sigma{E}_uncorr``, ``error_model``,
        ``fitness``, and ``sigma`` columns (the last two from
        ``_inverse_variance_merge``).
    exp_design_df:
        Experiment design table (``selection_id``, ``experiment``,
        ``cell_density``, ``selection_time`` columns).
    replicates:
        List of replicate integers.

    Returns
    -------
    DataFrame extended with ``growthrate`` and ``growthrate_sigma``.
    """
    if df.is_empty():
        return df

    reps = replicates

    # ---- Extract per-replicate experiment design parameters ----
    sel1 = exp_design_df.filter(pl.col("selection_id") == 1)
    sel0 = exp_design_df.filter(pl.col("selection_id") == 0)

    def _mean_per_rep(sub: pl.DataFrame, col: str) -> dict[int, float]:
        grp = (
            sub.filter(pl.col(col).is_not_null())
            .group_by("experiment")
            .agg(pl.col(col).mean().alias("_m"))
        )
        return {int(row["experiment"]): float(row["_m"]) for row in grp.iter_rows(named=True)}

    selection_time = _mean_per_rep(sel1, "selection_time")
    cell_density_out = _mean_per_rep(sel1, "cell_density")
    cell_density_in = _mean_per_rep(sel0, "cell_density")

    # ---- Step 1: per-replicate growth rate from raw counts ----
    # growthrate_E = log(freq_out * cd_out / (freq_in * cd_in)) / time
    # where freq = count / sum(count) among error_model rows
    em_mask = pl.col("error_model").fill_null(False)
    em_idx = df["error_model"].fill_null(False).to_numpy()

    for E in reps:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        t = selection_time.get(E)
        cd_out = cell_density_out.get(E)
        cd_in = cell_density_in.get(E)

        if t is None or cd_out is None or cd_in is None or t <= 0:
            logger.warning(
                "Replicate %d: missing selection_time or cell_density — "
                "skipping growth rate calculation for this replicate.",
                E,
            )
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"growthrate{E}"))
            continue

        # Sums over error_model==True rows
        em_s0 = df.filter(em_mask)[s0].fill_null(0).cast(pl.Float64).to_numpy()
        em_s1 = df.filter(em_mask)[s1].fill_null(0).cast(pl.Float64).to_numpy()
        sum_s0 = float(em_s0.sum())
        sum_s1 = float(em_s1.sum())

        if sum_s0 <= 0 or sum_s1 <= 0:
            logger.warning("Replicate %d: zero total counts — skipping.", E)
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"growthrate{E}"))
            continue

        # Compute on all rows (not just error_model), then we only FIT on error_model rows
        counts_s0 = df[s0].fill_null(0).cast(pl.Float64).to_numpy()
        counts_s1 = df[s1].fill_null(0).cast(pl.Float64).to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            gr = np.log(
                (counts_s1 / sum_s1 * cd_out) /
                (counts_s0 / sum_s0 * cd_in + _EPS)
            ) / t

        # Mask zeros/negatives → nan
        gr = np.where(counts_s0 > 0, gr, np.nan)
        gr = np.where(counts_s1 > 0, gr, np.nan)

        df = df.with_columns(
            pl.Series(f"growthrate{E}_raw", gr.tolist(), dtype=pl.Float64)
        )

    # ---- Step 2: fit linear model growthrate_E_raw ~ fitness_E_uncorr ----
    # (on error_model==True rows with finite values only)
    for E in reps:
        gr_col = f"growthrate{E}_raw"
        if gr_col not in df.columns:
            continue
        fit_col = f"fitness{E}_uncorr"
        sig_col = f"sigma{E}_uncorr"
        if fit_col not in df.columns:
            continue

        em_df = df.filter(em_mask).select([gr_col, fit_col])
        gr_arr = em_df[gr_col].to_numpy().astype(float)
        fit_arr = em_df[fit_col].to_numpy().astype(float)

        finite = np.isfinite(gr_arr) & np.isfinite(fit_arr)
        if finite.sum() < 3:
            logger.warning(
                "Replicate %d: too few finite points for growth rate regression.",
                E,
            )
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"growthrate{E}"))
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"growthrate{E}_sigma"))
            continue

        slope, intercept, r, *_ = linregress(fit_arr[finite], gr_arr[finite])
        logger.info(
            "Replicate %d: growthrate = fitness * %.4f + %.4f  (r=%.3f)",
            E, slope, intercept, r,
        )

        # Apply model to all variants (from fitness, not raw growth rate)
        if fit_col in df.columns and sig_col in df.columns:
            df = df.with_columns([
                (pl.col(fit_col) * slope + intercept).alias(f"growthrate{E}"),
                (pl.col(sig_col) * abs(slope)).alias(f"growthrate{E}_sigma"),
            ])
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"growthrate{E}"))
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"growthrate{E}_sigma"))

    # Drop the raw intermediate columns
    raw_cols = [c for c in df.columns if c.endswith("_raw") and c.startswith("growthrate")]
    if raw_cols:
        df = df.drop(raw_cols)

    # ---- Step 3: inverse-variance merge across replicates ----
    gr_cols = [f"growthrate{E}" for E in reps if f"growthrate{E}" in df.columns]
    sig_cols = [f"growthrate{E}_sigma" for E in reps if f"growthrate{E}_sigma" in df.columns]

    if not gr_cols or not sig_cols:
        logger.warning("No per-replicate growth rates computed — skipping merge.")
        return df

    # Inverse-variance weighted mean
    numerator = pl.lit(0.0)
    denominator = pl.lit(0.0)
    for gc, sc in zip(gr_cols, sig_cols):
        valid = pl.col(gc).is_not_null() & pl.col(sc).is_not_null() & (pl.col(sc) > 0)
        w = pl.when(valid).then(1.0 / (pl.col(sc) ** 2)).otherwise(pl.lit(0.0))
        numerator = numerator + pl.when(valid).then(
            pl.col(gc) / (pl.col(sc) ** 2)
        ).otherwise(pl.lit(0.0))
        denominator = denominator + w

    df = df.with_columns([
        pl.when(denominator > 0).then(numerator / denominator).otherwise(None)
        .alias("growthrate_merged"),
    ])

    # ---- Step 4: final linear model on merged values ----
    # growthrate_merged ~ fitness (inverse-variance merged fitness)
    if "fitness" not in df.columns or "sigma" not in df.columns:
        df = df.rename({"growthrate_merged": "growthrate"})
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("growthrate_sigma"))
        return df

    em_df2 = df.filter(em_mask).select(["growthrate_merged", "fitness"])
    gr2 = em_df2["growthrate_merged"].to_numpy().astype(float)
    fit2 = em_df2["fitness"].to_numpy().astype(float)
    finite2 = np.isfinite(gr2) & np.isfinite(fit2)

    if finite2.sum() >= 3:
        slope2, intercept2, r2, *_ = linregress(fit2[finite2], gr2[finite2])
        logger.info(
            "Final growth rate model: growthrate = fitness * %.4f + %.4f  (r=%.3f)",
            slope2, intercept2, r2,
        )
        df = df.with_columns([
            (pl.col("fitness") * slope2 + intercept2).alias("growthrate"),
            (pl.col("sigma") * abs(slope2)).alias("growthrate_sigma"),
        ])
    else:
        df = df.rename({"growthrate_merged": "growthrate"})
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("growthrate_sigma"))
        return df

    df = df.drop("growthrate_merged")
    return df


def has_growth_rate_data(exp_design_df: pl.DataFrame) -> bool:
    """Return True if exp_design has non-null cell_density and selection_time
    for all selection_id==1 rows."""
    sel1 = exp_design_df.filter(pl.col("selection_id") == 1)
    if sel1.is_empty():
        return False
    for col in ("cell_density", "selection_time"):
        if col not in sel1.columns:
            return False
        if sel1[col].is_null().any():
            return False
    sel0 = exp_design_df.filter(pl.col("selection_id") == 0)
    if "cell_density" not in sel0.columns:
        return False
    if sel0["cell_density"].is_null().any():
        return False
    return True
