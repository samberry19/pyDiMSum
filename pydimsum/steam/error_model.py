"""Stage 5 (STEAM) — replicate normalisation and error model fitting.

Mirrors:
  R/dimsum__error_model.R
  R/dimsum__fit_error_model.R
  R/dimsum__fit_error_model_bootstrap.R
  R/dimsum__replicate_fitness_deviation.R

Key improvements over the R version:
  - scipy.optimize.minimize (L-BFGS-B) for normalisation (replaces nlm)
  - scipy.optimize.least_squares for error model (replaces nls + retry loop)
  - ProcessPoolExecutor for bootstraps (shared-memory-compatible)
  - No per-bootstrap rbindlist of 500 matrices; use NumPy directly
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import least_squares, minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public result containers
# ---------------------------------------------------------------------------

ErrorModelResult = dict  # keys: "error_model" (pl.DataFrame | None), "norm_model" (pl.DataFrame | None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_error_model(
    df: pl.DataFrame,
    replicates: list[int],
    fitness_normalise: bool,
    fitness_error_model: bool,
    num_cores: int = 1,
    n_bootstraps: int = 100,
    max_n_per_bootstrap: int = 10_000,
    lower_rep: float = 1e-4,
    seed: int = 1234567,
) -> ErrorModelResult:
    """Fit the replicate normalisation and over-sequencing error models.

    Parameters
    ----------
    df:
        Variant table with ``Nham_nt``, ``Nham_aa``, ``WT``, ``error_model``
        flags and ``count_e{E}_s0`` / ``count_e{E}_s1`` columns.
        Also needs ``Nmut_codons`` for coding sequences.
    replicates:
        List of replicate integers.
    fitness_normalise, fitness_error_model:
        Flags from config.
    num_cores:
        Parallelism for bootstrap fitting.
    seed:
        RNG seed (mirrors R clusterSetRNGStream(cl, 1234567)).

    Returns
    -------
    dict with keys:
      ``error_model`` : pl.DataFrame or None
      ``norm_model``  : pl.DataFrame or None
    """
    if not fitness_error_model:
        return {"error_model": None, "norm_model": None}

    nreps = len(replicates)
    reps = replicates

    # ---- Work data: keep only needed columns ----
    all_count_cols = (
        [f"count_e{E}_s0" for E in reps] +
        [f"count_e{E}_s1" for E in reps]
    )
    meta_cols = ["Nham_nt", "Nham_aa", "WT", "error_model"]
    keep = [c for c in meta_cols + all_count_cols if c in df.columns]
    work = df.select(keep)

    # ---- Calculate per-replicate fitness ----
    for E in reps:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        work = work.with_columns(
            (
                pl.when(
                    (pl.col(s0) > 0) & (pl.col(s1) > 0)
                ).then(
                    (pl.col(s1).cast(pl.Float64) / pl.col(s0).cast(pl.Float64)).log()
                ).otherwise(None)
            ).alias(f"fitness{E}")
        )

    # ---- Flag variants with reads in ALL input/output replicates ----
    has_all_expr = pl.lit(True)
    for E in reps:
        has_all_expr = has_all_expr & (pl.col(f"count_e{E}_s0") > 0) & (pl.col(f"count_e{E}_s1") > 0)
    work = work.with_columns(has_all_expr.alias("all_reads"))

    # Check WT present in all replicates
    wt_all_reads = work.filter(pl.col("WT") & pl.col("all_reads"))
    if len(wt_all_reads) == 0:
        raise RuntimeError(
            "WT variant has zero count in at least one input/output replicate. "
            "Cannot proceed with error modelling."
        )

    # ---- Input count threshold (1st percentile of fitness → exp(-p1) of counts) ----
    fitness_cols = [f"fitness{E}" for E in reps]
    above_threshold_mask = work.filter(pl.col("all_reads"))
    # Collect all fitness values above threshold
    flat_fitness = np.concatenate([
        above_threshold_mask[fc].drop_nulls().to_numpy() for fc in fitness_cols
    ])
    input_count_threshold = float(np.exp(-np.percentile(flat_fitness, 1)))

    # Mark variants above threshold
    above_thresh_expr = pl.lit(True)
    for E in reps:
        above_thresh_expr = above_thresh_expr & (
            pl.col(f"count_e{E}_s0").cast(pl.Float64) > input_count_threshold
        )
    work = work.with_columns(above_thresh_expr.alias("input_above_threshold"))

    # WT fitness correction
    wt_rows = work.filter(pl.col("WT"))
    for E in reps:
        wt_fit = float(wt_rows[f"fitness{E}"].drop_nulls()[0])
        work = work.with_columns(
            (pl.col(f"fitness{E}") - wt_fit).alias(f"fitness{E}")
        )

    # ---- Check sufficient variants for fitting ----
    min_n_variants = 10 * 3 * nreps
    n_usable = len(work.filter(
        pl.col("input_above_threshold") & pl.col("all_reads") & ~pl.col("WT").fill_null(False)
    ))
    if n_usable < min_n_variants:
        raise RuntimeError(
            f"Insufficient variants for error model fitting: "
            f"{n_usable} < {min_n_variants} required."
        )

    # ---- Replicate normalisation ----
    norm_model_df = _fit_normalisation(work, reps)
    logger.info("Normalisation model: %s", norm_model_df.to_dict(as_series=False))

    # Apply normalisation
    scale_vals = {E: float(norm_model_df[f"scale_{E}"][0]) for E in reps}
    shift_vals = {E: float(norm_model_df[f"shift_{E}"][0]) for E in reps}

    # WT correction so mean(WT) = 0
    wt_fit_mat = np.array([
        float(work.filter(pl.col("WT"))[f"fitness{E}"].drop_nulls()[0])
        for E in reps
    ])
    wt_corr = float(np.mean(
        (wt_fit_mat + np.array([shift_vals[E] for E in reps])) *
        np.array([scale_vals[E] for E in reps])
    ))

    for E in reps:
        work = work.with_columns(
            pl.when(pl.col("all_reads"))
            .then(
                (pl.col(f"fitness{E}") + shift_vals[E]) * scale_vals[E] - wt_corr
            )
            .otherwise(pl.col(f"fitness{E}"))
            .alias(f"fitness{E}_norm")
        )

    if fitness_normalise:
        for E in reps:
            work = work.with_columns(
                pl.when(pl.col("all_reads"))
                .then(pl.col(f"fitness{E}_norm"))
                .otherwise(pl.col(f"fitness{E}"))
                .alias(f"fitness{E}")
            )

    # ---- Count-based error (cbe) ----
    for E in reps:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        # WT count-based error for correction
        wt_s0 = float(work.filter(pl.col("WT"))[s0].drop_nulls()[0])
        wt_s1 = float(work.filter(pl.col("WT"))[s1].drop_nulls()[0])
        wt_cbe_add = 1.0 / wt_s1 + 1.0 / wt_s0

        if fitness_normalise:
            corr_factor = float(abs(scale_vals[E])) ** 0.5
        else:
            corr_factor = 1.0

        work = work.with_columns(
            (
                corr_factor * (
                    1.0 / pl.col(s1).cast(pl.Float64) +
                    1.0 / pl.col(s0).cast(pl.Float64) +
                    wt_cbe_add
                ).sqrt()
            ).alias(f"cbe{E}")
        )
        work = work.with_columns(
            pl.when(pl.col(f"fitness{E}").is_null())
            .then(None)
            .otherwise(pl.col(f"cbe{E}"))
            .alias(f"cbe{E}")
        )

    # Mean count-based error
    cbe_cols = [f"cbe{E}" for E in reps]
    work = work.with_columns(
        pl.mean_horizontal(*cbe_cols).alias("mean_cbe")
    )

    # Bin variants by mean_cbe^2
    n_bins = 50
    usable = work.filter(
        pl.col("input_above_threshold") & pl.col("all_reads")
    )
    cbe2 = usable["mean_cbe"].to_numpy() ** 2
    log_min = np.log10(np.nanpercentile(cbe2, 0.1))
    error_range = np.linspace(log_min, 0.0, n_bins)
    work = work.with_columns(
        pl.Series(
            "bin_error",
            np.digitize(work["mean_cbe"].to_numpy() ** 2, 10 ** error_range).tolist(),
            dtype=pl.Int32,
        )
    )

    # Diversity weighting (error_model_weighting)
    # Mirror: sqrt(max(.N, sqrt(nrow))) grouped by Nham_nt
    n_total = len(work)
    if "Nham_nt" in work.columns:
        grp_col = "Nham_nt"
    else:
        grp_col = "Nham_aa"
    grp_counts = work.group_by(grp_col).agg(pl.len().alias("_n"))
    work = work.join(grp_counts, on=grp_col, how="left")
    work = work.with_columns(
        (
            pl.when(pl.col("_n") > (n_total ** 0.5))
            .then(pl.col("_n").cast(pl.Float64).sqrt())
            .otherwise(pl.lit(float(n_total) ** 0.25))
        ).alias("error_model_weighting")
    ).drop("_n")

    # ---- Fit error model ----
    Fcorr = None
    if fitness_normalise:
        Fcorr = np.array([scale_vals[E] for E in reps])

    error_model_df = _fit_error_model_parallel(
        work=work,
        reps=reps,
        Fcorr=Fcorr,
        n_bootstraps=n_bootstraps,
        max_n=max_n_per_bootstrap,
        lower_rep=lower_rep,
        seed=seed,
        num_cores=num_cores,
    )

    return {
        "error_model": error_model_df,
        "norm_model": norm_model_df,
    }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _fit_normalisation(
    work: pl.DataFrame,
    reps: list[int],
) -> pl.DataFrame:
    """Fit scale/shift parameters to minimise inter-replicate fitness deviation.

    Mirrors: R/dimsum__error_model.R:152-182 + dimsum__replicate_fitness_deviation.R
    """
    nreps = len(reps)
    fitness_cols = [f"fitness{E}" for E in reps]

    # Only use variants with reads in all replicates and above threshold
    usable = work.filter(
        pl.col("input_above_threshold") & pl.col("all_reads")
    )
    F_data = usable.select(fitness_cols).to_numpy().astype(float)

    def deviation(p: np.ndarray) -> float:
        scales = p[:nreps]
        shifts = p[nreps:]
        F_norm = (F_data + shifts) * scales
        F_avg = (F_data + shifts).mean(axis=1, keepdims=True)
        diff = F_norm - F_avg
        return float(np.sum(np.sqrt(np.sum(diff ** 2, axis=1))))

    # Initialise with scale=1, shift=0 (mirrors R set.seed(1603) → nlm with p=rep(c(1,0), each=nreps))
    rng = np.random.default_rng(1603)
    p0 = np.concatenate([np.ones(nreps), np.zeros(nreps)])

    result = minimize(
        deviation,
        p0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12},
    )
    p = result.x

    # Normalise: set scale of first replicate to 1
    p[:nreps] = p[:nreps] / p[0]

    if np.any(p[:nreps] < 0):
        logger.warning(
            "Some scaling factors from replicate normalisation are negative. "
            "Check that input/output samples are not switched in experimentDesign file!"
        )

    col_names = [f"scale_{E}" for E in reps] + [f"shift_{E}" for E in reps]
    return pl.DataFrame({k: [round(float(v), 4)] for k, v in zip(col_names, p)})


# ---------------------------------------------------------------------------
# Error model bootstrap fitting
# ---------------------------------------------------------------------------

def _fit_error_model_parallel(
    work: pl.DataFrame,
    reps: list[int],
    Fcorr: np.ndarray | None,
    n_bootstraps: int,
    max_n: int,
    lower_rep: float,
    seed: int,
    num_cores: int,
) -> pl.DataFrame:
    """Run bootstrap fitting in parallel and aggregate results."""
    nreps = len(reps)

    # Build all combinations of replicates (length >= 2), capped at 500
    idx_list = []
    for k in range(nreps, 1, -1):
        for combo in combinations(range(nreps), k):
            idx_list.append(list(combo))
    if len(idx_list) > 500:
        idx_list = idx_list[:500]

    # Prepare arrays for workers (pass only what's needed to avoid large copies)
    fitness_cols = [f"fitness{E}" for E in reps]
    cbe_cols = [f"cbe{E}" for E in reps]
    count_cols_in = [f"count_e{E}_s0" for E in reps]
    count_cols_out = [f"count_e{E}_s1" for E in reps]

    usable = work.filter(
        pl.col("input_above_threshold") & pl.col("all_reads") & ~pl.col("WT").fill_null(False)
    )

    F_arr = usable.select(fitness_cols).to_numpy().astype(float)
    E_arr = usable.select(cbe_cols).to_numpy().astype(float)
    Cin_arr = usable.select(count_cols_in).to_numpy().astype(float)
    Cout_arr = usable.select(count_cols_out).to_numpy().astype(float)
    Dw_arr = usable["error_model_weighting"].to_numpy().astype(float)

    rng = np.random.default_rng(seed)
    bootstrap_seeds = rng.integers(0, 2**31, size=n_bootstraps)

    params_list = []

    if num_cores > 1:
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            futures = {
                pool.submit(
                    _bootstrap_worker,
                    F_arr, E_arr, Cin_arr, Cout_arr, Dw_arr,
                    idx_list, nreps, max_n, lower_rep, Fcorr,
                    int(bootstrap_seeds[m]),
                ): m
                for m in range(n_bootstraps)
            }
            for future in as_completed(futures):
                try:
                    params_list.append(future.result())
                except Exception as exc:
                    logger.warning("Bootstrap failed: %s", exc)
                    params_list.append(np.full(3 * nreps, np.nan))
    else:
        for m in range(n_bootstraps):
            p = _bootstrap_worker(
                F_arr, E_arr, Cin_arr, Cout_arr, Dw_arr,
                idx_list, nreps, max_n, lower_rep, Fcorr,
                int(bootstrap_seeds[m]),
            )
            params_list.append(p)

    params = np.vstack(params_list)  # (n_bootstraps, 3*nreps)

    # Build error model DataFrame
    rows = []
    for i, param_name in enumerate(["input"] * nreps + ["output"] * nreps + ["reperror"] * nreps):
        rep_idx = i % nreps
        col_values = params[:, i]
        rows.append({
            "parameter": param_name,
            "rep": reps[rep_idx],
            "mean_value": float(np.nanmean(col_values)),
            "CI90_lower": float(np.nanpercentile(col_values, 10)),
            "CI90_upper": float(np.nanpercentile(col_values, 90)),
            "ensemble": int((~np.isnan(params[:, 0])).sum()),
        })

    return pl.DataFrame(rows)


def _bootstrap_worker(
    F_arr: np.ndarray,
    E_arr: np.ndarray,
    Cin_arr: np.ndarray,
    Cout_arr: np.ndarray,
    Dw_arr: np.ndarray,
    idx_list: list[list[int]],
    nreps: int,
    max_n: int,
    lower_rep: float,
    Fcorr: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Fit one bootstrap iteration of the error model.

    Mirrors: dimsum__fit_error_model_bootstrap.R

    The model fitted:
      V_data ~ Σ_j BV_j * (p[j]*C_in[:,j] + p[nreps+j]*C_out[:,j] + p[2*nreps+j]) / NRT

    where C_in = 1/count_in, C_out = 1/count_out.
    Weighted: 1 / (Ew * Dw).
    """
    rng = np.random.default_rng(seed)
    n_data = len(F_arr)
    sample_size = min(n_data, max_n)
    idx = rng.choice(n_data, size=sample_size, replace=True)

    Fs = F_arr[idx]       # (sample, nreps)
    Es = E_arr[idx]       # (sample, nreps)
    Cins = (1.0 / np.where(Cin_arr[idx] > 0, Cin_arr[idx], np.nan))  # 1/count_in
    Couts = (1.0 / np.where(Cout_arr[idx] > 0, Cout_arr[idx], np.nan))
    Dws = Dw_arr[idx]

    # Build combo-stacked arrays
    F_stack_parts = []
    Cin_stack_parts = []
    Cout_stack_parts = []
    NR_parts = []

    for combo in idx_list:
        F_part = np.full((sample_size, nreps), np.nan)
        Cin_part = np.zeros((sample_size, nreps))
        Cout_part = np.zeros((sample_size, nreps))
        for ci, ri in enumerate(combo):
            F_part[:, ri] = Fs[:, ri]
            Cin_part[:, ri] = Cins[:, ri]
            Cout_part[:, ri] = Couts[:, ri]
        F_stack_parts.append(F_part)
        Cin_stack_parts.append(Cin_part)
        Cout_stack_parts.append(Cout_part)
        NR_parts.append(np.full(sample_size, len(combo)))

    F_stack = np.concatenate(F_stack_parts, axis=0)   # (N_total, nreps)
    C_in = np.concatenate(Cin_stack_parts, axis=0)    # (N_total, nreps)
    C_out = np.concatenate(Cout_stack_parts, axis=0)  # (N_total, nreps)
    NRT = np.concatenate(NR_parts, axis=0)             # (N_total,)
    E_stack = np.tile(Es, (len(idx_list), 1))         # (N_total, nreps)
    Dw_stack = np.tile(Dws, len(idx_list))             # (N_total,)

    # Observed variance
    V_data = np.nanvar(F_stack, axis=1, ddof=1)       # (N_total,)

    # Binary presence variable (1 if not nan in this combo)
    BV = (~np.isnan(F_stack)).astype(float)            # (N_total, nreps)

    # Weights: 1 / (Ew * Dw)
    Ew = np.nanmean(E_stack, axis=1) ** 2              # (N_total,)
    weights = 1.0 / (np.where(Ew > 0, Ew, np.nan) * np.where(Dw_stack > 0, Dw_stack, np.nan))
    valid = np.isfinite(V_data) & np.isfinite(weights)

    if valid.sum() < 3 * nreps + 1:
        return np.full(3 * nreps, np.nan)

    # Correct count terms for normalisation factors (Fcorr)
    if Fcorr is not None:
        C_in = C_in * Fcorr          # broadcast (N_total, nreps) * (nreps,)
        C_out = C_out * Fcorr

    # Replace nan count terms with 0 for fitting
    C_in = np.nan_to_num(C_in, nan=0.0)
    C_out = np.nan_to_num(C_out, nan=0.0)

    V_obs = V_data[valid]
    BV_v = BV[valid]
    C_in_v = C_in[valid]
    C_out_v = C_out[valid]
    NRT_v = NRT[valid]
    W_v = weights[valid]

    def residuals(p: np.ndarray) -> np.ndarray:
        # p = [p_in_0, .., p_in_k, p_out_0, .., p_out_k, p_rep_0, .., p_rep_k]
        p_in = p[:nreps]
        p_out = p[nreps:2*nreps]
        p_rep = p[2*nreps:]
        V_pred = np.sum(
            BV_v * (
                p_in * C_in_v + p_out * C_out_v + p_rep
            ),
            axis=1,
        ) / NRT_v
        return np.sqrt(W_v) * (V_obs - V_pred)

    # Bounds: multiplicative >= 1, additive >= lower_rep
    lower = np.concatenate([np.ones(2*nreps), np.full(nreps, lower_rep)])
    upper = np.full(3*nreps, np.inf)

    # Multi-start to avoid the 20-retry loop in R
    best_cost = np.inf
    best_p = np.full(3 * nreps, np.nan)

    rng_starts = np.random.default_rng(seed + 42)
    for _ in range(3):
        p0 = np.concatenate([
            np.ones(2*nreps) + rng_starts.exponential(0.5, 2*nreps),
            np.full(nreps, 0.01) * 10 ** rng_starts.standard_normal(nreps),
        ])
        p0 = np.clip(p0, lower, None)
        try:
            result = least_squares(
                residuals,
                p0,
                bounds=(lower, upper),
                method="trf",
                xtol=1e-8,
                ftol=1e-8,
            )
            if result.cost < best_cost and result.success:
                best_cost = result.cost
                best_p = result.x
        except Exception:
            continue

    return best_p


# ---------------------------------------------------------------------------
# Per-variant sigma from fitted error model
# ---------------------------------------------------------------------------

def compute_sigma(
    df: pl.DataFrame,
    replicates: list[int],
    error_model_df: pl.DataFrame | None,
    norm_model_df: pl.DataFrame | None,
    fitness_normalise: bool,
    fitness_error_model: bool,
) -> pl.DataFrame:
    """Compute per-variant per-replicate sigma (fitness uncertainty).

    Mirrors: dimsum__calculate_fitness.R:58-84 and dimsum__error_model.R:282-318.

    Adds columns ``sigma{E}_uncorr`` for each replicate E.
    """
    reps = replicates

    if fitness_normalise and norm_model_df is not None:
        scale_vals = {E: float(norm_model_df[f"scale_{E}"][0]) for E in reps}
    else:
        scale_vals = {E: 1.0 for E in reps}

    for E in reps:
        s0 = f"count_e{E}_s0"
        s1 = f"count_e{E}_s1"
        Corr = abs(scale_vals[E])

        if fitness_error_model and error_model_df is not None:
            em = error_model_df
            p_in = float(em.filter(
                (pl.col("parameter") == "input") & (pl.col("rep") == E)
            )["mean_value"][0])
            p_out = float(em.filter(
                (pl.col("parameter") == "output") & (pl.col("rep") == E)
            )["mean_value"][0])
            p_rep = float(em.filter(
                (pl.col("parameter") == "reperror") & (pl.col("rep") == E)
            )["mean_value"][0])

            df = df.with_columns(
                (
                    (
                        Corr * (
                            p_in / pl.col(s0).cast(pl.Float64) +
                            p_out / pl.col(s1).cast(pl.Float64)
                        ) + p_rep
                    ).sqrt()
                ).alias(f"sigma{E}_uncorr")
            )
        else:
            # Count-based error only (Poisson)
            wt_s0 = float(df.filter(pl.col("WT") & pl.col("error_model"))[s0].drop_nulls()[0])
            wt_s1 = float(df.filter(pl.col("WT") & pl.col("error_model"))[s1].drop_nulls()[0])
            wt_cbe = 1.0 / wt_s1 + 1.0 / wt_s0

            df = df.with_columns(
                (
                    (
                        1.0 / pl.col(s1).cast(pl.Float64) +
                        1.0 / pl.col(s0).cast(pl.Float64) +
                        wt_cbe
                    ).sqrt()
                ).alias(f"sigma{E}_uncorr")
            )

        # Set sigma to null where fitness is null
        df = df.with_columns(
            pl.when(pl.col(f"fitness{E}_uncorr").is_null())
            .then(None)
            .otherwise(pl.col(f"sigma{E}_uncorr"))
            .alias(f"sigma{E}_uncorr")
        )

    return df
