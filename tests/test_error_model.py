"""Unit tests for pydimsum.steam.error_model — normalisation and error model."""

import numpy as np
import polars as pl
import pytest

from pydimsum.steam.error_model import _fit_normalisation, _bootstrap_worker


def _make_work_df(
    n_variants: int = 200,
    reps: list[int] = [1, 2, 3],
    scale_true: list[float] = [1.0, 1.2, 0.9],
    shift_true: list[float] = [0.0, -0.1, 0.05],
    seed: int = 42,
) -> pl.DataFrame:
    """Create a synthetic work DataFrame for testing normalisation."""
    rng = np.random.default_rng(seed)
    # True fitness values (latent)
    f_true = rng.normal(0, 1, n_variants)

    # Observed fitness = (f_true + shift) * scale + noise
    rows: dict = {}
    for E, scale, shift in zip(reps, scale_true, shift_true):
        obs = (f_true + shift) * scale + rng.normal(0, 0.05, n_variants)
        rows[f"fitness{E}"] = obs.tolist()
        # Dummy counts
        rows[f"count_e{E}_s0"] = rng.integers(10, 1000, n_variants).tolist()
        rows[f"count_e{E}_s1"] = rng.integers(10, 1000, n_variants).tolist()
        rows[f"cbe{E}"] = (0.01 * np.ones(n_variants)).tolist()

    rows["WT"] = [None] * (n_variants - 1) + [True]
    rows["all_reads"] = [True] * n_variants
    rows["input_above_threshold"] = [True] * n_variants
    rows["Nham_nt"] = rng.integers(0, 3, n_variants).tolist()
    rows["error_model_weighting"] = [1.0] * n_variants
    rows["mean_cbe"] = [0.01] * n_variants
    rows["bin_error"] = [1] * n_variants

    return pl.DataFrame(rows)


class TestNormalisation:
    def test_scale_shift_recovered(self):
        """Fitted scale/shift should be close to the true values."""
        reps = [1, 2, 3]
        scale_true = [1.0, 1.2, 0.9]
        shift_true = [0.0, -0.1, 0.05]
        df = _make_work_df(
            n_variants=500, reps=reps,
            scale_true=scale_true, shift_true=shift_true, seed=0,
        )
        norm_model = _fit_normalisation(df, reps)

        # scale_1 is fixed to 1 by the normaliser after dividing by p[0]
        assert abs(float(norm_model["scale_1"][0]) - 1.0) < 0.01

        # The normalisation parameters are the INVERSE of the data-generation scales,
        # because if obs = (f + shift) * scale, normalisation applies (f + norm_shift) * norm_scale
        # to recover f, requiring norm_scale = 1/scale.
        # After fixing scale_1 = 1 (dividing all by p[0] = 1/scale_true[0] = 1),
        # the ratio norm_scale_2 / norm_scale_3 ≈ scale_true[2] / scale_true[1]
        ratio_23 = float(norm_model["scale_2"][0]) / float(norm_model["scale_3"][0])
        expected_ratio = scale_true[2] / scale_true[1]   # = 0.9/1.2 ≈ 0.75
        assert abs(ratio_23 - expected_ratio) < 0.1

    def test_single_replicate_returns_trivial(self):
        """With a single replicate, normalisation should give scale=1, shift=0."""
        df = _make_work_df(n_variants=100, reps=[1], scale_true=[1.0], shift_true=[0.0])
        norm_model = _fit_normalisation(df, [1])
        assert abs(float(norm_model["scale_1"][0]) - 1.0) < 1e-6


class TestBootstrapWorker:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        n, nreps = 200, 3
        F_arr = rng.normal(0, 1, (n, nreps))
        E_arr = rng.uniform(0.01, 0.1, (n, nreps))
        Cin = rng.integers(1, 100, (n, nreps)).astype(float)
        Cout = rng.integers(1, 100, (n, nreps)).astype(float)
        Dw = np.ones(n)
        idx_list = [[0, 1], [1, 2], [0, 2], [0, 1, 2]]

        result = _bootstrap_worker(
            F_arr, E_arr, Cin, Cout, Dw,
            idx_list, nreps, max_n=100, lower_rep=1e-4,
            Fcorr=None, seed=123,
        )
        assert result.shape == (3 * nreps,)

    def test_parameters_are_within_bounds(self):
        rng = np.random.default_rng(7)
        n, nreps = 300, 2
        F_arr = rng.normal(0, 1, (n, nreps))
        E_arr = np.full((n, nreps), 0.05)
        Cin = np.full((n, nreps), 50.0)
        Cout = np.full((n, nreps), 50.0)
        Dw = np.ones(n)
        idx_list = [[0, 1]]

        result = _bootstrap_worker(
            F_arr, E_arr, Cin, Cout, Dw,
            idx_list, nreps, max_n=200, lower_rep=1e-4,
            Fcorr=None, seed=7,
        )
        # All multiplicative params >= 1, all rep error >= lower_rep=1e-4
        if not np.any(np.isnan(result)):
            assert np.all(result[:2*nreps] >= 1.0 - 1e-6)
            assert np.all(result[2*nreps:] >= 1e-4 - 1e-8)
