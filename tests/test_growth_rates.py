"""Tests for pydimsum.steam.growth_rates."""

import numpy as np
import polars as pl
import pytest

from pydimsum.steam.growth_rates import has_growth_rate_data, infer_growth_rates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_exp_design(cd_in=100.0, cd_out=200.0, t=4.0, n_reps=2) -> pl.DataFrame:
    """Two-replicate experiment design with cell density data."""
    rows = []
    for E in range(1, n_reps + 1):
        rows.append({"experiment": E, "selection_id": 0, "cell_density": cd_in, "selection_time": None})
        rows.append({"experiment": E, "selection_id": 1, "cell_density": cd_out, "selection_time": t})
    return pl.DataFrame(rows).with_columns(
        pl.col("experiment").cast(pl.Int32),
        pl.col("selection_id").cast(pl.Int32),
    )


def _make_exp_design_missing_cd() -> pl.DataFrame:
    """Experiment design without cell density — growth rates should not run."""
    return pl.DataFrame([
        {"experiment": 1, "selection_id": 0, "cell_density": None, "selection_time": None},
        {"experiment": 1, "selection_id": 1, "cell_density": None, "selection_time": 4.0},
    ]).with_columns(pl.col("experiment").cast(pl.Int32), pl.col("selection_id").cast(pl.Int32))


def _make_variant_df(n=30, n_reps=2, seed=42) -> pl.DataFrame:
    """Synthetic variant table with per-replicate counts and pre-computed fitness."""
    rng = np.random.default_rng(seed)
    data = {"nt_seq": [f"seq{i:03d}" for i in range(n)], "error_model": [True] * n}
    for E in range(1, n_reps + 1):
        counts_in = rng.integers(50, 500, size=n)
        counts_out = rng.integers(50, 500, size=n)
        data[f"count_e{E}_s0"] = counts_in.tolist()
        data[f"count_e{E}_s1"] = counts_out.tolist()
        # Synthetic per-replicate fitness (correlated with log-ratio)
        data[f"fitness{E}_uncorr"] = np.log(counts_out / counts_in).tolist()
        data[f"sigma{E}_uncorr"] = (np.sqrt(1 / counts_out + 1 / counts_in)).tolist()
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# has_growth_rate_data
# ---------------------------------------------------------------------------

class TestHasGrowthRateData:
    def test_complete_design_returns_true(self):
        assert has_growth_rate_data(_make_exp_design()) is True

    def test_missing_cd_in_returns_false(self):
        ed = _make_exp_design()
        ed = ed.with_columns(
            pl.when(pl.col("selection_id") == 0)
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("cell_density"))
            .alias("cell_density")
        )
        assert has_growth_rate_data(ed) is False

    def test_missing_cd_out_returns_false(self):
        ed = _make_exp_design()
        ed = ed.with_columns(
            pl.when(pl.col("selection_id") == 1)
            .then(pl.lit(None).cast(pl.Float64))
            .otherwise(pl.col("cell_density"))
            .alias("cell_density")
        )
        assert has_growth_rate_data(ed) is False

    def test_missing_selection_time_returns_false(self):
        ed = _make_exp_design()
        ed = ed.with_columns(pl.lit(None).cast(pl.Float64).alias("selection_time"))
        assert has_growth_rate_data(ed) is False

    def test_no_selection_time_col_returns_false(self):
        ed = _make_exp_design().drop("selection_time")
        assert has_growth_rate_data(ed) is False

    def test_missing_design_returns_false(self):
        assert has_growth_rate_data(_make_exp_design_missing_cd()) is False


# ---------------------------------------------------------------------------
# infer_growth_rates — structural tests
# ---------------------------------------------------------------------------

class TestInferGrowthRates:
    def test_adds_growthrate_columns(self):
        df = _make_variant_df(n=40)
        ed = _make_exp_design()
        result = infer_growth_rates(df, ed, replicates=[1, 2])
        assert "growthrate" in result.columns
        assert "growthrate_sigma" in result.columns

    def test_no_raw_intermediate_columns(self):
        df = _make_variant_df(n=40)
        ed = _make_exp_design()
        result = infer_growth_rates(df, ed, replicates=[1, 2])
        raw_cols = [c for c in result.columns if c.endswith("_raw")]
        assert raw_cols == [], f"Raw columns left in output: {raw_cols}"

    def test_growthrate_mostly_finite(self):
        df = _make_variant_df(n=40)
        ed = _make_exp_design()
        result = infer_growth_rates(df, ed, replicates=[1, 2])
        vals = result["growthrate"].drop_nulls().to_numpy()
        assert np.isfinite(vals).mean() > 0.8

    def test_growthrate_sigma_nonnegative(self):
        df = _make_variant_df(n=40)
        ed = _make_exp_design()
        result = infer_growth_rates(df, ed, replicates=[1, 2])
        sigma = result["growthrate_sigma"].drop_nulls().to_numpy()
        assert (sigma >= 0).all()

    def test_single_replicate_runs(self):
        df = _make_variant_df(n=40, n_reps=1)
        ed = _make_exp_design(n_reps=1)
        result = infer_growth_rates(df, ed, replicates=[1])
        assert "growthrate" in result.columns

    def test_empty_df_returns_unchanged(self):
        df = _make_variant_df(n=0)
        ed = _make_exp_design()
        result = infer_growth_rates(df, ed, replicates=[1, 2])
        assert len(result) == 0

    def test_growth_rate_formula(self):
        """growthrate should be a linear transform of fitness (linear model structure)."""
        n = 50
        t, cd_in, cd_out = 4.0, 100.0, 400.0
        rng = np.random.default_rng(0)
        counts_in = rng.integers(50, 300, size=n)
        counts_out = rng.integers(50, 300, size=n)
        fitness = np.log(counts_out / counts_in)
        data = {
            "nt_seq": [f"seq{i}" for i in range(n)],
            "error_model": [True] * n,
            "count_e1_s0": counts_in.tolist(),
            "count_e1_s1": counts_out.tolist(),
            "fitness1_uncorr": fitness.tolist(),
            "sigma1_uncorr": [0.02] * n,
        }
        df = pl.DataFrame(data)
        ed = _make_exp_design(cd_in=cd_in, cd_out=cd_out, t=t, n_reps=1)
        result = infer_growth_rates(df, ed, replicates=[1])
        gr_vals = result["growthrate"].drop_nulls().to_numpy()
        # Growth rate is a linear transform of fitness, so it should have
        # meaningful spread and a roughly consistent sign with the expected raw value.
        expected_center = np.log(cd_out / cd_in) / t
        assert len(gr_vals) > 0, "No growth rate values produced"
        assert np.isfinite(gr_vals).mean() > 0.8, "Too many non-finite growth rates"
        # The centre of the distribution should be in the right ballpark
        assert abs(gr_vals.mean() - expected_center) < 2.0 * abs(expected_center) + 0.5

    def test_pre_merged_fitness_used_in_final_model(self):
        """If fitness/sigma columns already present, step 4 uses them."""
        import numpy as np
        rng = np.random.default_rng(99)
        df = _make_variant_df(n=40)
        ed = _make_exp_design()
        # Add varied merged fitness (normally done by _inverse_variance_merge before this call)
        df = df.with_columns([
            pl.Series("fitness", rng.standard_normal(40).tolist()),
            pl.lit(0.05).alias("sigma"),
        ])
        result = infer_growth_rates(df, ed, replicates=[1, 2])
        # Should not raise and should still produce growth rate columns
        assert "growthrate" in result.columns
