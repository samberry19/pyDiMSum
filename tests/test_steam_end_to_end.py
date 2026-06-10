"""End-to-end integration test: compare pyDiMSum output to R DiMSum reference.

The reference files in tests/data/reference/ were produced by running:
  R DiMSum 1.4 on the Toy demo count file with default STEAM settings
  (fitnessNormalise=TRUE, fitnessErrorModel=TRUE, numCores=1).

Tolerances:
  - Fitness values: absolute ≤ 0.05 (different optimizer + normalization)
  - Sigma values: absolute ≤ 0.05 (different error model bootstrap RNG/algorithm)
  - Row counts: must match exactly
  - Columns: must include all expected output columns
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

DATA_DIR = Path(__file__).parent / "data"
REF_DIR  = DATA_DIR / "reference"

WT_SEQ = "GGTAATAGCAGAGGGGGTGGAGCTGGTTTGGGAAACAATCAAGGTAGTAATATGGGTGGTGGGATGAACTTTGGTGCGTTCAGCATTAATCCAGCCATGATGGCTGCCGCCCAGGCAGCACTACAG"

# Fitness agreement tolerances
FITNESS_ATOL = 0.05   # absolute tolerance for per-variant fitness
SIGMA_ATOL   = 0.05   # absolute tolerance for per-variant sigma


# ---------------------------------------------------------------------------
# Fixture: run the pipeline once and return output paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory) -> Path:
    """Run pyDiMSum STEAM pipeline on the Toy demo and return output directory."""
    from pydimsum.config import RunConfig
    from pydimsum.pipeline import run_pipeline

    out_root = tmp_path_factory.mktemp("e2e_output")
    config = RunConfig(
        experiment_design_path=DATA_DIR / "experimentDesign_Toy.txt",
        wildtype_sequence=WT_SEQ,
        count_path=DATA_DIR / "countFile_Toy.txt",
        output_path=out_root,
        project_name="PyDiMSum_e2e",
        fitness_normalise=True,
        fitness_error_model=True,
        num_cores=1,
    )
    run_pipeline(config)
    return out_root / "PyDiMSum_e2e"


# ---------------------------------------------------------------------------
# Helper: load both Python and R output tables
# ---------------------------------------------------------------------------

def _load_py(output_dir: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(
        str(output_dir / f"fitness_{name}.txt"),
        separator="\t",
        null_values=["NA"],
    )


def _load_r(name: str) -> pl.DataFrame:
    return pl.read_csv(
        str(REF_DIR / f"fitness_{name}.txt"),
        separator=" ",
        null_values=["NA"],
    )


def _join_on(py_df, r_df, keys: list[str], fitness_col="fitness", sigma_col="sigma"):
    """Inner-join py vs R on keys and return comparison arrays."""
    py_sel = py_df.select(keys + [fitness_col, sigma_col]).rename(
        {fitness_col: "py_f", sigma_col: "py_s"}
    )
    r_sel = r_df.select(keys + [fitness_col, sigma_col]).rename(
        {fitness_col: "r_f", sigma_col: "r_s"}
    )
    joined = py_sel.join(r_sel, on=keys, how="inner")
    valid = joined.filter(
        pl.col("py_f").is_not_null() & pl.col("r_f").is_not_null() &
        ~pl.col("py_f").is_nan()
    )
    return (
        valid["py_f"].to_numpy(),
        valid["r_f"].to_numpy(),
        valid["py_s"].to_numpy(),
        valid["r_s"].to_numpy(),
        len(joined),
    )


# ---------------------------------------------------------------------------
# Singles tests
# ---------------------------------------------------------------------------

class TestSingles:
    def test_row_count_matches_reference(self, pipeline_output):
        py = _load_py(pipeline_output, "singles")
        r  = _load_r("singles")
        assert len(py) == len(r), (
            f"Singles row count mismatch: Python={len(py)}, R={len(r)}"
        )

    def test_columns_present(self, pipeline_output):
        py = _load_py(pipeline_output, "singles")
        expected = ["Pos", "WT_AA", "Mut", "nt_seq", "aa_seq",
                    "Nham_nt", "Nham_aa", "Nmut_codons",
                    "STOP", "STOP_readthrough", "mean_count", "fitness", "sigma"]
        for col in expected:
            assert col in py.columns, f"Missing column: {col}"

    def test_fitness_close_to_reference(self, pipeline_output):
        py = _load_py(pipeline_output, "singles")
        r  = _load_r("singles")
        py_f, r_f, _, _, n_joined = _join_on(py, r, ["Pos", "Mut"])
        assert n_joined == len(py), (
            f"Not all singles variants joined: {n_joined}/{len(py)}"
        )
        max_diff = float(np.max(np.abs(py_f - r_f)))
        assert max_diff < FITNESS_ATOL, (
            f"Singles fitness max abs diff {max_diff:.4f} ≥ {FITNESS_ATOL}"
        )

    def test_sigma_close_to_reference(self, pipeline_output):
        py = _load_py(pipeline_output, "singles")
        r  = _load_r("singles")
        _, _, py_s, r_s, _ = _join_on(py, r, ["Pos", "Mut"])
        # Keep non-null sigma pairs
        valid = np.isfinite(py_s) & np.isfinite(r_s)
        if valid.sum() == 0:
            pytest.skip("No valid sigma pairs to compare")
        max_diff = float(np.max(np.abs(py_s[valid] - r_s[valid])))
        assert max_diff < SIGMA_ATOL, (
            f"Singles sigma max abs diff {max_diff:.4f} ≥ {SIGMA_ATOL}"
        )

    def test_no_nan_fitness(self, pipeline_output):
        py = _load_py(pipeline_output, "singles")
        nan_count = py["fitness"].is_nan().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in Python singles fitness"

    def test_nt_seq_null_for_nonsynonymous(self, pipeline_output):
        """For coding AA-level singles (Nham_aa==1), nt_seq should be null."""
        py = _load_py(pipeline_output, "singles")
        nonsyn = py.filter(pl.col("Nham_aa").fill_null(0) > 0)
        n_non_null = nonsyn["nt_seq"].is_not_null().sum()
        assert n_non_null == 0, (
            f"{n_non_null} nonsynonymous singles have non-null nt_seq "
            "(expected NA, matching R output)"
        )


# ---------------------------------------------------------------------------
# Doubles tests
# ---------------------------------------------------------------------------

class TestDoubles:
    def test_row_count_matches_reference(self, pipeline_output):
        py = _load_py(pipeline_output, "doubles")
        r  = _load_r("doubles")
        assert len(py) == len(r), (
            f"Doubles row count mismatch: Python={len(py)}, R={len(r)}"
        )

    def test_columns_present(self, pipeline_output):
        py = _load_py(pipeline_output, "doubles")
        expected = ["Pos1", "Pos2", "WT_AA1", "WT_AA2", "Mut1", "Mut2",
                    "fitness1", "sigma1", "fitness2", "sigma2",
                    "fitness_uncorr", "sigma_uncorr"]
        for col in expected:
            assert col in py.columns, f"Missing column: {col}"

    def test_fitness_close_to_reference(self, pipeline_output):
        py = _load_py(pipeline_output, "doubles")
        r  = _load_r("doubles")
        py_f, r_f, _, _, n_joined = _join_on(
            py, r, ["Pos1", "Pos2", "Mut1", "Mut2"],
            fitness_col="fitness_uncorr", sigma_col="sigma_uncorr",
        )
        assert n_joined == len(py), (
            f"Not all doubles joined: {n_joined}/{len(py)}"
        )
        valid = np.isfinite(py_f) & np.isfinite(r_f)
        if valid.sum() == 0:
            pytest.skip("No valid double-fitness pairs")
        max_diff = float(np.max(np.abs(py_f[valid] - r_f[valid])))
        assert max_diff < FITNESS_ATOL, (
            f"Doubles fitness max abs diff {max_diff:.4f} ≥ {FITNESS_ATOL}"
        )

    def test_no_nan_fitness(self, pipeline_output):
        py = _load_py(pipeline_output, "doubles")
        # fitness_uncorr may be null for low-count variants; NaN is a bug
        if "fitness_uncorr" in py.columns:
            nan_count = py["fitness_uncorr"].is_nan().sum()
            assert nan_count == 0, f"Found {nan_count} NaN in doubles fitness_uncorr"


# ---------------------------------------------------------------------------
# Wildtype test
# ---------------------------------------------------------------------------

class TestWildtype:
    def test_wildtype_row_present(self, pipeline_output):
        py = _load_py(pipeline_output, "wildtype")
        r  = _load_r("wildtype")
        assert len(py) == 1, f"Expected 1 WT row, got {len(py)}"
        assert len(r) == 1

    def test_wildtype_fitness_close(self, pipeline_output):
        py = _load_py(pipeline_output, "wildtype")
        r  = _load_r("wildtype")
        # WT fitness should be near 0 (it's the reference)
        py_f = float(py["fitness"][0])
        r_f  = float(r["fitness"][0])
        assert abs(py_f) < FITNESS_ATOL, f"WT fitness {py_f:.4f} far from 0"
        assert abs(r_f)  < FITNESS_ATOL, f"R WT fitness {r_f:.4f} far from 0"
        assert abs(py_f - r_f) < FITNESS_ATOL, (
            f"WT fitness mismatch: Python={py_f:.4f}, R={r_f:.4f}"
        )

    def test_wildtype_nt_seq_present(self, pipeline_output):
        """WT nt_seq should be the lowercase wildtype nucleotide sequence."""
        py = _load_py(pipeline_output, "wildtype")
        wt_nt = py["nt_seq"][0]
        assert wt_nt is not None, "WT nt_seq is null"
        assert wt_nt.lower() == WT_SEQ.lower(), (
            f"WT nt_seq mismatch.\n  Got:      {wt_nt}\n  Expected: {WT_SEQ.lower()}"
        )


# ---------------------------------------------------------------------------
# Parquet bundle test
# ---------------------------------------------------------------------------

class TestParquetBundle:
    def test_parquet_bundle_exists(self, pipeline_output):
        bundle_dir = pipeline_output / "PyDiMSum_e2e_fitness_replicates"
        assert bundle_dir.exists(), f"Parquet bundle directory not found: {bundle_dir}"
        parquet_files = list(bundle_dir.glob("*.parquet"))
        assert len(parquet_files) >= 2, (
            f"Expected ≥2 parquet files in bundle, found {len(parquet_files)}"
        )

    def test_parquet_all_variants_readable(self, pipeline_output):
        bundle_dir = pipeline_output / "PyDiMSum_e2e_fitness_replicates"
        all_variants_path = bundle_dir / "all_variants.parquet"
        assert all_variants_path.exists()
        df = pl.read_parquet(str(all_variants_path))
        assert len(df) > 0
        assert "fitness" in df.columns
        assert "sigma" in df.columns
