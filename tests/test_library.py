"""Tests for enrichment / library mode (pydimsum/steam/library.py).

All tests use synthetic in-memory fixtures — no real FASTQ or count files needed.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal RunConfig and ExperimentDesign-like objects
# ---------------------------------------------------------------------------

def _make_exp_design_tsv(tmp_path: Path, n_reps: int = 2) -> Path:
    """Write a minimal experiment design TSV with n_reps replicates."""
    rows = []
    for e in range(1, n_reps + 1):
        rows.append(f"input{e}\t{e}\t0\t\t\t\t")
        rows.append(f"output{e}\t{e}\t1\t1\t\t\t")
    header = "sample_name\texperiment_replicate\tselection_id\tselection_replicate\ttechnical_replicate\tpair1\tpair2"
    content = header + "\n" + "\n".join(rows) + "\n"
    p = tmp_path / "experimentDesign.txt"
    p.write_text(content)
    return p


def _make_config(tmp_path: Path, n_reps: int = 2, **kwargs):
    """Return a RunConfig in enrichment mode."""
    from pydimsum.config import RunConfig
    exp_path = _make_exp_design_tsv(tmp_path, n_reps)
    defaults = dict(
        experiment_design_path=exp_path,
        wildtype_sequence="",
        enrichment_mode=True,
        enrichment_normalise="none",
        output_path=tmp_path,
        project_name="test_enrichment",
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)


def _count_table(seqs: list[str], counts_s0: list[list[int]], counts_s1: list[list[int]]) -> pl.DataFrame:
    """Build a synthetic wide count table for n_reps replicates.

    counts_s0[i] = list of input counts for each sequence at replicate i+1.
    counts_s1[i] = list of output counts for each sequence at replicate i+1.
    """
    n_reps = len(counts_s0)
    data: dict[str, list] = {"nt_seq": seqs}
    for e in range(1, n_reps + 1):
        data[f"count_e{e}_s0"] = counts_s0[e - 1]
        data[f"count_e{e}_s1"] = counts_s1[e - 1]
    return pl.DataFrame(data).with_columns([
        pl.col(c).cast(pl.UInt32)
        for c in data if c != "nt_seq"
    ])


# ---------------------------------------------------------------------------
# Tests: Config validation
# ---------------------------------------------------------------------------

class TestEnrichmentConfig:
    def test_empty_wt_allowed_in_enrichment_mode(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.enrichment_mode is True
        assert cfg.wildtype_sequence == ""

    def test_wt_still_required_in_mutation_mode(self, tmp_path):
        from pydimsum.config import RunConfig
        exp_path = _make_exp_design_tsv(tmp_path)
        with pytest.raises(ValueError, match="wildtypeSequence is required"):
            RunConfig(
                experiment_design_path=exp_path,
                wildtype_sequence="",
                enrichment_mode=False,
            )

    def test_invalid_normalise_raises(self, tmp_path):
        from pydimsum.config import RunConfig
        exp_path = _make_exp_design_tsv(tmp_path)
        with pytest.raises(ValueError, match="enrichment_normalise"):
            RunConfig(
                experiment_design_path=exp_path,
                wildtype_sequence="",
                enrichment_mode=True,
                enrichment_normalise="bogus",
            )

    def test_reference_requires_id(self, tmp_path):
        from pydimsum.config import RunConfig
        exp_path = _make_exp_design_tsv(tmp_path)
        with pytest.raises(ValueError, match="enrichment_reference_id is required"):
            RunConfig(
                experiment_design_path=exp_path,
                wildtype_sequence="",
                enrichment_mode=True,
                enrichment_normalise="reference",
                enrichment_reference_id=None,
            )

    def test_sequence_type_noncoding_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.sequence_type_resolved == "noncoding"

    def test_sequence_type_coding_respected(self, tmp_path):
        cfg = _make_config(tmp_path, sequence_type="coding")
        assert cfg.sequence_type_resolved == "coding"


# ---------------------------------------------------------------------------
# Tests: process_library_variants
# ---------------------------------------------------------------------------

class TestProcessLibraryVariants:
    def test_all_sequences_retained(self, tmp_path):
        """No sequences should be filtered (no WT/Hamming logic)."""
        from pydimsum.steam.library import process_library_variants
        seqs = ["acgt", "tttt", "gcgc", "aaaa"]
        df = _count_table(
            seqs,
            counts_s0=[[10, 20, 15, 5]],
            counts_s1=[[30, 10, 60, 2]],
        )
        cfg = _make_config(tmp_path, n_reps=1)
        out = process_library_variants(df, cfg)
        assert len(out) == 4
        assert set(out["nt_seq"].to_list()) == set(seqs)

    def test_annotations_added(self, tmp_path):
        from pydimsum.steam.library import process_library_variants
        seqs = ["acgtacgt", "tttttttt"]
        df = _count_table(seqs, [[10, 20]], [[30, 10]])
        cfg = _make_config(tmp_path, n_reps=1)
        out = process_library_variants(df, cfg)
        assert "length" in out.columns
        assert list(out["length"].to_list()) == [8, 8]
        assert "merge_seq" in out.columns
        assert "WT" in out.columns
        assert out["WT"].to_list() == [False, False]
        assert "error_model" in out.columns
        assert out["error_model"].to_list() == [True, True]
        assert "Nham_nt" in out.columns
        assert out["Nham_nt"].to_list() == [0, 0]
        assert "is_reference" in out.columns
        assert "is_spikein" in out.columns

    def test_aa_seq_added_in_coding_mode(self, tmp_path):
        from pydimsum.steam.library import process_library_variants
        # Simple in-frame triplets
        seqs = ["atgatg", "atgttt"]  # Met-Met, Met-Phe
        df = _count_table(seqs, [[10, 20]], [[30, 10]])
        cfg = _make_config(tmp_path, n_reps=1, sequence_type="coding")
        out = process_library_variants(df, cfg)
        assert "aa_seq" in out.columns
        aa = out["aa_seq"].to_list()
        assert aa[0] == "MM"
        assert aa[1] == "MF"

    def test_no_aa_seq_in_noncoding_mode(self, tmp_path):
        from pydimsum.steam.library import process_library_variants
        seqs = ["acgt", "tttt"]
        df = _count_table(seqs, [[10, 20]], [[30, 10]])
        cfg = _make_config(tmp_path, n_reps=1)
        out = process_library_variants(df, cfg)
        assert "aa_seq" not in out.columns

    def test_reference_flagged(self, tmp_path):
        from pydimsum.steam.library import process_library_variants
        seqs = ["aaaa", "tttt", "cccc"]
        df = _count_table(seqs, [[10, 20, 15]], [[30, 10, 45]])
        cfg = _make_config(
            tmp_path, n_reps=1,
            enrichment_normalise="reference",
            enrichment_reference_id="tttt",
        )
        out = process_library_variants(df, cfg)
        is_ref = dict(zip(out["nt_seq"].to_list(), out["is_reference"].to_list()))
        assert is_ref["tttt"] is True
        assert is_ref["aaaa"] is False
        assert is_ref["cccc"] is False

    def test_spikein_flagged(self, tmp_path):
        from pydimsum.steam.library import process_library_variants
        seqs = ["aaaa", "tttt", "cccc"]
        df = _count_table(seqs, [[10, 20, 15]], [[30, 10, 45]])
        cfg = _make_config(
            tmp_path, n_reps=1,
            enrichment_normalise="spikein",
            enrichment_spikein_ids="tttt,cccc",
        )
        out = process_library_variants(df, cfg)
        is_si = dict(zip(out["nt_seq"].to_list(), out["is_spikein"].to_list()))
        assert is_si["tttt"] is True
        assert is_si["cccc"] is True
        assert is_si["aaaa"] is False


# ---------------------------------------------------------------------------
# Tests: calculate_enrichment — raw log(out/in)
# ---------------------------------------------------------------------------

class TestCalculateEnrichment:
    def _make_annotated(self, seqs, counts_s0, counts_s1, tmp_path, n_reps=1, **cfg_kwargs):
        """Build a count table and run process_library_variants."""
        from pydimsum.steam.library import process_library_variants
        df = _count_table(seqs, counts_s0, counts_s1)
        cfg = _make_config(tmp_path, n_reps=n_reps, **cfg_kwargs)
        return process_library_variants(df, cfg), cfg

    def test_raw_log_ratio(self, tmp_path):
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt"]
        # Rep 1: in=10/out=20 → log(2); in=100/out=25 → log(0.25)
        df, cfg = self._make_annotated(
            seqs, counts_s0=[[10, 100]], counts_s1=[[20, 25]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="none",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        vals = dict(zip(out["nt_seq"].to_list(), out["enrichment1_uncorr"].to_list()))
        assert abs(vals["aaaa"] - math.log(2.0)) < 1e-9
        assert abs(vals["tttt"] - math.log(0.25)) < 1e-9

    def test_zero_count_drops_row(self, tmp_path):
        """A sequence with zero input count has null enrichment in all reps → dropped."""
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt"]
        # seq aaaa: in=0 → null enrichment → row dropped (no valid rep)
        df, cfg = self._make_annotated(
            seqs, counts_s0=[[0, 10]], counts_s1=[[5, 20]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="none",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        assert "aaaa" not in out["nt_seq"].to_list()
        assert "tttt" in out["nt_seq"].to_list()

    def test_sigma_poisson(self, tmp_path):
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa"]
        # in=10, out=20 → sigma = sqrt(1/20 + 1/10)
        df, cfg = self._make_annotated(
            seqs, counts_s0=[[10]], counts_s1=[[20]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="none",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        sigma = out["sigma1_uncorr"].to_list()[0]
        expected = math.sqrt(1.0 / 20 + 1.0 / 10)
        assert abs(sigma - expected) < 1e-9

    def test_normalise_none(self, tmp_path):
        """enrichment_normalise='none' leaves values unchanged."""
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt", "cccc"]
        df, cfg = self._make_annotated(
            seqs,
            counts_s0=[[10, 10, 10]],
            counts_s1=[[20, 10, 30]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="none",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        vals = sorted(out["enrichment1_uncorr"].drop_nulls().to_list())
        raw = sorted([math.log(20 / 10), math.log(10 / 10), math.log(30 / 10)])
        for a, b in zip(vals, raw):
            assert abs(a - b) < 1e-9

    def test_normalise_median(self, tmp_path):
        """enrichment_normalise='median' should center the median near zero."""
        from pydimsum.steam.library import calculate_enrichment
        # 3 seqs: log ratios will be log(2), log(1), log(3) = 0.693, 0.0, 1.099
        # median raw = log(2) = 0.693 → after centering: -0.693, -0.693, 0.406
        seqs = ["aaaa", "tttt", "cccc"]
        df, cfg = self._make_annotated(
            seqs,
            counts_s0=[[10, 10, 10]],
            counts_s1=[[20, 10, 30]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="median",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        centered = sorted(out["enrichment1_uncorr"].drop_nulls().to_list())
        # Median of centered values should be ~0
        import statistics
        assert abs(statistics.median(centered)) < 1e-9

    def test_normalise_total(self, tmp_path):
        """enrichment_normalise='total' uses summed counts."""
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt"]
        # in_total=20, out_total=60 → offset = log(60/20) = log(3)
        df, cfg = self._make_annotated(
            seqs,
            counts_s0=[[10, 10]],
            counts_s1=[[40, 20]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="total",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        vals = dict(zip(out["nt_seq"].to_list(), out["enrichment1_uncorr"].to_list()))
        offset = math.log(60 / 20)
        assert abs(vals["aaaa"] - (math.log(40 / 10) - offset)) < 1e-9
        assert abs(vals["tttt"] - (math.log(20 / 10) - offset)) < 1e-9

    def test_normalise_reference(self, tmp_path):
        """enrichment_normalise='reference' subtracts the reference sequence enrichment."""
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt", "cccc"]
        # ref = "tttt": in=10, out=10 → log(1)=0 → no shift
        df, cfg = self._make_annotated(
            seqs,
            counts_s0=[[10, 10, 10]],
            counts_s1=[[20, 10, 40]],
            tmp_path=tmp_path, n_reps=1,
            enrichment_normalise="reference",
            enrichment_reference_id="tttt",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        vals = dict(zip(out["nt_seq"].to_list(), out["enrichment1_uncorr"].to_list()))
        # offset = log(10/10) = 0, so enrichments are unchanged
        assert abs(vals["aaaa"] - math.log(2.0)) < 1e-9
        assert abs(vals["tttt"] - 0.0) < 1e-9
        assert abs(vals["cccc"] - math.log(4.0)) < 1e-9

    def test_normalise_spikein(self, tmp_path):
        """enrichment_normalise='spikein' uses mean of spike-in enrichments."""
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "spike1", "spike2"]
        # spikes: log(2) and log(4), mean=log(2)+log(4)/2 = (0.693+1.386)/2 = 1.0397
        df, cfg = self._make_annotated(
            seqs,
            counts_s0=[[100, 10, 10]],
            counts_s1=[[200, 20, 40]],
            tmp_path=tmp_path, n_reps=1,
            enrichment_normalise="spikein",
            enrichment_spikein_ids="spike1,spike2",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        vals = dict(zip(out["nt_seq"].to_list(), out["enrichment1_uncorr"].to_list()))
        mean_spikein = (math.log(2.0) + math.log(4.0)) / 2
        assert abs(vals["aaaa"] - (math.log(2.0) - mean_spikein)) < 1e-9

    def test_no_fitness_drops_row(self, tmp_path):
        """Rows with zero counts in all replicates are dropped."""
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt"]
        # aaaa: in=0 and out=0 in the one rep → null enrichment → dropped
        df, cfg = self._make_annotated(
            seqs, counts_s0=[[0, 10]], counts_s1=[[0, 20]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="none",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        assert len(out) == 1
        assert out["nt_seq"].to_list() == ["tttt"]

    def test_mean_count_added(self, tmp_path):
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt"]
        df, cfg = self._make_annotated(
            seqs, counts_s0=[[10, 20]], counts_s1=[[30, 40]],
            tmp_path=tmp_path, n_reps=1, enrichment_normalise="none",
        )
        out = calculate_enrichment(df, cfg, [1], None)
        assert "mean_count" in out.columns

    def test_two_replicate_merging(self, tmp_path):
        """With 2 replicates, enrichment columns exist for both."""
        from pydimsum.steam.library import calculate_enrichment
        seqs = ["aaaa", "tttt"]
        df, cfg = self._make_annotated(
            seqs,
            counts_s0=[[10, 20], [15, 25]],
            counts_s1=[[30, 10], [45, 50]],
            tmp_path=tmp_path, n_reps=2, enrichment_normalise="none",
        )
        out = calculate_enrichment(df, cfg, [1, 2], None)
        assert "enrichment1_uncorr" in out.columns
        assert "enrichment2_uncorr" in out.columns
        assert "sigma1_uncorr" in out.columns
        assert "sigma2_uncorr" in out.columns


# ---------------------------------------------------------------------------
# Tests: write_enrichment_outputs
# ---------------------------------------------------------------------------

class TestWriteEnrichmentOutputs:
    def _run_enrichment(self, tmp_path, n_reps=2):
        from pydimsum.steam.library import (
            calculate_enrichment,
            process_library_variants,
            write_enrichment_outputs,
        )
        seqs = ["aaaa", "tttt", "cccc"]
        counts_s0 = [[10, 20, 15]] * n_reps
        counts_s1 = [[30, 10, 60]] * n_reps
        df = _count_table(seqs, counts_s0, counts_s1)
        cfg = _make_config(tmp_path, n_reps=n_reps, enrichment_normalise="none")
        reps = list(range(1, n_reps + 1))
        lib_df = process_library_variants(df, cfg)
        enr_df = calculate_enrichment(lib_df, cfg, reps, None)
        write_enrichment_outputs(enr_df, reps, cfg)
        return cfg

    def test_output_files_created(self, tmp_path):
        cfg = self._run_enrichment(tmp_path)
        tsv = cfg.project_path / "enrichment_variant_data.txt"
        assert tsv.exists(), "enrichment_variant_data.txt not created"
        # _write_parquet_bundle strips .parquet suffix to make a bundle directory
        parquet_dir = cfg.project_path / "test_enrichment_enrichment"
        assert parquet_dir.is_dir(), "Parquet bundle dir not created"

    def test_output_has_expected_columns(self, tmp_path):
        cfg = self._run_enrichment(tmp_path)
        tsv = cfg.project_path / "enrichment_variant_data.txt"
        out = pl.read_csv(str(tsv), separator="\t", null_values=["NA"])
        assert "nt_seq" in out.columns
        assert "fitness" in out.columns
        assert "sigma" in out.columns
        assert "mean_count" in out.columns
        assert "length" in out.columns

    def test_output_row_count(self, tmp_path):
        cfg = self._run_enrichment(tmp_path)
        tsv = cfg.project_path / "enrichment_variant_data.txt"
        out = pl.read_csv(str(tsv), separator="\t", null_values=["NA"])
        assert len(out) == 3

    def test_parquet_readable(self, tmp_path):
        cfg = self._run_enrichment(tmp_path)
        # bundle dir = project_name + "_enrichment" (no .parquet suffix)
        parquet_path = cfg.project_path / "test_enrichment_enrichment" / "variants.parquet"
        df = pl.read_parquet(str(parquet_path))
        assert len(df) == 3

    def test_single_replicate_works(self, tmp_path):
        cfg = self._run_enrichment(tmp_path, n_reps=1)
        tsv = cfg.project_path / "enrichment_variant_data.txt"
        assert tsv.exists()
        out = pl.read_csv(str(tsv), separator="\t", null_values=["NA"])
        assert len(out) == 3


# ---------------------------------------------------------------------------
# Tests: end-to-end through run_pipeline
# ---------------------------------------------------------------------------

class TestEndToEndEnrichment:
    def test_pipeline_enrichment_mode(self, tmp_path):
        """Full end-to-end: run_pipeline with enrichment_mode=True."""
        from pydimsum.config import RunConfig
        from pydimsum.pipeline import run_pipeline

        # Write a minimal count file (STEAM-only entry point)
        # Two sequences, two replicates, no WT row
        count_data = (
            "nt_seq\tinput1\toutput1\tinput2\toutput2\n"
            "acgtacgt\t100\t300\t80\t240\n"
            "ttttttttt\t50\t25\t60\t30\n"
            "gcgcgcgc\t200\t600\t150\t450\n"
        )
        count_file = tmp_path / "counts.txt"
        count_file.write_text(count_data)

        # Experiment design with 2 replicates, no biological_replicate column
        exp_design_data = (
            "sample_name\texperiment_replicate\tselection_id\n"
            "input1\t1\t0\n"
            "output1\t1\t1\n"
            "input2\t2\t0\n"
            "output2\t2\t1\n"
        )
        exp_file = tmp_path / "design.txt"
        exp_file.write_text(exp_design_data)

        cfg = RunConfig(
            experiment_design_path=exp_file,
            wildtype_sequence="",
            count_path=count_file,
            enrichment_mode=True,
            enrichment_normalise="none",
            output_path=tmp_path,
            project_name="e2e_test",
            # Disable error model (requires mutation-specific columns)
            fitness_error_model=False,
            fitness_normalise=False,
        )

        run_pipeline(cfg)

        tsv = tmp_path / "e2e_test" / "enrichment_variant_data.txt"
        assert tsv.exists(), "enrichment_variant_data.txt not created by pipeline"
        out = pl.read_csv(str(tsv), separator="\t", null_values=["NA"])
        assert len(out) == 3, f"Expected 3 rows, got {len(out)}"
        assert "fitness" in out.columns
        assert "sigma" in out.columns

        # No singles/doubles/wildtype files should be written
        for fname in ["fitness_singles.txt", "fitness_doubles.txt", "fitness_wildtype.txt"]:
            assert not (tmp_path / "e2e_test" / fname).exists(), \
                f"{fname} should not exist in enrichment mode"

    def test_pipeline_mutation_mode_unchanged(self, tmp_path):
        """Mutation mode still works and doesn't emit enrichment files."""
        from pydimsum.config import RunConfig
        from pydimsum.pipeline import run_pipeline

        toy_count = Path(__file__).parent / "data" / "countFile_Toy.txt"
        toy_design = Path(__file__).parent / "data" / "experimentDesign_Toy.txt"
        if not toy_count.exists():
            pytest.skip("Toy data not available")

        # WT sequence for the Toy demo (TDP-1 domain, 126 nt)
        WT_SEQ = "GGTAATAGCAGAGGGGGTGGAGCTGGTTTGGGAAACAATCAAGGTAGTAATATGGGTGGTGGGATGAACTTTGGTGCGTTCAGCATTAATCCAGCCATGATGGCTGCCGCCCAGGCAGCACTACAG"

        cfg = RunConfig(
            experiment_design_path=toy_design,
            wildtype_sequence=WT_SEQ,
            count_path=toy_count,
            output_path=tmp_path,
            project_name="mutation_test",
            fitness_error_model=False,
            fitness_normalise=False,
        )
        run_pipeline(cfg)

        singles = tmp_path / "mutation_test" / "fitness_singles.txt"
        assert singles.exists(), "Mutation mode should produce fitness_singles.txt"
        enrichment = tmp_path / "mutation_test" / "enrichment_variant_data.txt"
        assert not enrichment.exists(), "Mutation mode should NOT produce enrichment_variant_data.txt"
