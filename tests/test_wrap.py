"""Unit tests for WRAP helper modules.

These tests do NOT require external binaries (cutadapt, FastQC, vsearch,
starcode) — they test Python-only logic only.

Tests that would require external binaries are marked with
``pytest.mark.external_binary`` and skipped unless the binary is available.
"""

from __future__ import annotations

import gzip
import subprocess
import tempfile
from pathlib import Path

import polars as pl
import pytest

from pydimsum.wrap import check_binaries, BinaryNotFoundError


# ---------------------------------------------------------------------------
# Binary presence checks
# ---------------------------------------------------------------------------


class TestCheckBinaries:
    def test_missing_binary_raises(self, monkeypatch):
        """check_binaries raises BinaryNotFoundError when a binary is absent."""
        import shutil
        # Patch shutil.which to always return None for "cutadapt"
        original_which = shutil.which
        def patched_which(name, *args, **kwargs):
            if name == "cutadapt":
                return None
            return original_which(name, *args, **kwargs)

        monkeypatch.setattr(shutil, "which", patched_which)
        with pytest.raises(BinaryNotFoundError, match="cutadapt"):
            check_binaries([0])

    def test_no_stages_no_error(self, monkeypatch):
        """check_binaries([]) should not raise even if all binaries absent."""
        import shutil
        monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
        check_binaries([])  # empty stages list — no binaries required

    def test_message_includes_install_tip(self, monkeypatch):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
        try:
            check_binaries([0])
        except BinaryNotFoundError as e:
            assert "cutadapt" in str(e).lower()
            assert "conda" in str(e).lower() or "pip" in str(e).lower()


# ---------------------------------------------------------------------------
# Trim: cutadapt option builders
# ---------------------------------------------------------------------------


class TestGetAdapterOptions:
    def test_5prime_only(self):
        from pydimsum.wrap.trim import _get_adapter_options
        row = {"cutadapt5First": "ATCG", "cutadapt5Second": None,
               "cutadapt3First": None, "cutadapt3Second": None,
               "run_cutadapt_cutonly": False}
        opts = _get_adapter_options(row, paired=True)
        assert "-g" in opts
        idx = opts.index("-g")
        assert opts[idx + 1] == "ATCG"
        assert "--discard-untrimmed" in opts

    def test_paired_adapters(self):
        from pydimsum.wrap.trim import _get_adapter_options
        row = {
            "cutadapt5First": "AAAA",
            "cutadapt5Second": "CCCC",
            "cutadapt3First": "TTTT",
            "cutadapt3Second": "GGGG",
            "run_cutadapt_cutonly": False,
        }
        opts = _get_adapter_options(row, paired=True)
        assert "-g" in opts and "AAAA" in opts
        assert "-G" in opts and "CCCC" in opts
        assert "-a" in opts and "TTTT" in opts
        assert "-A" in opts and "GGGG" in opts

    def test_cutonly_no_discard_untrimmed(self):
        from pydimsum.wrap.trim import _get_adapter_options
        row = {"cutadapt5First": "ATCG", "cutadapt5Second": None,
               "cutadapt3First": None, "cutadapt3Second": None,
               "run_cutadapt_cutonly": True}
        opts = _get_adapter_options(row, paired=False)
        assert "--discard-untrimmed" not in opts

    def test_cut_options(self):
        from pydimsum.wrap.trim import _get_cut_options
        row = {"cutadaptCut5First": 3, "cutadaptCut3First": 5,
               "cutadaptCut5Second": 2, "cutadaptCut3Second": 4}
        opts = _get_cut_options(row, paired=True)
        assert "-u" in opts
        idx5 = opts.index("-u")
        assert opts[idx5 + 1] == "3"
        assert "-U" in opts

    def test_no_options_empty_list(self):
        from pydimsum.wrap.trim import _get_adapter_options, _get_cut_options
        row = {k: None for k in [
            "cutadapt5First", "cutadapt5Second", "cutadapt3First", "cutadapt3Second",
            "cutadaptCut5First", "cutadaptCut3First", "cutadaptCut5Second", "cutadaptCut3Second",
            "run_cutadapt_cutonly",
        ]}
        # No adapter opts (only --discard-untrimmed remains — but cutonly=None→False)
        adapter_opts = _get_adapter_options(row, paired=True)
        # With all adapters None, only --discard-untrimmed remains
        assert "-g" not in adapter_opts
        assert "-a" not in adapter_opts

        cut_opts = _get_cut_options(row, paired=True)
        assert cut_opts == []


class TestLinkedAdapters:
    def test_converts_to_linked_when_both_adapters_and_long_read(self):
        from pydimsum.wrap.trim import _convert_linked_adapters
        from pydimsum.config import RunConfig

        config = RunConfig(
            experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
            wildtype_sequence="ACGTACGTACGT",  # 12 nt, all variable (uppercase)
        )

        # Build a minimal exp_design_df with both 5' and 3' adapters
        df = pl.DataFrame({
            "sample_name": ["s1"],
            "experiment_replicate": [1],
            "experiment": [1],
            "selection_id": [0],
            "biological_replicate": [1],
            "pair_directory": ["/tmp"],
            "pair1": ["r1.fastq.gz"],
            "pair2": ["r2.fastq.gz"],
            "pair1_length": [100],
            "pair2_length": [100],
            "cutadapt5First": ["AAAAAAAAAA"],   # 10 nt 5' adapter
            "cutadapt3First": ["TTTTTTTTTT"],   # 10 nt 3' adapter
            "cutadapt5Second": [None],
            "cutadapt3Second": [None],
            "cutadaptCut5First": [None],
            "cutadaptCut3First": [None],
            "cutadaptCut5Second": [None],
            "cutadaptCut3Second": [None],
        })

        result = _convert_linked_adapters(config, df)

        # 5' adapter should be consumed into the linked format for 3' adapter
        # pair1_length(100) - 0 - 0 = 100 > len("AAAAAAAAAA") + 12 = 22 → convert
        row = result.row(0, named=True)
        assert row["cutadapt5First"] is None
        assert "AAAAAAAAAA;required" in row["cutadapt3First"]


# ---------------------------------------------------------------------------
# Align: quality filter
# ---------------------------------------------------------------------------


class TestFilterReads:
    def _write_fastq(self, path: Path, reads: list[tuple[str, str]]) -> None:
        """Write a minimal FASTQ file (gzipped). Each tuple is (seq, qual)."""
        with gzip.open(path, "wt") as fh:
            for i, (seq, qual) in enumerate(reads):
                fh.write(f"@read{i}\n{seq}\n+\n{qual}\n")

    def test_filters_low_quality_reads(self, tmp_path):
        from pydimsum.wrap.align import _filter_reads

        input_fastq = tmp_path / "in.fastq.gz"
        # Phred 30 = ASCII '?'; Phred 2 = '#' (low quality)
        reads = [
            ("ACGT", "????"),  # all Phred 30 → keep
            ("TTTT", "????"),  # keep
            ("GGGG", "#???"),  # first base Phred 2 (<30) → discard
        ]
        self._write_fastq(input_fastq, reads)

        # Write a minimal fake VSEARCH report
        report_in = tmp_path / "report.prefilter"
        report_in.write_text("")

        output_fastq = tmp_path / "out.fastq.gz"
        output_report = tmp_path / "report.final"

        _filter_reads(
            input_fastq=input_fastq,
            input_report=report_in,
            output_fastq=output_fastq,
            output_report=output_report,
            min_qual=30,
        )

        # Only 2 reads should pass
        with gzip.open(output_fastq, "rt") as fh:
            lines = fh.readlines()
        assert len(lines) == 8  # 2 reads × 4 lines each

    def test_all_pass_when_qual_threshold_zero(self, tmp_path):
        from pydimsum.wrap.align import _filter_reads

        input_fastq = tmp_path / "in.fastq.gz"
        reads = [("ACGT", "!!!!"), ("TTTT", "!!!!")]  # Phred 0
        self._write_fastq(input_fastq, reads)

        report_in = tmp_path / "report.prefilter"
        report_in.write_text("")
        output_fastq = tmp_path / "out.fastq.gz"
        output_report = tmp_path / "report.final"

        _filter_reads(
            input_fastq=input_fastq,
            input_report=report_in,
            output_fastq=output_fastq,
            output_report=output_report,
            min_qual=0,
        )
        with gzip.open(output_fastq, "rt") as fh:
            lines = fh.readlines()
        assert len(lines) == 8  # both reads pass

    def test_length_distribution_in_report(self, tmp_path):
        from pydimsum.wrap.align import _filter_reads

        input_fastq = tmp_path / "in.fastq.gz"
        reads = [("ACGT", "????"), ("ACGTAC", "??????"), ("ACGTACGT", "????????")]
        self._write_fastq(input_fastq, reads)

        report_in = tmp_path / "report.prefilter"
        report_in.write_text("")
        output_fastq = tmp_path / "out.fastq.gz"
        output_report = tmp_path / "report.final"

        _filter_reads(
            input_fastq=input_fastq,
            input_report=report_in,
            output_fastq=output_fastq,
            output_report=output_report,
            min_qual=1,
        )
        report_text = output_report.read_text()
        assert "Min" in report_text
        assert "Median" in report_text
        assert "4  Min" in report_text      # shortest sequence
        assert "8  Max" in report_text      # longest sequence


# ---------------------------------------------------------------------------
# Tally: starcode output trimming
# ---------------------------------------------------------------------------


class TestTrimStarcodeOutput:
    def test_trims_third_column(self, tmp_path):
        from pydimsum.wrap.tally import _trim_starcode_output

        output_file = tmp_path / "counts.vsearch.unique"
        output_file.write_text(
            "ACGT\t100\tACGT,ACGT\n"
            "TTTT\t50\tTTTT\n"
        )
        _trim_starcode_output(output_file)
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            assert line.count("\t") == 1, f"Expected 1 tab, got: {line!r}"

    def test_handles_two_column_input(self, tmp_path):
        from pydimsum.wrap.tally import _trim_starcode_output

        output_file = tmp_path / "counts.vsearch.unique"
        output_file.write_text("ACGT\t100\nTTTT\t50\n")
        _trim_starcode_output(output_file)
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "ACGT\t100"

    def test_empty_file(self, tmp_path):
        from pydimsum.wrap.tally import _trim_starcode_output

        output_file = tmp_path / "counts.vsearch.unique"
        output_file.write_text("")
        _trim_starcode_output(output_file)  # Should not raise
        assert output_file.read_text() == ""


# ---------------------------------------------------------------------------
# Merge: build_variant_table_from_wrap (end-to-end with mock starcode files)
# ---------------------------------------------------------------------------


class TestBuildFromWrapFiles:
    """Test that build_variant_table correctly reads WRAP starcode output files."""

    def _write_starcode(self, path: Path, seqs: dict[str, int]) -> None:
        """Write a minimal 2-column starcode count file."""
        with open(path, "w") as fh:
            for seq, count in seqs.items():
                fh.write(f"{seq}\t{count}\n")

    def test_single_replicate(self, tmp_path):
        """Single experiment replicate with one input and one output sample."""
        from pydimsum.config import RunConfig
        from pydimsum.io.designs import ExperimentDesign
        from pydimsum.steam.merge import build_variant_table

        # Write mock starcode files
        unique_dir = tmp_path / "tally"
        unique_dir.mkdir()

        s0_file = unique_dir / "sampleA_e1_s0_b1_t1.vsearch.unique"
        s1_file = unique_dir / "sampleB_e1_s1_b1_t1.vsearch.unique"

        self._write_starcode(s0_file, {"acgt": 100, "tttt": 50, "gcgc": 20})
        self._write_starcode(s1_file, {"acgt": 200, "tttt": 30})

        # Build exp_design_df with WRAP columns
        exp_df = pl.DataFrame({
            "sample_name": ["sampleA", "sampleB"],
            "experiment_replicate": [1, 1],
            "experiment": [1, 1],
            "selection_id": [0, 1],
            "biological_replicate": [1, 1],
            "selection_replicate": [1, 1],
            "technical_replicate": [1, 1],
            "pair_directory": [str(tmp_path), str(tmp_path)],
            "pair1": ["dummy.fastq.gz", "dummy.fastq.gz"],
            "pair2": ["dummy.fastq.gz", "dummy.fastq.gz"],
            "aligned_pair": ["sampleA_e1_s0_b1_t1.vsearch.gz", "sampleB_e1_s1_b1_t1.vsearch.gz"],
            "aligned_pair_directory": [str(unique_dir), str(unique_dir)],
            "aligned_pair_unique": [
                "sampleA_e1_s0_b1_t1.vsearch.unique",
                "sampleB_e1_s1_b1_t1.vsearch.unique",
            ],
            "aligned_pair_unique_directory": [str(unique_dir), str(unique_dir)],
            "generations": [None, None],
            "cell_density": [None, None],
            "selection_time": [None, None],
        })

        # Patch ExperimentDesign to avoid reading from filesystem
        exp_design_path = tmp_path / "design.tsv"
        exp_design_path.write_text(
            "sample_name\texperiment\tselection_id\tbiological_replicate\n"
            "sampleA\t1\t0\t1\n"
            "sampleB\t1\t1\t1\n"
        )
        exp_design = ExperimentDesign(exp_design_path)
        exp_design.df = exp_df

        # Build without count_path (triggers WRAP path)
        config = RunConfig(
            experiment_design_path=exp_design_path,
            wildtype_sequence="ACGT",
            count_path=None,
        )

        result = build_variant_table(config, exp_design)

        # Should have 3 variants (union of acgt, tttt, gcgc)
        assert len(result) == 3
        assert "count_e1_s0" in result.columns
        assert "count_e1_s1" in result.columns

        # Check counts
        r = result.sort("nt_seq")
        seqs = r["nt_seq"].to_list()
        assert set(seqs) == {"acgt", "gcgc", "tttt"}

        # acgt: s0=100, s1=200
        acgt_row = r.filter(pl.col("nt_seq") == "acgt").row(0, named=True)
        assert acgt_row["count_e1_s0"] == 100
        assert acgt_row["count_e1_s1"] == 200

        # gcgc: s0=20, s1=0 (absent in output)
        gcgc_row = r.filter(pl.col("nt_seq") == "gcgc").row(0, named=True)
        assert gcgc_row["count_e1_s0"] == 20
        assert gcgc_row["count_e1_s1"] == 0

    def test_technical_replicates_summed(self, tmp_path):
        """Technical replicates for the same sample should be summed."""
        from pydimsum.config import RunConfig
        from pydimsum.io.designs import ExperimentDesign
        from pydimsum.steam.merge import build_variant_table

        unique_dir = tmp_path / "tally"
        unique_dir.mkdir()

        # Two technical replicates for the input
        t1_file = unique_dir / "sampleA_e1_s0_b1_t1.vsearch.unique"
        t2_file = unique_dir / "sampleA_e1_s0_b1_t2.vsearch.unique"
        s1_file = unique_dir / "sampleB_e1_s1_b1_t1.vsearch.unique"

        self._write_starcode(t1_file, {"acgt": 100})
        self._write_starcode(t2_file, {"acgt": 50, "tttt": 25})
        self._write_starcode(s1_file, {"acgt": 200})

        exp_df = pl.DataFrame({
            "sample_name": ["sampleA", "sampleA", "sampleB"],
            "experiment_replicate": [1, 1, 1],
            "experiment": [1, 1, 1],
            "selection_id": [0, 0, 1],
            "biological_replicate": [1, 1, 1],
            "selection_replicate": [1, 1, 1],
            "technical_replicate": [1, 2, 1],
            "pair_directory": [str(tmp_path)] * 3,
            "pair1": ["d.fastq.gz"] * 3,
            "pair2": ["d.fastq.gz"] * 3,
            "aligned_pair": [
                "sampleA_e1_s0_b1_t1.vsearch.gz",
                "sampleA_e1_s0_b1_t2.vsearch.gz",
                "sampleB_e1_s1_b1_t1.vsearch.gz",
            ],
            "aligned_pair_directory": [str(unique_dir)] * 3,
            "aligned_pair_unique": [
                "sampleA_e1_s0_b1_t1.vsearch.unique",
                "sampleA_e1_s0_b1_t2.vsearch.unique",
                "sampleB_e1_s1_b1_t1.vsearch.unique",
            ],
            "aligned_pair_unique_directory": [str(unique_dir)] * 3,
            "generations": [None, None, None],
            "cell_density": [None, None, None],
            "selection_time": [None, None, None],
        })

        exp_design_path = tmp_path / "design.tsv"
        exp_design_path.write_text(
            "sample_name\texperiment\tselection_id\tbiological_replicate\n"
            "sampleA\t1\t0\t1\n"
            "sampleA\t1\t0\t1\n"
            "sampleB\t1\t1\t1\n"
        )
        exp_design = ExperimentDesign(exp_design_path)
        exp_design.df = exp_df

        config = RunConfig(
            experiment_design_path=exp_design_path,
            wildtype_sequence="ACGT",
            count_path=None,
        )

        result = build_variant_table(config, exp_design)

        # acgt: t1+t2 = 150 in input
        acgt_row = result.filter(pl.col("nt_seq") == "acgt").row(0, named=True)
        assert acgt_row["count_e1_s0"] == 150  # 100 + 50

        # tttt: only in t2 = 25 in input, 0 in output
        tttt_row = result.filter(pl.col("nt_seq") == "tttt").row(0, named=True)
        assert tttt_row["count_e1_s0"] == 25
        assert tttt_row["count_e1_s1"] == 0
