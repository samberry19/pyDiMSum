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


# ---------------------------------------------------------------------------
# Trans-library concatenation
# ---------------------------------------------------------------------------


class TestConcatenateReads:
    """Tests for the trans-library R1+R2 concatenation path."""

    _PHRED_OFFSET = 33

    def _phred(self, score: int) -> str:
        return chr(score + self._PHRED_OFFSET)

    def _qual(self, score: int, length: int) -> str:
        return self._phred(score) * length

    def _write_fastq(self, path: Path, reads: list[tuple[str, str]]) -> None:
        """Write a (possibly gzipped) FASTQ file. Each tuple is (seq, qual)."""
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "wt") as fh:  # type: ignore[call-overload]
            for i, (seq, qual) in enumerate(reads):
                fh.write(f"@read{i}\n{seq}\n+\n{qual}\n")

    def _read_fastq(self, path: Path) -> list[tuple[str, str]]:
        """Read all (seq, qual) pairs from a FASTQ file."""
        reads = []
        with gzip.open(path, "rt") as fh:
            while True:
                h = fh.readline()
                if not h:
                    break
                seq = fh.readline().rstrip()
                fh.readline()
                qual = fh.readline().rstrip()
                reads.append((seq, qual))
        return reads

    def _fake_config(self, min_len=4, min_qual=20, max_ee=10.0, rc=False):
        """Minimal config-like object with trans-library quality parameters."""
        class FakeConfig:
            cutadapt_min_length = min_len
            vsearch_min_qual = min_qual
            vsearch_max_ee = max_ee
            trans_library_reverse_complement = rc
        return FakeConfig()

    def test_basic_concatenation(self, tmp_path):
        """R1 and R2 sequences are concatenated into a single read."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        self._write_fastq(r1, [("ACGT", self._qual(30, 4))])
        self._write_fastq(r2, [("TTTT", self._qual(30, 4))])

        _concatenate_reads(str(r1), str(r2), out, report, self._fake_config())

        reads = self._read_fastq(out)
        assert len(reads) == 1
        assert reads[0][0] == "ACGTTTTT"
        assert len(reads[0][1]) == 8

    def test_reverse_complement_r2(self, tmp_path):
        """With trans_library_reverse_complement=True, R2 is revcomped before concat."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        self._write_fastq(r1, [("ACGT", self._qual(30, 4))])
        self._write_fastq(r2, [("AAAC", self._qual(30, 4))])  # revcomp = GTTT

        _concatenate_reads(str(r1), str(r2), out, report, self._fake_config(rc=True))

        reads = self._read_fastq(out)
        assert reads[0][0] == "ACGTGTTT"

    def test_rc_qual_string_reversed(self, tmp_path):
        """Quality string for R2 is reversed when reverse-complementing."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        q1 = "IIII"  # Phred 40 × 4
        q2 = "ABCD"  # distinct per-base qualities
        self._write_fastq(r1, [("ACGT", q1)])
        self._write_fastq(r2, [("TTTT", q2)])

        _concatenate_reads(str(r1), str(r2), out, report, self._fake_config(rc=True))

        reads = self._read_fastq(out)
        # R2 qual should be reversed: "DCBA"
        assert reads[0][1] == "IIII" + "DCBA"

    def test_too_short_filtered(self, tmp_path):
        """Pairs where either read is shorter than min_len are discarded."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        # Read 2 of second pair is only 2 nt (below min_len=4)
        self._write_fastq(r1, [("ACGT", self._qual(30, 4)), ("ACGT", self._qual(30, 4))])
        self._write_fastq(r2, [("TTTT", self._qual(30, 4)), ("TT", self._qual(30, 2))])

        _concatenate_reads(str(r1), str(r2), out, report, self._fake_config(min_len=4))

        reads = self._read_fastq(out)
        assert len(reads) == 1
        assert reads[0][0] == "ACGTTTTT"

    def test_low_quality_filtered(self, tmp_path):
        """Pairs with any base below min_qual are discarded."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        good_qual = self._qual(30, 4)
        bad_qual = self._phred(10) + self._qual(30, 3)  # first base Phred 10
        self._write_fastq(r1, [("ACGT", good_qual), ("ACGT", bad_qual)])
        self._write_fastq(r2, [("TTTT", good_qual), ("TTTT", good_qual)])

        _concatenate_reads(str(r1), str(r2), out, report, self._fake_config(min_qual=20))

        reads = self._read_fastq(out)
        assert len(reads) == 1

    def test_high_expected_errors_filtered(self, tmp_path):
        """Pairs with combined expected errors > max_ee are discarded."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        # Phred 10 → error prob = 0.1 per base; 4+4=8 bases → ee=0.8 (< max_ee=1.0 → pass)
        # Phred 3  → error prob = 0.5 per base; 4+4=8 bases → ee=4.0 (> max_ee=1.0 → fail)
        good_qual = self._qual(10, 4)
        bad_qual  = self._qual(3, 4)   # Phred 3 ≈ 50% error per base
        self._write_fastq(r1, [("ACGT", good_qual), ("ACGT", bad_qual)])
        self._write_fastq(r2, [("TTTT", good_qual), ("TTTT", good_qual)])

        _concatenate_reads(str(r1), str(r2), out, report, self._fake_config(min_qual=1, max_ee=1.0))

        reads = self._read_fastq(out)
        assert len(reads) == 1

    def test_report_written(self, tmp_path):
        """A report file is written with read pair statistics."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        self._write_fastq(r1, [("ACGT", self._qual(30, 4)), ("ACGT", self._qual(30, 4))])
        self._write_fastq(r2, [("TTTT", self._qual(30, 4)), ("TT", self._qual(30, 2))])

        _concatenate_reads(str(r1), str(r2), out, report, self._fake_config(min_len=4))

        assert report.exists()
        text = report.read_text()
        assert "Pairs" in text
        assert "Merged" in text
        assert "Too short" in text

    def test_all_reads_pass_when_thresholds_zero(self, tmp_path):
        """With min_len=0, min_qual=0, max_ee=∞ all reads pass."""
        from pydimsum.wrap.align import _concatenate_reads

        r1 = tmp_path / "r1.fastq.gz"
        r2 = tmp_path / "r2.fastq.gz"
        out = tmp_path / "out.fastq.gz"
        report = tmp_path / "out.report"

        reads_in = [("ACGT", "!!!!"), ("TTTT", "!!!!"), ("GCGC", "!!!!")]
        self._write_fastq(r1, reads_in)
        self._write_fastq(r2, reads_in)

        _concatenate_reads(
            str(r1), str(r2), out, report,
            self._fake_config(min_len=0, min_qual=0, max_ee=float("inf"))
        )

        reads_out = self._read_fastq(out)
        assert len(reads_out) == 3


# ---------------------------------------------------------------------------
# Config: trans_library validation
# ---------------------------------------------------------------------------


class TestTransLibraryConfig:
    def test_trans_library_requires_paired(self):
        from pydimsum.config import RunConfig

        with pytest.raises(ValueError, match="trans_library requires paired"):
            RunConfig(
                experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
                wildtype_sequence="ACGT",
                trans_library=True,
                paired=False,
            )

    def test_trans_library_paired_is_valid(self):
        from pydimsum.config import RunConfig

        config = RunConfig(
            experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
            wildtype_sequence="ACGT",
            trans_library=True,
            paired=True,
        )
        assert config.trans_library is True

    def test_trans_library_false_by_default(self):
        from pydimsum.config import RunConfig

        config = RunConfig(
            experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
            wildtype_sequence="ACGT",
        )
        assert config.trans_library is False
        assert config.trans_library_reverse_complement is False


# ---------------------------------------------------------------------------
# Single-end mode: config, _filter_reads_se, and _run_single_end
# ---------------------------------------------------------------------------


class TestSingleEndConfig:
    """paired=False is accepted by RunConfig and is distinct from trans_library."""

    def test_paired_true_by_default(self):
        from pydimsum.config import RunConfig

        config = RunConfig(
            experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
            wildtype_sequence="ACGT",
        )
        assert config.paired is True

    def test_paired_false_accepted(self):
        from pydimsum.config import RunConfig

        config = RunConfig(
            experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
            wildtype_sequence="ACGT",
            paired=False,
        )
        assert config.paired is False

    def test_no_paired_cli_flag(self):
        """--no_paired CLI flag must parse without error and set paired=False."""
        from typer.testing import CliRunner
        from unittest.mock import patch
        from pydimsum.cli import app

        runner = CliRunner()
        captured = {}

        def fake_run(cfg):
            captured["paired"] = cfg.paired

        with patch("pydimsum.pipeline.run_pipeline", fake_run):
            result = runner.invoke(
                app,
                [
                    "--experiment_design_path",
                    str(Path(__file__).parent / "data" / "experimentDesign_Toy.txt"),
                    "--wildtype_sequence", "ACGT",
                    "--no_paired",
                    "--start_stage", "5",  # skip all processing stages
                    "--stop_stage", "5",
                ],
            )
        assert result.exit_code == 0, result.output
        assert captured.get("paired") is False


class TestFilterReadsSe:
    """Unit tests for the single-end quality filter (_filter_reads_se)."""

    def _write_fastq(self, path: Path, reads: list[tuple[str, str]]) -> None:
        with gzip.open(path, "wt") as fh:
            for i, (seq, qual) in enumerate(reads):
                fh.write(f"@read{i}\n{seq}\n+\n{qual}\n")

    def test_passes_high_quality_reads(self, tmp_path):
        from pydimsum.wrap.align import _filter_reads_se

        input_fastq = tmp_path / "in.fastq.gz"
        # Phred 30 = '?'
        self._write_fastq(input_fastq, [("ACGT", "????"), ("TTTT", "????")])

        output_fastq = tmp_path / "out.fastq.gz"
        output_report = tmp_path / "report"

        _filter_reads_se(str(input_fastq), output_fastq, output_report, min_qual=30)

        with gzip.open(output_fastq, "rt") as fh:
            lines = fh.readlines()
        assert len(lines) == 8  # 2 reads × 4 lines

    def test_discards_low_quality_reads(self, tmp_path):
        from pydimsum.wrap.align import _filter_reads_se

        input_fastq = tmp_path / "in.fastq.gz"
        # '#' = Phred 2 (fails min_qual=30), '?' = Phred 30 (passes)
        self._write_fastq(input_fastq, [("ACGT", "????"), ("TTTT", "#???")])

        output_fastq = tmp_path / "out.fastq.gz"
        output_report = tmp_path / "report"

        _filter_reads_se(str(input_fastq), output_fastq, output_report, min_qual=30)

        with gzip.open(output_fastq, "rt") as fh:
            lines = fh.readlines()
        assert len(lines) == 4  # only the first read survives

    def test_report_written(self, tmp_path):
        from pydimsum.wrap.align import _filter_reads_se

        input_fastq = tmp_path / "in.fastq.gz"
        self._write_fastq(input_fastq, [("ACGT", "????")])

        output_fastq = tmp_path / "out.fastq.gz"
        output_report = tmp_path / "report"

        _filter_reads_se(str(input_fastq), output_fastq, output_report, min_qual=0)

        assert output_report.exists()
        text = output_report.read_text()
        assert "Merged" in text


class TestRunSingleEnd:
    """Integration test for _run_single_end via the full align entry point."""

    def _write_fastq(self, path: Path, reads: list[tuple[str, str]]) -> None:
        with gzip.open(path, "wt") as fh:
            for i, (seq, qual) in enumerate(reads):
                fh.write(f"@read{i}\n{seq}\n+\n{qual}\n")

    def test_single_end_produces_output(self, tmp_path):
        """align_reads with paired=False writes a .vsearch.gz without calling vsearch."""
        from pydimsum.config import RunConfig
        from pydimsum.wrap.align import run_align

        # Write a single-end FASTQ (pair1 only)
        fq_dir = tmp_path / "fastq"
        fq_dir.mkdir()
        input_fastq = fq_dir / "sample.fastq.gz"
        # All Phred 30 reads — should all survive the quality filter
        self._write_fastq(input_fastq, [("ACGTACGT", "????????"), ("TTTTTTTT", "????????")])

        exp_df = pl.DataFrame({
            "sample_name": ["sample"],
            "experiment_replicate": [1],
            "experiment": [1],
            "selection_id": [0],
            "biological_replicate": [1],
            "selection_replicate": [1],
            "technical_replicate": [1],
            "pair_directory": [str(fq_dir)],
            "pair1": ["sample.fastq.gz"],
        })

        design_path = tmp_path / "design.tsv"
        design_path.write_text(
            "sample_name\texperiment\tselection_id\tbiological_replicate\n"
            "sample\t1\t0\t1\n"
        )

        config = RunConfig(
            experiment_design_path=design_path,
            wildtype_sequence="ACGTACGT",
            paired=False,
        )

        outpath = tmp_path / "aligned"
        result_df = run_align(config, exp_df, outpath)

        # Output FASTQ must exist and contain the 2 reads (both pass Phred 30 filter)
        expected_gz = outpath / "sample_e1_s0_b1_t1.vsearch.gz"
        assert expected_gz.exists(), f"Expected output not found: {expected_gz}"

        with gzip.open(expected_gz, "rt") as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == 8  # 2 reads × 4 lines

        # align_reads must return the updated df with aligned_pair columns
        assert "aligned_pair" in result_df.columns
        assert "aligned_pair_directory" in result_df.columns


# ---------------------------------------------------------------------------
# Cutadapt global defaults: _fill_cutadapt_defaults
# ---------------------------------------------------------------------------


class TestFillCutadaptDefaults:
    """_fill_cutadapt_defaults mirrors R get_experiment_design.R:48-67, 154-161."""

    def _make_config(self, **kwargs):
        from pydimsum.config import RunConfig
        return RunConfig(
            experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
            wildtype_sequence="ACGT",
            **kwargs,
        )

    def _minimal_df(self, extra_cols: dict | None = None):
        data = {
            "sample_name": ["s1", "s2"],
            "experiment_replicate": [1, 1],
            "experiment": [1, 1],
            "selection_id": [0, 1],
            "biological_replicate": [1, 1],
            "pair_directory": ["/tmp", "/tmp"],
            "pair1": ["r1.fastq.gz", "r2.fastq.gz"],
        }
        if extra_cols:
            data.update(extra_cols)
        return pl.DataFrame(data)

    def test_global_adapter_fills_absent_column(self):
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config(cutadapt_5_first="AAAA")
        df = self._minimal_df()
        result = _fill_cutadapt_defaults(config, df)
        assert "cutadapt5First" in result.columns
        assert result["cutadapt5First"].to_list() == ["AAAA", "AAAA"]

    def test_per_sample_column_not_overridden(self):
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config(cutadapt_5_first="GLOBAL")
        # Per-sample column already present — must not be overridden
        df = self._minimal_df(extra_cols={"cutadapt5First": ["SAMP1", "SAMP2"]})
        result = _fill_cutadapt_defaults(config, df)
        assert result["cutadapt5First"].to_list() == ["SAMP1", "SAMP2"]

    def test_none_config_value_fills_null(self):
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config()  # cutadapt_5_first defaults to None
        df = self._minimal_df()
        result = _fill_cutadapt_defaults(config, df)
        assert "cutadapt5First" in result.columns
        assert all(v is None for v in result["cutadapt5First"].to_list())

    def test_cut_col_filled_as_int(self):
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config(cutadapt_cut_5_first=5)
        df = self._minimal_df()
        result = _fill_cutadapt_defaults(config, df)
        assert "cutadaptCut5First" in result.columns
        assert result["cutadaptCut5First"].to_list() == [5, 5]

    def test_revcomp_3first_from_5second(self):
        """cutadapt3First should be auto-filled as revcomp(cutadapt5Second)."""
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config(cutadapt_5_second="AATTCC")
        df = self._minimal_df()
        result = _fill_cutadapt_defaults(config, df)
        # revcomp("AATTCC") = revcomp of AATTCC
        # complement: TTAAGG, reversed: GGAATT
        assert result["cutadapt3First"].to_list() == ["GGAATT", "GGAATT"]

    def test_revcomp_3second_from_5first(self):
        """cutadapt3Second should be auto-filled as revcomp(cutadapt5First)."""
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config(cutadapt_5_first="GCGCGC")
        df = self._minimal_df()
        result = _fill_cutadapt_defaults(config, df)
        # revcomp("GCGCGC") = GCGCGC (palindrome)
        assert result["cutadapt3Second"].to_list() == ["GCGCGC", "GCGCGC"]

    def test_explicit_3first_not_overridden_by_revcomp(self):
        """If cutadapt3First is already set, revcomp auto-fill must not override it."""
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config(cutadapt_5_second="AATTCC", cutadapt_3_first="EXPLICIT")
        df = self._minimal_df()
        result = _fill_cutadapt_defaults(config, df)
        assert result["cutadapt3First"].to_list() == ["EXPLICIT", "EXPLICIT"]

    def test_revcomp_skipped_for_trans_library(self):
        """Trans-library mode must not auto-fill rev-comp adapters."""
        from pydimsum.wrap.trim import _fill_cutadapt_defaults
        config = self._make_config(cutadapt_5_second="AATTCC", trans_library=True)
        df = self._minimal_df()
        result = _fill_cutadapt_defaults(config, df)
        # cutadapt3First column should be all None (no revcomp applied)
        assert all(v is None for v in result["cutadapt3First"].to_list())

    def test_stranded_false_requires_paired(self):
        from pydimsum.config import RunConfig
        with pytest.raises(ValueError, match="stranded=False requires paired=True"):
            RunConfig(
                experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
                wildtype_sequence="ACGT",
                stranded=False,
                paired=False,
            )


# ---------------------------------------------------------------------------
# Passthrough cutadapt options
# ---------------------------------------------------------------------------


class TestGetPassthroughOptions:
    """_get_passthrough_options: shlex-split forwarding with managed-flag guard."""

    def _make_config(self, **kwargs):
        from pydimsum.config import RunConfig
        return RunConfig(
            experiment_design_path=Path(__file__).parent / "data" / "experimentDesign_Toy.txt",
            wildtype_sequence="ACGT",
            **kwargs,
        )

    def test_none_config_returns_empty(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config()  # cutadapt_options defaults to None
        assert _get_passthrough_options({}, config) == []

    def test_empty_string_returns_empty(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="")
        assert _get_passthrough_options({}, config) == []

    def test_single_flag(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="--discard-untrimmed")
        assert _get_passthrough_options({}, config) == ["--discard-untrimmed"]

    def test_multiple_flags_split_correctly(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="--discard-untrimmed --max-n 0")
        assert _get_passthrough_options({}, config) == [
            "--discard-untrimmed", "--max-n", "0"
        ]

    def test_per_sample_row_overrides_global_config(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="--discard-untrimmed")
        row = {"cutadaptOptions": "--max-n 1"}
        assert _get_passthrough_options(row, config) == ["--max-n", "1"]

    def test_discard_untrimmed_is_allowed(self):
        """--discard-untrimmed must not raise (PacBio use-case, idempotent)."""
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="--discard-untrimmed")
        result = _get_passthrough_options({}, config)
        assert "--discard-untrimmed" in result

    # ---- managed-flag guard ----

    def test_managed_short_error_rate_raises(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="-e 0.1")
        with pytest.raises(ValueError, match="managed by pyDiMSum"):
            _get_passthrough_options({}, config)

    def test_managed_long_error_rate_raises(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="--error-rate 0.1")
        with pytest.raises(ValueError, match="managed by pyDiMSum"):
            _get_passthrough_options({}, config)

    def test_managed_long_equals_form_raises(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="--error-rate=0.1")
        with pytest.raises(ValueError, match="managed by pyDiMSum"):
            _get_passthrough_options({}, config)

    def test_managed_short_attached_value_raises(self):
        """e.g. -e0.1 (value attached directly to short flag)."""
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="-e0.1")
        with pytest.raises(ValueError, match="managed by pyDiMSum"):
            _get_passthrough_options({}, config)

    def test_managed_minimum_length_raises(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="--minimum-length 10")
        with pytest.raises(ValueError, match="managed by pyDiMSum"):
            _get_passthrough_options({}, config)

    def test_managed_adapter_flag_raises(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="-g ATCG")
        with pytest.raises(ValueError, match="managed by pyDiMSum"):
            _get_passthrough_options({}, config)

    def test_managed_output_flag_raises(self):
        from pydimsum.wrap.trim import _get_passthrough_options
        config = self._make_config(cutadapt_options="-o out.fq")
        with pytest.raises(ValueError, match="managed by pyDiMSum"):
            _get_passthrough_options({}, config)
