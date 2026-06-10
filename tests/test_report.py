"""Smoke tests for the HTML report generator."""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("jinja2", reason="jinja2 not installed")
pytest.importorskip("matplotlib", reason="matplotlib not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_stats(count_cols: list[str]) -> dict:
    """Minimal stats dict as returned by process_variants._compute_stats."""
    nuc_subst = {}
    for col in count_cols:
        nuc_subst[col] = {0: 500, 1: 300, 2: 80, 3: 10}
    return {
        "nuc_subst_dict":  nuc_subst,
        "nuc_indel_dict":  {c: 15  for c in count_cols},
        "nuc_mxsub_dict":  {c: 5   for c in count_cols},
        "nuc_tmsub_dict":  {c: 30  for c in count_cols},
        "nuc_frbdn_dict":  {c: 20  for c in count_cols},
        "nuc_const_dict":  {c: 10  for c in count_cols},
        "nuc_nbarc_dict":  {c: 0   for c in count_cols},
        "n_input":         1000,
    }


def _make_variant_df(n=50, n_reps=2, seed=7) -> tuple[pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    count_cols = []
    data = {
        "nt_seq": [f"seq{i:03d}" for i in range(n)],
        "Nham_nt": rng.integers(0, 4, size=n).tolist(),
    }
    for E in range(1, n_reps + 1):
        for s in (0, 1):
            col = f"count_e{E}_s{s}"
            data[col] = rng.integers(0, 500, size=n).tolist()
            count_cols.append(col)
        data[f"fitness{E}_uncorr"] = rng.standard_normal(n).tolist()
        data[f"sigma{E}_uncorr"] = (rng.random(n) * 0.1 + 0.01).tolist()
    return pl.DataFrame(data), count_cols


class _FakeConfig:
    project_name = "test_report"
    sequence_type = "noncoding"
    sequence_type_resolved = "noncoding"
    wildtype_sequence = "acgt" * 10
    wt_nt_seq = "acgt" * 10
    mixed_substitutions = False
    max_substitutions = 2
    min_input_count = 10
    min_total_count = 0
    fitness_normalise = True
    fitness_error_model = True
    bayesian_double_fitness = False
    enrichment_mode = False
    enrichment_normalise = "median"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_file_created(self, tmp_path):
        from pydimsum.report.html import generate_report

        variant_df, count_cols = _make_variant_df(n=50, n_reps=2)
        stats = _make_stats(count_cols)
        config = _FakeConfig()
        config.project_name = "test_proj"

        out = generate_report(
            stats=stats,
            variant_df=variant_df,
            fitness_df=variant_df,
            singles_df=None,
            replicates=[1, 2],
            count_cols=count_cols,
            config=config,
            output_path=tmp_path,
        )
        assert out.exists()
        assert out.suffix == ".html"

    def test_report_is_valid_html(self, tmp_path):
        from pydimsum.report.html import generate_report

        variant_df, count_cols = _make_variant_df(n=50, n_reps=2)
        config = _FakeConfig()
        config.project_name = "html_check"

        out = generate_report(
            stats=_make_stats(count_cols),
            variant_df=variant_df,
            fitness_df=variant_df,
            singles_df=None,
            replicates=[1, 2],
            count_cols=count_cols,
            config=config,
            output_path=tmp_path,
        )
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "html_check" in html
        assert "pyDiMSum" in html

    def test_report_contains_base64_images(self, tmp_path):
        from pydimsum.report.html import generate_report

        variant_df, count_cols = _make_variant_df(n=50, n_reps=2)
        config = _FakeConfig()
        config.project_name = "img_check"

        out = generate_report(
            stats=_make_stats(count_cols),
            variant_df=variant_df,
            fitness_df=variant_df,
            singles_df=None,
            replicates=[1, 2],
            count_cols=count_cols,
            config=config,
            output_path=tmp_path,
        )
        html = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," in html

    def test_report_single_replicate(self, tmp_path):
        """Single replicate should produce a report without crashing."""
        from pydimsum.report.html import generate_report

        variant_df, count_cols = _make_variant_df(n=30, n_reps=1)
        config = _FakeConfig()
        config.project_name = "single_rep"

        out = generate_report(
            stats=_make_stats(count_cols),
            variant_df=variant_df,
            fitness_df=variant_df,
            singles_df=None,
            replicates=[1],
            count_cols=count_cols,
            config=config,
            output_path=tmp_path,
        )
        assert out.exists()

    def test_report_empty_stats(self, tmp_path):
        """Empty stats dict should not crash the report."""
        from pydimsum.report.html import generate_report

        variant_df, count_cols = _make_variant_df(n=30, n_reps=2)
        config = _FakeConfig()
        config.project_name = "empty_stats"

        out = generate_report(
            stats={},
            variant_df=variant_df,
            fitness_df=variant_df,
            singles_df=None,
            replicates=[1, 2],
            count_cols=count_cols,
            config=config,
            output_path=tmp_path,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# Error model QQ plot
# ---------------------------------------------------------------------------

class TestQQPlot:
    def test_qqplot_returns_bytes(self):
        from pydimsum.report.plots import plot_error_model_qqplot

        rng = np.random.default_rng(42)
        n = 100
        df = pl.DataFrame({
            "nt_seq": [f"s{i}" for i in range(n)],
            "error_model": [True] * n,
            "fitness1_uncorr": rng.standard_normal(n).tolist(),
            "sigma1_uncorr": (rng.random(n) * 0.1 + 0.05).tolist(),
            "fitness2_uncorr": rng.standard_normal(n).tolist(),
            "sigma2_uncorr": (rng.random(n) * 0.1 + 0.05).tolist(),
        })
        png = plot_error_model_qqplot(df, [1, 2])
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_qqplot_single_replicate_graceful(self):
        from pydimsum.report.plots import plot_error_model_qqplot

        df = pl.DataFrame({
            "fitness1_uncorr": [0.1, 0.2, 0.3],
            "sigma1_uncorr": [0.05, 0.05, 0.05],
        })
        png = plot_error_model_qqplot(df, [1])
        assert isinstance(png, bytes)

    def test_qqplot_no_error_model_rows(self):
        """All error_model=False rows should produce a fallback plot without crashing."""
        from pydimsum.report.plots import plot_error_model_qqplot

        rng = np.random.default_rng(1)
        n = 20
        df = pl.DataFrame({
            "error_model": [False] * n,
            "fitness1_uncorr": rng.standard_normal(n).tolist(),
            "sigma1_uncorr": (rng.random(n) * 0.1 + 0.05).tolist(),
            "fitness2_uncorr": rng.standard_normal(n).tolist(),
            "sigma2_uncorr": (rng.random(n) * 0.1 + 0.05).tolist(),
        })
        png = plot_error_model_qqplot(df, [1, 2])
        assert isinstance(png, bytes)


# ---------------------------------------------------------------------------
# WRAP stats parsers
# ---------------------------------------------------------------------------

class TestVsearchReportParser:
    def _write_report(self, path, stats: dict) -> None:
        lines = [
            "Merged length distribution:",
            f"\t {stats.get('min', 50)}  Min",
            f"\t {stats.get('low', 95)}  Low quartile",
            f"\t {stats.get('med', 100)}  Median",
            f"\t {stats.get('high', 105)}  High quartile",
            f"\t {stats.get('max', 120)}  Max",
            "",
            "Totals:",
            f"\t {stats['Pairs']}  Pairs",
            f"\t {stats['Merged']}  Merged",
            f"\t {stats['Too_short']}  Too short",
            f"\t {stats['No_alignment_found']}  No alignment found",
            f"\t {stats['Too_many_diffs']}  Too many diffs",
            f"\t {stats['Overlap_too_short']}  Overlap too short",
            f"\t {stats['Exp.errs._too_high']}  Exp.errs. too high",
            f"\t {stats['Min_Q_too_low']}  Min Q too low",
        ]
        path.write_text("\n".join(lines) + "\n")

    def test_parses_all_fields(self, tmp_path):
        from pydimsum.report.html import _parse_vsearch_report_file
        report = tmp_path / "sample.report"
        expected = {
            "Pairs": 10000, "Merged": 8500, "Too_short": 200,
            "No_alignment_found": 300, "Too_many_diffs": 500,
            "Overlap_too_short": 100, "Exp.errs._too_high": 250,
            "Min_Q_too_low": 150,
        }
        self._write_report(report, expected)
        result = _parse_vsearch_report_file(report)
        for key, val in expected.items():
            assert result[key] == val, f"{key}: expected {val}, got {result[key]}"

    def test_missing_file_returns_zeros(self, tmp_path):
        from pydimsum.report.html import _parse_vsearch_report_file
        result = _parse_vsearch_report_file(tmp_path / "nonexistent.report")
        assert result["Pairs"] == 0
        assert result["Merged"] == 0

    def test_collect_vsearch_stats_empty_dir(self, tmp_path):
        from pydimsum.report.html import _collect_vsearch_stats
        assert _collect_vsearch_stats(tmp_path) == []

    def test_collect_vsearch_stats_finds_reports(self, tmp_path):
        from pydimsum.report.html import _collect_vsearch_stats
        align_dir = tmp_path / "4_align"
        align_dir.mkdir()
        report = align_dir / "sample1_e1_s0_b1_t1.report"
        self._write_report(report, {
            "Pairs": 5000, "Merged": 4500, "Too_short": 100,
            "No_alignment_found": 0, "Too_many_diffs": 0,
            "Overlap_too_short": 0, "Exp.errs._too_high": 0,
            "Min_Q_too_low": 0,
        })
        results = _collect_vsearch_stats(tmp_path)
        assert len(results) == 1
        assert results[0]["Pairs"] == 5000
        assert results[0]["Merged"] == 4500

    def test_prefilter_reports_excluded(self, tmp_path):
        from pydimsum.report.html import _collect_vsearch_stats
        align_dir = tmp_path / "4_align"
        align_dir.mkdir()
        for name in ["sample1.report", "sample1.report.prefilter"]:
            r = align_dir / name
            self._write_report(r, {
                "Pairs": 100, "Merged": 90, "Too_short": 5,
                "No_alignment_found": 0, "Too_many_diffs": 0,
                "Overlap_too_short": 0, "Exp.errs._too_high": 0,
                "Min_Q_too_low": 0,
            })
        results = _collect_vsearch_stats(tmp_path)
        assert len(results) == 1  # prefilter excluded


class TestCutadaptStatsParser:
    _STDOUT = """\
This is cutadapt 4.4 with Python 3.11.5
Command line parameters: ...
Processing paired-end reads on 1 core ...

=== Summary ===

Total read pairs processed:          10,000
  Read 1 with adapter:               8,500 (85.0%)
  Read 2 with adapter:               8,300 (83.0%)

Pairs written (passing filters):      7,900 (79.0%)
"""

    def test_parses_total_reads(self, tmp_path):
        from pydimsum.report.html import _parse_cutadapt_stdout
        f = tmp_path / "sample.stdout"
        f.write_text(self._STDOUT)
        result = _parse_cutadapt_stdout(f)
        assert result["total_reads"] == 10000

    def test_parses_pairs_written(self, tmp_path):
        from pydimsum.report.html import _parse_cutadapt_stdout
        f = tmp_path / "sample.stdout"
        f.write_text(self._STDOUT)
        result = _parse_cutadapt_stdout(f)
        assert result["pairs_written"] == 7900

    def test_missing_file_returns_zeros(self, tmp_path):
        from pydimsum.report.html import _parse_cutadapt_stdout
        result = _parse_cutadapt_stdout(tmp_path / "nonexistent.stdout")
        assert result["total_reads"] == 0

    def test_collect_cutadapt_stats_empty_dir(self, tmp_path):
        from pydimsum.report.html import _collect_cutadapt_stats
        assert _collect_cutadapt_stats(tmp_path) == []

    def test_collect_cutadapt_stats_finds_files(self, tmp_path):
        from pydimsum.report.html import _collect_cutadapt_stats
        trim_dir = tmp_path / "3_trim"
        trim_dir.mkdir()
        (trim_dir / "sample1.cutadapt.gz.stdout").write_text(self._STDOUT)
        results = _collect_cutadapt_stats(tmp_path)
        assert len(results) == 1
        assert results[0]["total_reads"] == 10000
        assert results[0]["sample_name"] == "sample1"
