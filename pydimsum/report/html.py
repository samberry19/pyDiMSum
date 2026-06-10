"""HTML report generator for pyDiMSum runs.

Produces a self-contained HTML file with embedded base64 plot images.
Requires ``jinja2`` and ``matplotlib`` (both soft-optional — if missing,
the pipeline continues and just logs a warning).
"""

from __future__ import annotations

import base64
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)


def generate_report(
    stats: dict,
    variant_df: "pl.DataFrame",
    fitness_df: "pl.DataFrame",
    singles_df: "pl.DataFrame | None",
    replicates: list[int],
    count_cols: list[str],
    config: "RunConfig",
    output_path: Path,
) -> Path:
    """Render the HTML report and write it to *output_path*.

    Parameters
    ----------
    stats:
        Dict from ``process_variants._compute_stats`` with keys like
        ``nuc_subst_dict``, ``nuc_indel_dict``, etc.
    variant_df:
        Full count table (all variants, before filtering) — used for count
        distribution / scatter plots.
    fitness_df:
        Post-fitness DataFrame with per-replicate ``fitness{E}_uncorr``
        columns — used for replicate scatter.
    singles_df:
        Merged singles table (may be None or empty).
    replicates:
        List of replicate integers.
    count_cols:
        List of count column names (e.g. ``["count_e1_s0", "count_e1_s1", …]``).
    config:
        Validated RunConfig.
    output_path:
        Directory where ``{project_name}_report.html`` is written.

    Returns
    -------
    Path to the written HTML file.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        logger.warning("jinja2 not installed — skipping HTML report generation")
        raise

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logger.warning("matplotlib not installed — skipping HTML report generation")
        raise

    from pydimsum.report.plots import (
        _png_to_data_uri,
        plot_count_distributions,
        plot_count_scatter_matrix,
        plot_cutadapt_stats,
        plot_error_model_qqplot,
        plot_replicate_fitness_scatter,
        plot_variant_processing,
        plot_variant_processing_pct,
        plot_vsearch_stats,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Generate plots ----
    processing_counts_plot = ""
    processing_pct_plot = ""
    try:
        processing_counts_plot = _png_to_data_uri(
            plot_variant_processing(stats, count_cols)
        )
        processing_pct_plot = _png_to_data_uri(
            plot_variant_processing_pct(stats, count_cols)
        )
    except Exception as exc:
        logger.warning("Could not generate processing plots: %s", exc)

    count_dist_plot = ""
    try:
        count_dist_plot = _png_to_data_uri(
            plot_count_distributions(variant_df, count_cols)
        )
    except Exception as exc:
        logger.warning("Could not generate count distribution plot: %s", exc)

    count_scatter_plot = ""
    try:
        count_scatter_plot = _png_to_data_uri(
            plot_count_scatter_matrix(variant_df, count_cols)
        )
    except Exception as exc:
        logger.warning("Could not generate count scatter plot: %s", exc)

    fitness_scatter_plot = ""
    try:
        if fitness_df is not None and len(fitness_df) > 0:
            fitness_scatter_plot = _png_to_data_uri(
                plot_replicate_fitness_scatter(fitness_df, replicates)
            )
    except Exception as exc:
        logger.warning("Could not generate fitness scatter plot: %s", exc)

    qqplot = ""
    try:
        if fitness_df is not None and len(fitness_df) > 0 and len(replicates) >= 2:
            qqplot = _png_to_data_uri(plot_error_model_qqplot(fitness_df, replicates))
    except Exception as exc:
        logger.warning("Could not generate QQ plot: %s", exc)

    # ---- WRAP stats (only when a full WRAP run was performed) ----
    vsearch_stats_plot = ""
    cutadapt_stats_plot = ""
    if getattr(config, "count_path", None) is None:
        try:
            vsearch_sample_stats = _collect_vsearch_stats(output_path)
            if vsearch_sample_stats:
                vsearch_stats_plot = _png_to_data_uri(plot_vsearch_stats(vsearch_sample_stats))
        except Exception as exc:
            logger.warning("Could not generate VSEARCH stats plot: %s", exc)

        try:
            cutadapt_sample_stats = _collect_cutadapt_stats(output_path)
            if cutadapt_sample_stats:
                cutadapt_stats_plot = _png_to_data_uri(plot_cutadapt_stats(cutadapt_sample_stats))
        except Exception as exc:
            logger.warning("Could not generate cutadapt stats plot: %s", exc)

    # ---- Build stats summary table ----
    stats_table = _build_stats_table(stats, count_cols)

    # ---- Build settings table ----
    settings = _collect_settings(config)

    # ---- Context ----
    n_retained = len(variant_df) if variant_df is not None else 0
    n_input = int(stats.get("n_input", n_retained))

    # Detect mode
    mode = "enrichment" if getattr(config, "enrichment_mode", False) else "mutation"

    ctx = {
        "project_name": config.project_name,
        "output_path": str(output_path),
        "run_date": date.today().isoformat(),
        "mode": mode,
        "replicates": replicates,
        "n_count_cols": len(count_cols),
        "sequence_type": getattr(config, "sequence_type_resolved", config.sequence_type),
        "wt_seq": getattr(config, "wt_nt_seq", "") or "",
        "n_input_variants": n_input,
        "n_retained_variants": n_retained,
        "processing_counts_plot": processing_counts_plot,
        "processing_pct_plot": processing_pct_plot,
        "count_dist_plot": count_dist_plot,
        "count_scatter_plot": count_scatter_plot,
        "fitness_scatter_plot": fitness_scatter_plot,
        "qqplot": qqplot,
        "vsearch_stats_plot": vsearch_stats_plot,
        "cutadapt_stats_plot": cutadapt_stats_plot,
        "stats_table": stats_table,
        "settings": settings,
    }

    # ---- Render template ----
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
    template = env.get_template("report.html.j2")
    html = template.render(**ctx)

    out_file = output_path / f"{config.project_name}_report.html"
    out_file.write_text(html, encoding="utf-8")
    logger.info("HTML report written to: %s", out_file)
    return out_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_stats_table(stats: dict, count_cols: list[str]) -> dict | None:
    """Build a rows/cols dict suitable for the Jinja template."""
    if not stats:
        return None

    from pydimsum.report.plots import _short_name

    category_map = [
        ("0 substitutions",   "nuc_subst_dict", 0),
        ("1 substitution",    "nuc_subst_dict", 1),
        ("2 substitutions",   "nuc_subst_dict", 2),
        ("3+ substitutions",  "nuc_subst_dict", "3+"),
        ("Indels (retained)", "nuc_indel_dict", None),
        ("Mixed codon",       "nuc_mxsub_dict", None),
        ("Too many subs",     "nuc_tmsub_dict", None),
        ("Not permitted",     "nuc_frbdn_dict", None),
        ("Constant region",   "nuc_const_dict", None),
        ("Invalid barcode",   "nuc_nbarc_dict", None),
    ]

    short_cols = [_short_name(c) for c in count_cols]
    rows = []
    nuc_subst = stats.get("nuc_subst_dict", {})

    for label, key, sub_key in category_map:
        row = [label]
        for col in count_cols:
            if key == "nuc_subst_dict":
                d = nuc_subst.get(col, {})
                if sub_key == "3+":
                    total = sum(d.values())
                    val = max(0, total - d.get(0, 0) - d.get(1, 0) - d.get(2, 0))
                else:
                    val = d.get(sub_key, 0)
            else:
                val = stats.get(key, {}).get(col, 0)
            row.append(f"{int(val):,}" if val else "0")
        # Skip entirely-zero rows
        if any(v != "0" for v in row[1:]):
            rows.append(row)

    if not rows:
        return None

    return {"cols": short_cols, "rows": rows}


def _parse_vsearch_report_file(path: Path) -> dict:
    """Parse a pyDiMSum .report file (written by align._write_report)."""
    stats: dict = {
        "Pairs": 0, "Merged": 0, "Too_short": 0, "No_alignment_found": 0,
        "Too_many_diffs": 0, "Overlap_too_short": 0,
        "Exp.errs._too_high": 0, "Min_Q_too_low": 0,
    }
    try:
        text = path.read_text()
    except OSError:
        return stats
    for line in text.splitlines():
        line = line.strip()
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            val = int(parts[0])
        except ValueError:
            continue
        rest = " ".join(parts[1:]).lower()
        if "pairs" in rest:
            stats["Pairs"] = val
        elif "merged" in rest:
            stats["Merged"] = val
        elif "overlap too short" in rest:
            stats["Overlap_too_short"] = val
        elif "too short" in rest:
            stats["Too_short"] = val
        elif "no alignment" in rest:
            stats["No_alignment_found"] = val
        elif "too many diffs" in rest:
            stats["Too_many_diffs"] = val
        elif "exp.errs" in rest or "expected" in rest:
            stats["Exp.errs._too_high"] = val
        elif "min q" in rest:
            stats["Min_Q_too_low"] = val
    return stats


def _collect_vsearch_stats(project_path: Path) -> list[dict]:
    """Collect per-sample VSEARCH stats from {project_path}/4_align/*.report."""
    align_dir = project_path / "4_align"
    if not align_dir.is_dir():
        return []
    results = []
    for report_file in sorted(align_dir.glob("*.report")):
        if "prefilter" in report_file.name:
            continue
        stats = _parse_vsearch_report_file(report_file)
        # Use stem of the filename minus .report as sample label
        sample = report_file.stem
        stats["sample_name"] = sample
        results.append(stats)
    return results


def _parse_cutadapt_stdout(path: Path) -> dict:
    """Parse key statistics from cutadapt stdout file."""
    stats: dict = {
        "total_reads": 0, "read1_with_adapter": 0,
        "read2_with_adapter": 0, "pairs_written": 0,
    }
    try:
        text = path.read_text()
    except OSError:
        return stats
    for line in text.splitlines():
        line_stripped = line.strip()
        low = line_stripped.lower()
        # "Total read pairs processed:   N"  or  "Total reads processed:   N"
        if "total read" in low and "processed" in low:
            stats["total_reads"] = _extract_last_int(line_stripped)
        elif "read 1 with adapter" in low:
            stats["read1_with_adapter"] = _extract_last_int(line_stripped)
        elif "read 2 with adapter" in low:
            stats["read2_with_adapter"] = _extract_last_int(line_stripped)
        elif ("pairs written" in low or "reads written" in low) and "passing" in low:
            stats["pairs_written"] = _extract_last_int(line_stripped)
    return stats


def _extract_last_int(line: str) -> int:
    """Extract the largest integer from a line (ignoring commas).

    For lines like 'Pairs written: 7,900 (79.0%)' this correctly returns 7900
    (the count) rather than the small percentage-derived value.
    """
    import re
    tokens = re.findall(r"[\d,]+", line)
    best = 0
    for token in tokens:
        try:
            val = int(token.replace(",", ""))
            if val > best:
                best = val
        except ValueError:
            continue
    return best


def _collect_cutadapt_stats(project_path: Path) -> list[dict]:
    """Collect per-sample cutadapt stats from {project_path}/3_trim/*.stdout."""
    trim_dir = project_path / "3_trim"
    if not trim_dir.is_dir():
        return []
    results = []
    for stdout_file in sorted(trim_dir.glob("*.stdout")):
        stats = _parse_cutadapt_stdout(stdout_file)
        sample = stdout_file.name.split(".cutadapt")[0]
        stats["sample_name"] = sample
        results.append(stats)
    return results


def _collect_settings(config: "RunConfig") -> list[tuple[str, str]]:
    """Return a list of (label, value) pairs from the config for display."""
    items = [
        ("wildtype_sequence",        getattr(config, "wildtype_sequence", "") or ""),
        ("sequence_type",            str(getattr(config, "sequence_type", ""))),
        ("mixed_substitutions",      str(getattr(config, "mixed_substitutions", ""))),
        ("max_substitutions",        str(getattr(config, "max_substitutions", ""))),
        ("min_input_count",          str(getattr(config, "min_input_count", ""))),
        ("min_total_count",          str(getattr(config, "min_total_count", ""))),
        ("fitness_normalise",        str(getattr(config, "fitness_normalise", ""))),
        ("fitness_error_model",      str(getattr(config, "fitness_error_model", ""))),
        ("bayesian_double_fitness",  str(getattr(config, "bayesian_double_fitness", ""))),
        ("enrichment_mode",          str(getattr(config, "enrichment_mode", False))),
        ("enrichment_normalise",     str(getattr(config, "enrichment_normalise", ""))),
    ]
    return [(k, v) for k, v in items if v not in ("", "None", "False") or k in (
        "sequence_type", "enrichment_mode"
    )]
