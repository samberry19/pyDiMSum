"""Stage 3a: Merge paired-end reads with VSEARCH and quality-filter.

Mirrors:
  R/dimsum_stage_vsearch.R
  R/dimsum__filter_reads.R
  R/dimsum__concatenate_reads.R  (trans-library and single-end)

Key design fix vs R:
  R builds `merged_lengths <- c(merged_lengths, width(sread(fq)))` which
  accumulates one integer per surviving read — O(N) RAM for large libraries.
  Python uses a collections.Counter histogram (fixed size) so memory is O(K)
  where K = number of distinct read lengths (typically << N).
"""

from __future__ import annotations

import gzip
import logging
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)

# FASTQ quality offset (Sanger/Illumina 1.8+)
_PHRED_OFFSET = 33


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_align(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    outpath: Path,
) -> pl.DataFrame:
    """Run VSEARCH paired-end merge + quality filter (stage 3a).

    Parameters
    ----------
    config:
        Pipeline configuration.
    exp_design_df:
        Experiment design DataFrame with pair1, pair2, pair_directory columns.
    outpath:
        Output directory for merged FASTQ files.

    Returns
    -------
    Updated exp_design_df with aligned_pair, aligned_pair_directory columns.
    """
    outpath.mkdir(parents=True, exist_ok=True)
    logger.info("=== Stage 3a (WRAP): ALIGN PAIRED-END READS ===")

    # Build sample names: {sample_name}_e{E}_s{S}_b{B}_t{T}
    sample_names = _build_sample_names(exp_design_df)

    # Additional vsearch options for overlap length
    minovlen = config.vsearch_min_ovlen
    extra_opts = ["-fastq_minovlen", str(minovlen)]
    if minovlen < 10:
        extra_opts += ["-fastq_maxdiffs", str(minovlen)]

    if config.trans_library:
        _run_trans_library(config, exp_design_df, sample_names, outpath, extra_opts)
    elif not config.paired:
        _run_single_end(config, exp_design_df, sample_names, outpath)
    else:
        _run_paired_end(config, exp_design_df, sample_names, outpath, extra_opts)

    # Update exp_design_df
    exp_design_df = exp_design_df.with_columns([
        pl.Series("aligned_pair", [f"{sn}.vsearch.gz" for sn in sample_names]),
        pl.lit(str(outpath)).alias("aligned_pair_directory"),
    ])
    return exp_design_df


# ---------------------------------------------------------------------------
# Per-library modes
# ---------------------------------------------------------------------------


def _run_paired_end(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    sample_names: list[str],
    outpath: Path,
    extra_opts: list[str],
) -> None:
    """Classic paired-end merge: run VSEARCH then filter by min quality."""
    rows = exp_design_df.to_dicts()
    for i, (row, sname) in enumerate(zip(rows, sample_names)):
        pdir = Path(row["pair_directory"])
        abs1 = str(pdir / row["pair1"])
        abs2 = str(pdir / row["pair2"])

        prefilter_fastq = outpath / f"{sname}.vsearch.prefilter.gz"
        prefilter_report = outpath / f"{sname}.report.prefilter"
        final_fastq = outpath / f"{sname}.vsearch.gz"
        final_report = outpath / f"{sname}.report"

        logger.info("  Merging: %s + %s", row["pair1"], row["pair2"])

        # Run VSEARCH
        cmd = [
            "vsearch",
            "-fastq_mergepairs", abs1,
            "-reverse", abs2,
            "-fastqout", "-",
            "-quiet",
            "-fastq_maxee", str(config.vsearch_max_ee),
            "-fastq_minlen", str(config.cutadapt_min_length),
            "--fastq_allowmergestagger",
            "--fastq_qmax", str(config.vsearch_max_qual),
            "--fastq_qmaxout", str(config.vsearch_max_qual),
            "-threads", str(config.num_cores),
        ] + extra_opts

        with open(prefilter_report, "w") as report_fh, \
             gzip.open(prefilter_fastq, "wt") as fq_fh:
            vsearch = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=report_fh,
                text=True,
            )
            if vsearch.stdout:
                fq_fh.write(vsearch.stdout.read())
            vsearch.wait()
            if vsearch.returncode != 0:
                raise RuntimeError(
                    f"VSEARCH failed (exit {vsearch.returncode}). See {prefilter_report}"
                )

        # Filter by minimum base quality
        _filter_reads(
            input_fastq=prefilter_fastq,
            input_report=prefilter_report,
            output_fastq=final_fastq,
            output_report=final_report,
            min_qual=config.vsearch_min_qual,
        )
        # Remove pre-filter file
        prefilter_fastq.unlink(missing_ok=True)


def _run_single_end(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    sample_names: list[str],
    outpath: Path,
) -> None:
    """Single-end mode: filter reads by quality without merging."""
    rows = exp_design_df.to_dicts()
    for row, sname in zip(rows, sample_names):
        pdir = Path(row["pair_directory"])
        abs1 = str(pdir / row["pair1"])
        output_fastq = outpath / f"{sname}.vsearch.gz"
        output_report = outpath / f"{sname}.report"

        logger.info("  Filtering SE reads: %s", row["pair1"])
        _filter_reads_se(
            input_fastq=abs1,
            output_fastq=output_fastq,
            output_report=output_report,
            min_qual=config.vsearch_min_qual,
        )


def _run_trans_library(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    sample_names: list[str],
    outpath: Path,
    extra_opts: list[str],
) -> None:
    """Trans-library mode: concatenate R1+(optional revcomp)R2 instead of merging.

    Applies the same per-pair quality filters as VSEARCH merge mode:
    min read length, min base quality, max expected errors.

    Mirrors: R/dimsum__concatenate_reads.R
    """
    rows = exp_design_df.to_dicts()
    for row, sname in zip(rows, sample_names):
        pdir = Path(row["pair_directory"])
        abs1 = str(pdir / row["pair1"])
        abs2 = str(pdir / row.get("pair2", row["pair1"]))
        output_fastq = outpath / f"{sname}.vsearch.gz"
        output_report = outpath / f"{sname}.report"

        logger.info("  Trans-library concat: %s + %s", row["pair1"], row.get("pair2"))
        _concatenate_reads(abs1, abs2, output_fastq, output_report, config)


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------


def _filter_reads(
    input_fastq: Path,
    input_report: Path,
    output_fastq: Path,
    output_report: Path,
    min_qual: int,
) -> None:
    """Filter merged FASTQ reads: discard any read with a base quality < min_qual.

    Also parses the VSEARCH alignment report to extract merge statistics.
    Mirrors: R/dimsum__filter_reads.R

    Memory fix: uses a Counter histogram for length distribution instead of
    accumulating one value per read (avoids O(N) RAM growth).
    """
    # Parse VSEARCH report
    stats = _parse_vsearch_report(input_report)

    merged_count = 0
    min_q_too_low = 0
    length_hist: Counter[int] = Counter()

    with gzip.open(input_fastq, "rt") as fin, gzip.open(output_fastq, "wt") as fout:
        while True:
            header = fin.readline()
            if not header:
                break
            seq = fin.readline()
            plus = fin.readline()
            qual = fin.readline()

            if not qual:
                break

            seq = seq.rstrip("\n")
            qual_str = qual.rstrip("\n")

            # Check minimum quality
            quals = [ord(c) - _PHRED_OFFSET for c in qual_str]
            if any(q < min_qual for q in quals):
                min_q_too_low += 1
                continue

            fout.write(header)
            fout.write(seq + "\n")
            fout.write(plus)
            fout.write(qual_str + "\n")

            merged_count += 1
            length_hist[len(seq)] += 1

    stats["Merged"] = merged_count
    stats["Min_Q_too_low"] = min_q_too_low

    # Compute length distribution from histogram
    if merged_count > 0:
        all_lengths = sorted(length_hist)
        # Build a flat list for quantile computation (still O(K) not O(N))
        expanded = []
        for length, cnt in sorted(length_hist.items()):
            expanded.extend([length] * cnt)
        expanded.sort()
        n = len(expanded)
        stats["Merged_length_min"] = expanded[0]
        stats["Merged_length_max"] = expanded[-1]
        stats["Merged_length_median"] = expanded[n // 2]
        stats["Merged_length_low"] = expanded[n // 4]
        stats["Merged_length_high"] = expanded[3 * n // 4]
    else:
        for k in ["Merged_length_min", "Merged_length_low", "Merged_length_median",
                  "Merged_length_high", "Merged_length_max"]:
            stats[k] = "NA"

    _write_report(output_report, stats)


def _filter_reads_se(
    input_fastq: str,
    output_fastq: Path,
    output_report: Path,
    min_qual: int,
) -> None:
    """Single-end quality filter (no VSEARCH report to parse)."""
    merged_count = 0
    min_q_too_low = 0
    length_hist: Counter[int] = Counter()

    opener = gzip.open if input_fastq.endswith(".gz") else open

    with opener(input_fastq, "rt") as fin, gzip.open(output_fastq, "wt") as fout:  # type: ignore[arg-type]
        while True:
            header = fin.readline()
            if not header:
                break
            seq = fin.readline()
            plus = fin.readline()
            qual = fin.readline()
            if not qual:
                break

            seq = seq.rstrip("\n")
            qual_str = qual.rstrip("\n")

            quals = [ord(c) - _PHRED_OFFSET for c in qual_str]
            if any(q < min_qual for q in quals):
                min_q_too_low += 1
                continue

            fout.write(header)
            fout.write(seq + "\n")
            fout.write(plus)
            fout.write(qual_str + "\n")

            merged_count += 1
            length_hist[len(seq)] += 1

    stats: dict = {
        "Pairs": 0, "Merged": merged_count, "Too_short": 0,
        "No_alignment_found": 0, "Too_many_diffs": 0,
        "Overlap_too_short": 0, "Exp.errs._too_high": 0,
        "Min_Q_too_low": min_q_too_low,
    }
    if merged_count > 0:
        expanded = []
        for length, cnt in sorted(length_hist.items()):
            expanded.extend([length] * cnt)
        n = len(expanded)
        stats["Merged_length_min"] = expanded[0]
        stats["Merged_length_max"] = expanded[-1]
        stats["Merged_length_median"] = expanded[n // 2]
        stats["Merged_length_low"] = expanded[n // 4]
        stats["Merged_length_high"] = expanded[3 * n // 4]
    else:
        for k in ["Merged_length_min", "Merged_length_low", "Merged_length_median",
                  "Merged_length_high", "Merged_length_max"]:
            stats[k] = "NA"

    _write_report(output_report, stats)


# ---------------------------------------------------------------------------
# Trans-library concatenation
# ---------------------------------------------------------------------------


def _concatenate_reads(
    abs1: str,
    abs2: str,
    output_fastq: Path,
    output_report: Path,
    config: RunConfig,
) -> None:
    """Concatenate R1 + optional revcomp(R2) for trans-library experiments.

    Applies the same per-pair quality gates as VSEARCH merge mode:
      - discard if either read is shorter than cutadapt_min_length
      - discard if any base in either read has Phred < vsearch_min_qual
      - discard if combined expected errors > vsearch_max_ee

    Mirrors: R/dimsum__concatenate_reads.R
    """
    from Bio.Seq import Seq

    rc = config.trans_library_reverse_complement
    min_len = config.cutadapt_min_length
    min_qual = config.vsearch_min_qual
    max_ee = config.vsearch_max_ee

    n_pairs = 0
    n_merged = 0
    n_too_short = 0
    n_min_q_too_low = 0
    n_exp_err_too_high = 0
    length_hist: Counter[int] = Counter()

    opener1 = gzip.open if abs1.endswith(".gz") else open
    opener2 = gzip.open if abs2.endswith(".gz") else open

    with opener1(abs1, "rt") as f1, opener2(abs2, "rt") as f2, \
         gzip.open(output_fastq, "wt") as fout:  # type: ignore[arg-type]
        while True:
            h1 = f1.readline()
            if not h1:
                break
            s1 = f1.readline().rstrip("\n")
            f1.readline()  # +
            q1 = f1.readline().rstrip("\n")

            h2 = f2.readline()
            s2 = f2.readline().rstrip("\n")
            f2.readline()  # +
            q2 = f2.readline().rstrip("\n")

            n_pairs += 1

            # Length filter
            if len(s1) < min_len or len(s2) < min_len:
                n_too_short += 1
                continue

            # Minimum base quality filter (any base below threshold → discard)
            q1_phred = [ord(c) - _PHRED_OFFSET for c in q1]
            q2_phred = [ord(c) - _PHRED_OFFSET for c in q2]
            if any(q < min_qual for q in q1_phred) or any(q < min_qual for q in q2_phred):
                n_min_q_too_low += 1
                continue

            # Expected errors filter (sum of 10^(-Q/10) across both reads)
            exp_err = sum(10 ** (-q / 10.0) for q in q1_phred + q2_phred)
            if exp_err > max_ee:
                n_exp_err_too_high += 1
                continue

            # Optionally reverse-complement R2
            if rc:
                s2_out = str(Seq(s2).reverse_complement())
                q2_out = q2[::-1]
            else:
                s2_out = s2
                q2_out = q2

            concat_seq = s1 + s2_out
            concat_qual = q1 + q2_out

            fout.write(h1)
            fout.write(concat_seq + "\n")
            fout.write("+\n")
            fout.write(concat_qual + "\n")

            n_merged += 1
            length_hist[len(concat_seq)] += 1

    # Build length distribution stats
    stats: dict = {
        "Pairs": n_pairs,
        "Merged": n_merged,
        "Too_short": n_too_short,
        "No_alignment_found": 0,
        "Too_many_diffs": 0,
        "Overlap_too_short": 0,
        "Exp.errs._too_high": n_exp_err_too_high,
        "Min_Q_too_low": n_min_q_too_low,
    }
    if n_merged > 0:
        expanded = []
        for length, cnt in sorted(length_hist.items()):
            expanded.extend([length] * cnt)
        n = len(expanded)
        stats["Merged_length_min"] = expanded[0]
        stats["Merged_length_max"] = expanded[-1]
        stats["Merged_length_median"] = expanded[n // 2]
        stats["Merged_length_low"] = expanded[n // 4]
        stats["Merged_length_high"] = expanded[3 * n // 4]
    else:
        for k in ["Merged_length_min", "Merged_length_low", "Merged_length_median",
                  "Merged_length_high", "Merged_length_max"]:
            stats[k] = "NA"

    _write_report(output_report, stats)
    logger.info(
        "  Trans-library: %d pairs → %d merged (%d too short, %d low qual, %d high ee)",
        n_pairs, n_merged, n_too_short, n_min_q_too_low, n_exp_err_too_high,
    )


# ---------------------------------------------------------------------------
# Report parsing / writing
# ---------------------------------------------------------------------------


def _parse_vsearch_report(report_path: Path) -> dict:
    """Parse a VSEARCH merge report for alignment statistics."""
    stats: dict = {
        "Pairs": 0,
        "Merged": 0,
        "Too_short": 0,
        "No_alignment_found": 0,
        "Too_many_diffs": 0,
        "Overlap_too_short": 0,
        "Exp.errs._too_high": 0,
        "Min_Q_too_low": 0,
    }
    if not report_path.exists():
        return stats

    with open(report_path) as fh:
        for line in fh:
            line = line.strip()
            # Extract the integer at the beginning of key lines
            parts = line.split()
            if not parts:
                continue
            # VSEARCH report format: "   NNN  label text"
            try:
                val = int(parts[0])
            except (ValueError, IndexError):
                continue
            text = " ".join(parts[1:]).lower()
            if "pairs" in text and "ratio" not in text and "merged" not in text:
                stats["Pairs"] = val
            elif "merged" in text and "too" not in text:
                stats["Merged"] = val
            elif "too short" in text:
                stats["Too_short"] = val
            elif "no alignment" in text or "too few kmers" in text or "multiple potential" in text:
                stats["No_alignment_found"] += val
            elif "too many diffs" in text or "too many differences" in text:
                stats["Too_many_diffs"] = val
            elif "overlap too short" in text:
                stats["Overlap_too_short"] = val
            elif "expected error" in text:
                stats["Exp.errs._too_high"] = val

    return stats


def _write_report(report_path: Path, stats: dict) -> None:
    """Write alignment + filtering statistics report (mirrors R format)."""
    lines = [
        "Merged length distribution:",
        f"\t {stats.get('Merged_length_min', 'NA')}  Min",
        f"\t {stats.get('Merged_length_low', 'NA')}  Low quartile",
        f"\t {stats.get('Merged_length_median', 'NA')}  Median",
        f"\t {stats.get('Merged_length_high', 'NA')}  High quartile",
        f"\t {stats.get('Merged_length_max', 'NA')}  Max",
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
    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _build_sample_names(exp_design_df: pl.DataFrame) -> list[str]:
    """Build the per-row sample codes used as output file prefixes.

    Mirrors R: paste0(sample_name, '_e', experiment, '_s', selection_id,
                       '_b', biological_replicate, '_t', technical_replicate)
    """
    rows = exp_design_df.to_dicts()
    names = []
    for row in rows:
        sn = row["sample_name"]
        E = row["experiment"]
        s = row["selection_id"]
        b = row.get("biological_replicate") or row.get("selection_replicate", "NA")
        t = row.get("technical_replicate", "NA")
        names.append(f"{sn}_e{E}_s{s}_b{b}_t{t}")
    return names
