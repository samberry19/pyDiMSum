"""Stage 1: Run FastQC on all FASTQ files.

Mirrors: R/dimsum_stage_fastqc.R
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import polars as pl

from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)


def run_fastqc(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    outpath: Path,
) -> pl.DataFrame:
    """Run FastQC on all FASTQ files referenced in exp_design_df (stage 1).

    Parameters
    ----------
    config:
        Pipeline configuration.
    exp_design_df:
        Experiment design DataFrame; must have pair1, pair2, pair_directory columns.
    outpath:
        Directory to write FastQC HTML reports and extracted text data.

    Returns
    -------
    Updated exp_design_df with pair1_fastqc, pair2_fastqc, fastqc_directory columns.
    """
    outpath.mkdir(parents=True, exist_ok=True)
    logger.info("=== Stage 1 (WRAP): ASSESS READ QUALITY ===")

    # Collect all unique FASTQ paths
    pair_dirs = exp_design_df["pair_directory"].to_list()
    pair1s = exp_design_df["pair1"].to_list()
    pair2s = exp_design_df["pair2"].to_list() if "pair2" in exp_design_df.columns else []

    all_fastq: list[str] = []
    seen: set[str] = set()
    for i, (pdir, p1) in enumerate(zip(pair_dirs, pair1s)):
        f1 = str(Path(pdir) / p1)
        if f1 not in seen:
            all_fastq.append(f1)
            seen.add(f1)
        if pair2s and pair2s[i] and pair2s[i] != p1:
            f2 = str(Path(pdir) / pair2s[i])
            if f2 not in seen:
                all_fastq.append(f2)
                seen.add(f2)

    if not all_fastq:
        logger.warning("  No FASTQ files found in experiment design.")
        return exp_design_df

    logger.info("  Running FastQC on %d files...", len(all_fastq))

    stdout_path = outpath / "fastqc.stdout"
    stderr_path = outpath / "fastqc.stderr"

    cmd = [
        "fastqc",
        "-o", str(outpath),
        "--extract",
        "-t", str(config.num_cores),
    ] + all_fastq

    result = subprocess.run(
        cmd,
        stdout=open(stdout_path, "w"),
        stderr=open(stderr_path, "w"),
    )
    if result.returncode != 0:
        logger.warning(
            "FastQC exited with code %d — see %s", result.returncode, stderr_path
        )

    # Derive FastQC report paths: {basename_without_ext}_fastqc/fastqc_data.txt
    ext = config.fastq_file_extension
    gzipped = config.gzipped

    def _fastqc_path(filename: str) -> str:
        name = filename
        if gzipped and name.endswith(".gz"):
            name = name[:-3]
        if name.endswith(ext):
            name = name[: -len(ext)]
        return f"{name}_fastqc/fastqc_data.txt"

    p1_fastqc = [_fastqc_path(p1) for p1 in pair1s]
    p2_fastqc = [_fastqc_path(p2) if p2 else None for p2 in pair2s] if pair2s else [None] * len(pair1s)

    exp_design_df = exp_design_df.with_columns([
        pl.Series("pair1_fastqc", p1_fastqc),
        pl.Series("pair2_fastqc", p2_fastqc),
        pl.lit(str(outpath)).alias("fastqc_directory"),
    ])

    logger.info("  FastQC complete. Results in: %s", outpath)
    return exp_design_df
