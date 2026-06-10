"""Stage 3b: Count unique aligned read sequences using starcode.

Mirrors: R/dimsum_stage_unique.R

starcode is run with -d 0 (exact match only) and -s (sorted output) to produce
a tab-separated file of (sequence, count) pairs, which is the input format
expected by pydimsum.steam.merge.build_variant_table.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import polars as pl

from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)


def run_tally(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    outpath: Path,
) -> pl.DataFrame:
    """Count unique sequences per sample using starcode (stage 3b).

    Each sample's merged FASTQ (from align.py) is piped through
    ``starcode -d 0 -s`` to count exact-match sequences.  The resulting
    tab-separated file has three columns (seq, count, cluster_members) but only
    the first two are needed by the merge stage.

    Parameters
    ----------
    config:
        Pipeline configuration.
    exp_design_df:
        Experiment design DataFrame with aligned_pair and aligned_pair_directory
        columns (written by :func:`~pydimsum.wrap.align.run_align`).
    outpath:
        Directory for starcode output files.

    Returns
    -------
    Updated exp_design_df with aligned_pair_unique and
    aligned_pair_unique_directory columns.
    """
    outpath.mkdir(parents=True, exist_ok=True)
    logger.info("=== Stage 3b (WRAP): COUNT UNIQUE VARIANTS ===")

    # sample_code = aligned_pair without the .split* suffix (handles split
    # FASTQ files; for simple cases sample_code == aligned_pair)
    rows = exp_design_df.to_dicts()

    # Determine the unique sample codes to avoid rerunning on split duplicates
    sample_code_seen: set[str] = set()
    unique_results: dict[str, str] = {}  # sample_code → .vsearch.unique filename

    for row in rows:
        aligned_pair = row["aligned_pair"]
        adir = Path(row["aligned_pair_directory"])
        input_fastq = adir / aligned_pair

        # sample_code strips ".split{N}" suffixes (mirrors R's strsplit on ".split"))
        sample_code = aligned_pair.split(".split")[0]

        if sample_code in sample_code_seen:
            continue
        sample_code_seen.add(sample_code)

        output_name = sample_code.replace(".vsearch.gz", ".vsearch.unique")
        output_file = outpath / output_name
        stdout_path = outpath / f"{output_name}.stdout"
        stderr_path = outpath / f"{output_name}.stderr"

        logger.info("  Counting unique reads: %s → %s", aligned_pair, output_name)

        # gunzip -c {input} | starcode -d 0 -s -t {cores} -o {output}
        # Use subprocess.Popen to pipe gunzip stdout into starcode stdin
        unzip_cmd = ["gunzip", "-c", str(input_fastq)]
        starcode_cmd = [
            "starcode",
            "-d", "0",
            "-s",
            "-t", str(config.num_cores),
            "-o", str(output_file),
        ]

        with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
            unzip = subprocess.Popen(unzip_cmd, stdout=subprocess.PIPE)
            starcode = subprocess.Popen(
                starcode_cmd,
                stdin=unzip.stdout,
                stdout=fout,
                stderr=ferr,
            )
            if unzip.stdout:
                unzip.stdout.close()  # allow unzip to receive SIGPIPE if starcode exits early
            starcode.wait()
            unzip.wait()

        if starcode.returncode != 0:
            raise RuntimeError(
                f"starcode failed (exit {starcode.returncode}). See {stderr_path}"
            )

        # Drop the 3rd column (cluster members) — only seq+count needed
        _trim_starcode_output(output_file)

        unique_results[sample_code] = output_name

    # Update exp_design_df: aligned_pair_unique = sample_code's output file
    unique_names = []
    for row in rows:
        sc = row["aligned_pair"].split(".split")[0]
        unique_names.append(unique_results.get(sc, ""))

    exp_design_df = exp_design_df.with_columns([
        pl.Series("aligned_pair_unique", unique_names),
        pl.lit(str(outpath)).alias("aligned_pair_unique_directory"),
    ])

    return exp_design_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trim_starcode_output(path: Path) -> None:
    """Keep only columns 1 and 2 (sequence, count) from starcode output.

    starcode outputs: sequence \\t count \\t cluster_members
    We need only: sequence \\t count
    (Matches the format expected by dimsum_stage_merge.R / pydimsum merge.py)
    """
    lines_out: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                lines_out.append(f"{parts[0]}\t{parts[1]}\n")
            else:
                lines_out.append(line + "\n")

    with open(path, "w") as fh:
        fh.writelines(lines_out)
