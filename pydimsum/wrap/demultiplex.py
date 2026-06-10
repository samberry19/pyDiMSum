"""Stage 0: Demultiplex FASTQ files using cutadapt barcode detection.

Mirrors: R/dimsum_stage_demultiplex.R
         R/dimsum__demultiplex_helper.R
         R/dimsum__demultiplex_cp_helper.R

Only runs when a barcode design file is present in config.  If absent,
the experiment FASTQ files are assumed to already be demultiplexed and this
stage is a no-op.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_demultiplex(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    outpath: Path,
    barcode_design_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Run cutadapt-based barcode demultiplexing (stage 0).

    Parameters
    ----------
    config:
        Pipeline configuration.
    exp_design_df:
        Experiment design DataFrame (may be mutated — returns updated copy).
    outpath:
        Directory for demultiplex output files.
    barcode_design_df:
        Barcode design table (from config.barcode_design_path). If None, skip.

    Returns
    -------
    Updated exp_design_df with pair_directory pointing to outpath.
    """
    if barcode_design_df is None:
        logger.info("Stage 0: No barcode design — assuming files already demultiplexed.")
        return exp_design_df

    outpath.mkdir(parents=True, exist_ok=True)
    logger.info("=== Stage 0 (WRAP): DEMULTIPLEX READS ===")

    # Build unique input FASTQ pairs (may span multiple rows of barcode_design)
    pair_col = "pair_directory"
    pairs_seen: dict[tuple[str, str], int] = {}  # (abs_pair1, abs_pair2) → pair_index
    pair_list: list[dict] = []  # [{pair_idx, pair1, pair2, abs1, abs2}]

    for row in barcode_design_df.iter_rows(named=True):
        pdir = Path(row["pair_directory"])
        abs1 = str(pdir / row["pair1"])
        abs2 = str(pdir / row["pair2"]) if config.paired else abs1
        key = (abs1, abs2)
        if key not in pairs_seen:
            idx = len(pair_list)
            pairs_seen[key] = idx
            pair_list.append({"idx": idx, "pair1": row["pair1"], "pair2": row["pair2"],
                               "abs1": abs1, "abs2": abs2})

    # Step 1: Copy/rename FASTQ files if extension incompatible with cutadapt
    ext = config.fastq_file_extension
    if ext != ".fastq":
        logger.info("  Copying/renaming FASTQ files (extension %s → .fastq.gz)", ext)
        with ProcessPoolExecutor(max_workers=config.num_cores) as pool:
            futs = [
                pool.submit(_copy_rename_fastq, p, outpath, ext, config.gzipped, config.paired)
                for p in pair_list
            ]
            for f in as_completed(futs):
                f.result()  # re-raise on error
        # Update paths to point to renamed files in outpath
        for p in pair_list:
            p["abs1"] = str(outpath / _renamed_fastq(p["pair1"], ext, config.gzipped))
            p["abs2"] = str(outpath / _renamed_fastq(p["pair2"], ext, config.gzipped)) \
                if config.paired else p["abs1"]

    # Step 2: Write barcode FASTA files and run cutadapt per pair
    logger.info("  Demultiplexing %d FASTQ pair(s) with cutadapt...", len(pair_list))
    with ProcessPoolExecutor(max_workers=config.num_cores) as pool:
        futs = [
            pool.submit(
                _demultiplex_pair,
                p, barcode_design_df, outpath, config
            )
            for p in pair_list
        ]
        for f in as_completed(futs):
            f.result()

    # Update exp_design_df: pair_directory → outpath; extension → .fastq
    exp_design_df = exp_design_df.with_columns(
        pl.lit(str(outpath)).alias("pair_directory")
    )
    return exp_design_df


# ---------------------------------------------------------------------------
# Workers (run in subprocess pool — must be picklable)
# ---------------------------------------------------------------------------


def _copy_rename_fastq(
    pair: dict,
    outpath: Path,
    ext: str,
    gzipped: bool,
    paired: bool,
) -> None:
    """Copy/gzip-compress a FASTQ file into outpath with .fastq.gz extension."""
    for key in (["abs1"] if not paired else ["abs1", "abs2"]):
        src = Path(pair[key])
        dst_name = _renamed_fastq(src.name, ext, gzipped)
        dst = outpath / dst_name
        if gzipped:
            shutil.copy2(src, dst)
        else:
            # Compress plain FASTQ to .fastq.gz
            with open(src, "rb") as fin, gzip.open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout)


def _renamed_fastq(filename: str, ext: str, gzipped: bool) -> str:
    """Return filename with old extension replaced by .fastq.gz."""
    # Remove old ext + optional .gz
    name = filename
    if gzipped and name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(ext):
        name = name[: -len(ext)]
    return name + ".fastq.gz"


def _demultiplex_pair(
    pair: dict,
    barcode_design_df: pl.DataFrame,
    outpath: Path,
    config: RunConfig,
) -> None:
    """Write barcode FASTA files and run cutadapt for one FASTQ pair."""
    abs1, abs2 = pair["abs1"], pair["abs2"]
    idx = pair["idx"]

    # Filter barcode design to rows matching this FASTQ pair
    abs1_base = Path(abs1).name
    bdf_subset = barcode_design_df.filter(
        pl.col("pair1").str.ends_with(abs1_base.replace(".fastq.gz", ""))
        | (pl.col("pair1") == pair["pair1"])
    )

    # Write barcode FASTA files
    bc1_fasta = outpath / f"demultiplex_barcode1-file_{idx}.fasta"
    bc2_fasta = outpath / f"demultiplex_barcode2-file_{idx}.fasta"

    with open(bc1_fasta, "w") as fh:
        for row in bdf_subset.iter_rows(named=True):
            fh.write(f">{row['new_pair_prefix']}\n")
            fh.write(f"^{row['barcode1']}\n")

    if config.paired:
        with open(bc2_fasta, "w") as fh:
            for row in bdf_subset.iter_rows(named=True):
                fh.write(f">{row['new_pair_prefix']}\n")
                fh.write(f"^{row['barcode2']}\n")

    # Build cutadapt command
    stdout_path = outpath / f"{Path(abs1).name}.demultiplex.stdout"
    stderr_path = outpath / f"{Path(abs1).name}.demultiplex.stderr"

    cmd: list[str] = [
        "cutadapt",
        "-g", f"file:{bc1_fasta}",
        "-e", str(config.barcode_error_rate),
        "--no-indels",
        "--untrimmed-output",
        str(outpath / f"{Path(abs1).name}.demultiplex.unknown.fastq.gz"),
    ]
    if config.paired:
        cmd += [
            "-G", f"file:{bc2_fasta}",
            "--pair-adapters",
            "--untrimmed-paired-output",
            str(outpath / f"{Path(abs2).name}.demultiplex.unknown.fastq.gz"),
            "-o", str(outpath / "{name}1.fastq.gz"),
            "-p", str(outpath / "{name}2.fastq.gz"),
            abs1, abs2,
        ]
    else:
        cmd += [
            "-o", str(outpath / "{name}1.fastq.gz"),
            abs1,
        ]

    result = subprocess.run(
        cmd,
        stdout=open(stdout_path, "w"),
        stderr=open(stderr_path, "w"),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cutadapt demultiplex failed (exit {result.returncode}). "
            f"See {stderr_path}"
        )
