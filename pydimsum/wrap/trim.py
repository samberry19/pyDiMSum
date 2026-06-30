"""Stage 2: Trim constant flanking regions from FASTQ files using cutadapt.

Mirrors:
  R/dimsum_stage_cutadapt.R
  R/dimsum__get_cutadapt_options.R
  R/dimsum__convert_linked_adapters.R
  R/dimsum__swap_reads.R  (not stranded + paired mode)
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
from pathlib import Path

import polars as pl

from pydimsum.config import RunConfig

logger = logging.getLogger(__name__)

# Complement table for reverse-complement computation
_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ---------------------------------------------------------------------------
# Experiment-design column filling
# ---------------------------------------------------------------------------

_ADAPTER_COLS = [
    ("cutadapt_5_first",   "cutadapt5First"),
    ("cutadapt_5_second",  "cutadapt5Second"),
    ("cutadapt_3_first",   "cutadapt3First"),
    ("cutadapt_3_second",  "cutadapt3Second"),
]
_CUT_COLS = [
    ("cutadapt_cut_5_first",  "cutadaptCut5First"),
    ("cutadapt_cut_5_second", "cutadaptCut5Second"),
    ("cutadapt_cut_3_first",  "cutadaptCut3First"),
    ("cutadapt_cut_3_second", "cutadaptCut3Second"),
]


def _fill_cutadapt_defaults(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
) -> pl.DataFrame:
    """Fill absent cutadapt columns from global config; compute rev-comp fallbacks.

    When a cutadapt adapter column is absent from the experiment design file,
    fills every row with the global config value (None when config is also unset).
    Per-sample columns that are already present are left untouched.

    Then, for non-trans libraries, auto-fills:
      cutadapt3First  = revcomp(cutadapt5Second)  where cutadapt3First  is None
      cutadapt3Second = revcomp(cutadapt5First)   where cutadapt3Second is None

    Mirrors: R/dimsum__get_experiment_design.R lines 48–67 and 154–161.
    """
    for config_field, col_name in _ADAPTER_COLS:
        if col_name not in exp_design_df.columns:
            exp_design_df = exp_design_df.with_columns(
                pl.lit(getattr(config, config_field)).cast(pl.String).alias(col_name)
            )

    for config_field, col_name in _CUT_COLS:
        if col_name not in exp_design_df.columns:
            exp_design_df = exp_design_df.with_columns(
                pl.lit(getattr(config, config_field)).cast(pl.Int64).alias(col_name)
            )

    # Rev-comp fallbacks (non-trans library only): use Polars expressions so the
    # column dtype stays String even when the source column was filled with None.
    if not config.trans_library:
        exp_design_df = exp_design_df.with_columns(
            pl.when(
                pl.col("cutadapt3First").is_null() & pl.col("cutadapt5Second").is_not_null()
            )
            .then(pl.col("cutadapt5Second").map_elements(_revcomp, return_dtype=pl.String))
            .otherwise(pl.col("cutadapt3First"))
            .alias("cutadapt3First"),
            pl.when(
                pl.col("cutadapt3Second").is_null() & pl.col("cutadapt5First").is_not_null()
            )
            .then(pl.col("cutadapt5First").map_elements(_revcomp, return_dtype=pl.String))
            .otherwise(pl.col("cutadapt3Second"))
            .alias("cutadapt3Second"),
        )

    return exp_design_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_trim(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
    outpath: Path,
) -> pl.DataFrame:
    """Run cutadapt adapter trimming (stage 2).

    For each unique FASTQ pair in exp_design_df, runs cutadapt to remove 5'/3'
    constant adapter sequences and optional fixed-length 5'/3' cuts.

    Parameters
    ----------
    config:
        Pipeline configuration.
    exp_design_df:
        Experiment design DataFrame.
    outpath:
        Directory for trimmed output files.

    Returns
    -------
    Updated exp_design_df with pair1/pair2 having .cutadapt.gz suffix,
    and pair_directory pointing to outpath.
    """
    outpath.mkdir(parents=True, exist_ok=True)
    logger.info("=== Stage 2 (WRAP): TRIM CONSTANT REGIONS ===")

    # Fill absent cutadapt columns from global config and compute rev-comp fallbacks
    exp_design_df = _fill_cutadapt_defaults(config, exp_design_df)

    # If not trans library: convert linked adapters when both 5' and 3' adapters specified
    if not config.trans_library:
        exp_design_df = _convert_linked_adapters(config, exp_design_df)

    # Unique pairs (multiple samples may share the same FASTQ files)
    pair_cols = ["pair1", "pair2"]
    present_pair_cols = [c for c in pair_cols if c in exp_design_df.columns]
    unique_pairs_df = exp_design_df.unique(subset=present_pair_cols, keep="first")

    for row in unique_pairs_df.iter_rows(named=True):
        pair1 = row["pair1"]
        pair2 = row.get("pair2")
        pdir = Path(row["pair_directory"])
        abs1 = str(pdir / pair1)
        abs2 = str(pdir / pair2) if (config.paired and pair2) else None

        out1 = str(outpath / f"{pair1}.cutadapt.gz")
        out2 = str(outpath / f"{pair2}.cutadapt.gz") if (config.paired and pair2) else None

        stdout_path = outpath / f"{pair1}.cutadapt.gz.stdout"
        stderr_path = outpath / f"{pair1}.cutadapt.gz.stderr"

        # No adapter options → copy file unchanged
        if not _row_needs_cutadapt(row):
            logger.info("  No cutadapt options for %s — copying.", pair1)
            shutil.copy2(abs1, out1)
            if config.paired and abs2:
                shutil.copy2(abs2, out2)
            # Write a minimal stdout (line count)
            with open(stdout_path, "w") as fh:
                fh.write("0\n")
            continue

        # Not stranded + paired: swap reads first then trim
        if not config.stranded and config.paired:
            abs1_swap, abs2_swap = _swap_reads(
                row, abs1, abs2, outpath, config
            )
            _run_cutadapt_single(
                row=row,
                abs1=abs1_swap,
                abs2=abs2_swap,
                out1=out1,
                out2=out2,
                outpath=outpath,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                config=config,
                is_swapped=True,
            )
        else:
            _run_cutadapt_single(
                row=row,
                abs1=abs1,
                abs2=abs2,
                out1=out1,
                out2=out2,
                outpath=outpath,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                config=config,
                is_swapped=False,
            )

    # Update exp_design_df
    suffix = ".cutadapt.gz"
    exp_design_df = exp_design_df.with_columns([
        (pl.col("pair1") + suffix).alias("pair1"),
        pl.lit(str(outpath)).alias("pair_directory"),
    ])
    if config.paired and "pair2" in exp_design_df.columns:
        exp_design_df = exp_design_df.with_columns(
            (pl.col("pair2") + suffix).alias("pair2")
        )

    return exp_design_df


# ---------------------------------------------------------------------------
# Cutadapt helpers
# ---------------------------------------------------------------------------


def _row_needs_cutadapt(row: dict) -> bool:
    """Return True if any adapter/cut option is specified for this row."""
    for k in [
        "cutadapt5First", "cutadapt5Second", "cutadapt3First", "cutadapt3Second",
        "cutadaptCut5First", "cutadaptCut5Second", "cutadaptCut3First", "cutadaptCut3Second",
    ]:
        if row.get(k) is not None:
            return True
    return False


def _get_adapter_options(row: dict, paired: bool) -> list[str]:
    """Build cutadapt adapter flags for read 1 and (if paired) read 2.

    Mirrors: dimsum__get_cutadapt_options(option_type="default")
    """
    opts: list[str] = []
    cut5f = row.get("cutadapt5First")
    cut5s = row.get("cutadapt5Second")
    cut3f = row.get("cutadapt3First")
    cut3s = row.get("cutadapt3Second")

    if cut5f is not None:
        opts += ["-g", cut5f]
    if cut5s is not None and paired:
        opts += ["-G", cut5s]
    if cut3f is not None:
        opts += ["-a", cut3f]
    if cut3s is not None and paired:
        opts += ["-A", cut3s]

    # Discard untrimmed reads unless running cut-only mode
    cut_only = row.get("run_cutadapt_cutonly", False)
    if not cut_only:
        opts.append("--discard-untrimmed")

    return opts


def _get_swap_options(row: dict) -> list[str]:
    """Build cutadapt flags for the read-swapping pre-pass.

    Mirrors: dimsum__get_cutadapt_options(option_type="swap")
    """
    opts: list[str] = []
    cut5f = row.get("cutadapt5First")
    cut3f = row.get("cutadapt3First")
    cut5s = row.get("cutadapt5Second")
    cut3s = row.get("cutadapt3Second")

    if cut5f is not None:
        opts += ["-g", f"forward={cut5f}"]
    elif cut3f is not None:
        opts += ["-a", f"forward={cut3f}"]

    if cut5s is not None:
        opts += ["-g", f"reverse={cut5s}"]
    elif cut3s is not None:
        opts += ["-a", f"reverse={cut3s}"]

    return opts


def _get_cut_options(row: dict, paired: bool) -> list[str]:
    """Build cutadapt fixed-length base-removal flags.

    Mirrors: dimsum__get_cutadapt_options(option_type="cut")
    """
    opts: list[str] = []
    c5f = row.get("cutadaptCut5First")
    c3f = row.get("cutadaptCut3First")
    c5s = row.get("cutadaptCut5Second")
    c3s = row.get("cutadaptCut3Second")

    if c5f is not None:
        opts += ["-u", str(c5f)]
    if c3f is not None:
        opts += ["-u", str(-c3f)]
    if paired:
        if c5s is not None:
            opts += ["-U", str(c5s)]
        if c3s is not None:
            opts += ["-U", str(-c3s)]

    return opts


# Flags that pyDiMSum manages itself and must not be overridden via the
# passthrough.  Long-form variants are stored without leading '--' for easy
# set-membership lookup after stripping.
_MANAGED_FLAGS: frozenset[str] = frozenset({
    # Adapter flags
    "-g", "-G", "-a", "-A",
    # Fixed-base removal
    "-u", "-U",
    # Length filters
    "-m", "--minimum-length",
    "-M", "--maximum-length",
    # Adapter-matching tuning
    "-e", "--error-rate",
    "-O", "--overlap",
    # Parallelism / I/O (always set by pyDiMSum)
    "-j", "--cores",
    "-o", "--output",
    "-p", "--paired-output",
})


def _get_passthrough_options(row: dict, config: RunConfig) -> list[str]:
    """Return extra cutadapt flags from the per-sample row or global config.

    The raw string is shlex-split and appended after pyDiMSum's own flags in
    the trim command.  Any flag that pyDiMSum already manages is rejected with
    a helpful error so the user knows which pydimsum option to use instead.
    Flags not in the managed set are forwarded as-is; cutadapt itself will
    error on genuinely unknown flags.

    ``--discard-untrimmed`` is intentionally NOT in the managed set: it is
    already appended by default in the non-cut-only trim path, is idempotent,
    and is the primary motivation for this feature (PacBio reads).
    """
    raw = row.get("cutadaptOptions") or config.cutadapt_options
    if not raw:
        return []

    tokens = shlex.split(raw)

    for token in tokens:
        if not token.startswith("-"):
            continue
        # Strip any attached value (--flag=value or -eVALUE)
        flag = token.split("=", 1)[0]
        # For short opts that may have an attached value (-e0.1) keep only -X
        if len(flag) > 2 and not flag.startswith("--"):
            flag = flag[:2]
        if flag in _MANAGED_FLAGS:
            raise ValueError(
                f"cutadapt_options contains '{flag}', which is managed by "
                f"pyDiMSum. Use the corresponding pyDiMSum flag instead "
                f"(e.g. --cutadapt_error_rate, --cutadapt_min_length)."
            )

    return tokens


def _run_cutadapt_single(
    row: dict,
    abs1: str,
    abs2: str | None,
    out1: str,
    out2: str | None,
    outpath: Path,
    stdout_path: Path,
    stderr_path: Path,
    config: RunConfig,
    is_swapped: bool,
) -> None:
    """Build and execute a single cutadapt command."""
    paired = config.paired

    adapter_opts = _get_adapter_options(row, paired)
    cut_opts = _get_cut_options(row, paired)

    min_len = str(row.get("cutadaptMinLength") or config.cutadapt_min_length)
    max_len = row.get("cutadaptMaxLength") or config.cutadapt_max_length
    error_rate = str(row.get("cutadaptErrorRate") or config.cutadapt_error_rate)
    overlap = str(row.get("cutadaptOverlap") or config.cutadapt_overlap)

    cmd: list[str] = (
        ["cutadapt"]
        + adapter_opts
        + cut_opts
        + ["--minimum-length", min_len, "-e", error_rate, "-O", overlap,
           "-j", str(config.num_cores)]
    )
    if max_len is not None:
        cmd += ["--maximum-length", str(max_len)]

    cmd += _get_passthrough_options(row, config)

    if paired and out2:
        cmd += ["-o", out1, "-p", out2, abs1, abs2 or ""]
    else:
        cmd += ["-o", out1, abs1]

    result = subprocess.run(
        cmd,
        stdout=open(stdout_path, "w"),
        stderr=open(stderr_path, "w"),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cutadapt trim failed (exit {result.returncode}). See {stderr_path}"
        )


def _swap_reads(
    row: dict,
    abs1: str,
    abs2: str | None,
    outpath: Path,
    config: RunConfig,
) -> tuple[str, str]:
    """Detect orientation and swap reads so R1 always carries the 5' adapter.

    Mirrors: R/dimsum__swap_reads.R

    Runs cutadapt with named adapter groups to detect orientation, then
    re-orients so that pair1 → forward strand.
    """
    name1 = Path(abs1).name
    swap_out1 = str(outpath / f"{name1}.cutadapt2.gz")
    swap_out2 = str(outpath / f"{Path(abs2).name}.cutadapt2.gz") if abs2 else ""

    swap_opts = _get_swap_options(row)
    if not swap_opts:
        # No swap adapters → return unchanged
        return abs1, abs2 or ""

    swap_cmd = (
        ["cutadapt"]
        + swap_opts
        + [
            "-o", swap_out1,
            "-p", swap_out2,
            abs1, abs2 or "",
        ]
    )
    result = subprocess.run(swap_cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(
            "cutadapt swap returned %d: %s",
            result.returncode,
            result.stderr.decode(errors="replace"),
        )
    return swap_out1, swap_out2


# ---------------------------------------------------------------------------
# Linked adapter conversion
# ---------------------------------------------------------------------------


def _convert_linked_adapters(
    config: RunConfig,
    exp_design_df: pl.DataFrame,
) -> pl.DataFrame:
    """Convert to linked adapters when both 5' and 3' adapters are specified
    and the read is long enough to see the 3' adapter.

    Mirrors: R/dimsum__convert_linked_adapters.R
    """
    wt_len = len(config.wt_nt_seq)
    len_shortest = wt_len
    if config.retain_indels and config._indel_lengths:
        len_shortest = min(len_shortest, min(config._indel_lengths))
    elif config.retain_indels:
        len_shortest = 1

    # Operate row-by-row to update adapter strings in place
    rows = exp_design_df.to_dicts()
    for row in rows:
        cut5f = row.get("cutadaptCut5First") or 0
        cut5s = row.get("cutadaptCut5Second") or 0
        cut3f = row.get("cutadaptCut3First") or 0
        cut3s = row.get("cutadaptCut3Second") or 0

        # Read 1: convert to linked adapter
        ca5f = row.get("cutadapt5First")
        ca3f = row.get("cutadapt3First")
        p1_len = row.get("pair1_length")
        if (
            ca5f is not None
            and ca3f is not None
            and "..." not in ca3f
            and p1_len is not None
        ):
            effective_len = int(p1_len) - cut5f - cut3f
            if effective_len > len(ca5f) + len_shortest:
                row["cutadapt3First"] = f"{ca5f};required...{ca3f};optional"
                row["cutadapt5First"] = None

        # Read 2: convert to linked adapter
        ca5s = row.get("cutadapt5Second")
        ca3s = row.get("cutadapt3Second")
        p2_len = row.get("pair2_length")
        if (
            ca5s is not None
            and ca3s is not None
            and "..." not in ca3s
            and p2_len is not None
        ):
            effective_len = int(p2_len) - cut5s - cut3s
            if effective_len > len(ca5s) + len_shortest:
                row["cutadapt3Second"] = f"{ca5s};required...{ca3s};optional"
                row["cutadapt5Second"] = None

    return pl.from_dicts(rows, schema_overrides=exp_design_df.schema)
