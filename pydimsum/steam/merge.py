"""Stage 4 (STEAM) — build the variant data table from count files.

Two entry paths:

Count-file (STEAM-only):
  1. Reads the pre-validated wide count DataFrame from io/counts.py.
  2. Aggregates biological output replicates per experiment replicate.
  3. Renames columns to the canonical ``count_e{E}_s{0|1}`` convention.

WRAP path (starcode .vsearch.unique files):
  1. Reads each per-sample starcode output file (nt_seq, count).
  2. Concatenates into a long Polars DataFrame.
  3. Single-pass pivot → wide count table (one column per sample).
  4. Sums technical replicates (same e/s/b, different t).
  5. Aggregates biological output replicates per experiment.
  6. Renames to ``count_e{E}_s{0|1}``.

Mirrors:
  R/dimsum_stage_merge.R (lines 63-127)
  R/dimsum_stage_counts_to_fitness.R:107-121 (bio-rep aggregation)
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from pydimsum.config import RunConfig
from pydimsum.io.counts import load_count_file
from pydimsum.io.designs import ExperimentDesign

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_variant_table(
    config: RunConfig,
    exp_design: ExperimentDesign,
) -> pl.DataFrame:
    """Build and return the wide variant count table.

    Returns a Polars DataFrame with columns:
      ``nt_seq`` (Utf8) and
      ``count_e{E}_s0`` / ``count_e{E}_s1`` (UInt32) for each replicate E.

    Parameters
    ----------
    config:
        Validated pipeline configuration.
    exp_design:
        Validated experiment design.
    """
    if config.count_path is not None:
        logger.info("Loading variant count file: %s", config.count_path)
        df = load_count_file(config.count_path, exp_design)
        logger.info("Loaded %d variants", len(df))
        df = _aggregate_and_rename(df, exp_design)
        return df

    # WRAP path: read starcode output files from exp_design_df
    return _build_from_wrap_files(exp_design)


# ---------------------------------------------------------------------------
# WRAP path: build from starcode files
# ---------------------------------------------------------------------------


def _build_from_wrap_files(exp_design: ExperimentDesign) -> pl.DataFrame:
    """Read per-sample starcode count files and build the wide variant table.

    Implements the single-pass pivot replacing R's iterative pairwise merge.
    Mirrors: R/dimsum_stage_merge.R:63-127 (non-countPath branch).

    Steps:
      1. Read each .vsearch.unique file → (nt_seq, count, sample_id)
      2. Concat all into a long DataFrame
      3. Single pivot to wide (one column per sample_id)
      4. Fill null → 0
      5. Sum technical replicates (same e/s/b, different t)
      6. Aggregate biological output replicates per E → count_e{E}_s1
      7. Rename input columns → count_e{E}_s0
    """
    df_esd = exp_design.df
    required_cols = {"aligned_pair_unique", "aligned_pair_unique_directory"}
    if not required_cols.issubset(set(df_esd.columns)):
        raise RuntimeError(
            "Experiment design does not contain WRAP output columns "
            "(aligned_pair_unique / aligned_pair_unique_directory). "
            "Run WRAP stages 0-3 first."
        )

    # Build sample_id → (experiment, selection_id, biological_replicate, technical_replicate)
    rows = df_esd.to_dicts()
    long_frames: list[pl.DataFrame] = []

    for row in rows:
        adir = Path(row["aligned_pair_unique_directory"])
        fname = row["aligned_pair_unique"]
        fpath = adir / fname

        # sample_id = file stem (without .split suffix)
        sample_id = fname.split(".split")[0]

        if not fpath.exists():
            logger.warning("Starcode file not found: %s — filling zeros", fpath)
            continue

        # Read the two-column starcode output: sequence, count
        raw = pl.read_csv(
            str(fpath),
            separator="\t",
            has_header=False,
            new_columns=["nt_seq", "count"],
        )
        raw = raw.with_columns([
            pl.col("nt_seq").str.to_lowercase(),
            pl.col("count").cast(pl.UInt32),
            pl.lit(sample_id).alias("sample_id"),
        ])
        long_frames.append(raw)

    if not long_frames:
        raise RuntimeError("No starcode count files could be loaded.")

    long_df = pl.concat(long_frames, how="diagonal_relaxed")
    logger.info(
        "Loaded %d total (nt_seq, sample_id) count records from %d files",
        len(long_df), len(long_frames),
    )

    # Single-pass pivot: rows = nt_seq, columns = sample_id
    wide_df = long_df.pivot(
        values="count",
        index="nt_seq",
        on="sample_id",
        aggregate_function="sum",
    )
    wide_df = wide_df.with_columns(
        [pl.col(c).fill_null(0).cast(pl.UInt32)
         for c in wide_df.columns if c != "nt_seq"]
    )
    logger.info("Wide count table: %d variants × %d samples", len(wide_df), len(wide_df.columns) - 1)

    # Sum technical replicates: group columns sharing the same e/s/b prefix
    # (split on '_t', keep prefix before first '_t')
    sample_cols = [c for c in wide_df.columns if c != "nt_seq"]
    prefix_map: dict[str, list[str]] = {}
    for col in sample_cols:
        parts = col.split("_t")
        prefix = parts[0]  # everything before _t{N}
        prefix_map.setdefault(prefix, []).append(col)

    summed_exprs = []
    summed_names = []
    for prefix, cols in prefix_map.items():
        if len(cols) == 1:
            summed_exprs.append(pl.col(cols[0]).alias(f"{prefix}_count"))
        else:
            summed_exprs.append(
                pl.sum_horizontal([pl.col(c) for c in cols]).cast(pl.UInt32).alias(f"{prefix}_count")
            )
        summed_names.append(f"{prefix}_count")

    wide_df = wide_df.select(["nt_seq"] + summed_exprs)

    # Now aggregate biological replicates and rename to count_e{E}_s{0|1}
    # For this we need to match column names to exp_design rows
    return _aggregate_wrap_cols(wide_df, exp_design)


def _aggregate_wrap_cols(
    df: pl.DataFrame,
    exp_design: ExperimentDesign,
) -> pl.DataFrame:
    """Aggregate biological output replicates from WRAP-path count columns.

    WRAP column names after tech-rep summing:
      ``{sample_name}_e{E}_s{S}_b{B}_count``

    Maps each column to its (E, S, B) from the experiment design and then:
    - Input (S=0): rename → ``count_e{E}_s0``
    - Output (S=1): sum all B replicates → ``count_e{E}_s1``

    Mirrors: R/dimsum_stage_merge.R:109-111 + _aggregate_and_rename logic
    """
    esd_rows = exp_design.df.to_dicts()

    # Build a lookup: sample_name → (experiment, selection_id, biological_replicate)
    sample_meta: dict[str, dict] = {
        row["sample_name"]: row for row in esd_rows
    }

    reps = exp_design.replicates
    exprs: list[pl.Expr] = []

    for E in reps:
        # Input (s0)
        input_rows = [r for r in esd_rows if r["experiment"] == E and r["selection_id"] == 0]
        if input_rows:
            sname = input_rows[0]["sample_name"]
            B = input_rows[0].get("biological_replicate") or input_rows[0].get("selection_replicate", "NA")
            col = f"{sname}_e{E}_s0_b{B}_count"
            if col in df.columns:
                exprs.append(pl.col(col).alias(f"count_e{E}_s0"))
            else:
                logger.warning("Input count column not found: %s", col)
                exprs.append(pl.lit(0).cast(pl.UInt32).alias(f"count_e{E}_s0"))

        # Output (s1): sum all biological replicates
        output_rows = [r for r in esd_rows if r["experiment"] == E and r["selection_id"] == 1]
        out_cols = []
        for r in output_rows:
            sname = r["sample_name"]
            B = r.get("biological_replicate") or r.get("selection_replicate", "NA")
            col = f"{sname}_e{E}_s1_b{B}_count"
            if col in df.columns:
                out_cols.append(col)
            else:
                logger.warning("Output count column not found: %s", col)

        if len(out_cols) == 1:
            exprs.append(pl.col(out_cols[0]).alias(f"count_e{E}_s1"))
        elif len(out_cols) > 1:
            exprs.append(
                pl.sum_horizontal([pl.col(c) for c in out_cols])
                .cast(pl.UInt32)
                .alias(f"count_e{E}_s1")
            )
        else:
            exprs.append(pl.lit(0).cast(pl.UInt32).alias(f"count_e{E}_s1"))

    result = df.select(["nt_seq"] + exprs)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_and_rename(
    df: pl.DataFrame,
    exp_design: ExperimentDesign,
) -> pl.DataFrame:
    """Aggregate biological output replicates and rename to count_e{E}_s{0|1}.

    For input samples (selection_id=0): there is one per experiment replicate;
    rename directly.

    For output samples (selection_id=1): sum across biological replicates
    within an experiment replicate.  Only sum rows where not *all* biological
    reps are 0 (mirrors R: rowSums(...) != 0 guard — though R fills NA→0
    before this step, so the guard just means at least one observation).

    Mirrors:
      R/dimsum_stage_counts_to_fitness.R:107-121
    """
    reps = exp_design.replicates
    expressions = []

    for E in reps:
        # ---- Input ----
        input_sample = exp_design.input_sample_for_replicate(E)
        input_col = exp_design.internal_col_name(input_sample)
        expressions.append(
            pl.col(input_col).alias(f"count_e{E}_s0")
        )

        # ---- Output ----
        output_samples = exp_design.output_samples_for_replicate(E)
        output_cols = [exp_design.internal_col_name(s) for s in output_samples]

        if len(output_cols) == 1:
            expressions.append(
                pl.col(output_cols[0]).alias(f"count_e{E}_s1")
            )
        else:
            # Sum across biological replicates (filling null → 0 first,
            # consistent with R's NA→0 replacement before aggregation)
            sum_expr = pl.sum_horizontal(
                [pl.col(c).fill_null(0) for c in output_cols]
            ).alias(f"count_e{E}_s1")
            expressions.append(sum_expr)

    result = df.select(["nt_seq"] + expressions)
    # Ensure all count columns are UInt32
    count_cols = [c for c in result.columns if c.startswith("count_e")]
    result = result.with_columns(
        [pl.col(c).fill_null(0).cast(pl.UInt32) for c in count_cols]
    )
    return result
