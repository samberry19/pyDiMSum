"""Read, validate, and reformat variant count files.

Mirrors:
  R/dimsum__check_countfile.R
  R/dimsum_stage_merge.R (count-file branch, lines 69-73)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pydimsum.io.designs import ExperimentDesign


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_count_file(
    count_path: Path,
    exp_design: ExperimentDesign,
) -> pl.DataFrame:
    """Load and validate a user-supplied variant count file.

    Returns a Polars DataFrame with columns:
      - ``nt_seq`` (Utf8, lower-cased)
      - One column per sample named with the internal DiMSum convention:
        ``{sample_name}_e{experiment}_s{selection_id}_b{bio_rep}_tNA_count``
        cast to UInt32.

    Parameters
    ----------
    count_path:
        Path to the tab-separated count file.  First column must be ``nt_seq``;
        remaining columns must match sample_name values in exp_design.
    exp_design:
        Validated ExperimentDesign object.

    Raises
    ------
    ValueError
        If mandatory columns are missing, sequences are invalid, count values
        are not non-negative integers, or duplicate nt_seq values are present.
    """
    # Normalize line endings (handle \r-only, \r\n, \n uniformly)
    import io
    raw = Path(count_path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    df = pl.read_csv(
        io.BytesIO(raw),
        separator="\t",
        null_values=["", "NA"],
    )

    # ---- nt_seq column checks ----
    if "nt_seq" not in df.columns:
        raise ValueError(
            "Mandatory column 'nt_seq' missing from variant count file."
        )
    if df["nt_seq"].dtype != pl.Utf8:
        raise ValueError(
            "'nt_seq' column must be a character string in the variant count file."
        )
    # Lower-case sequences
    df = df.with_columns(pl.col("nt_seq").str.to_lowercase())
    # Validate: only acgt
    bad_seqs = df.filter(
        pl.col("nt_seq").str.contains(r"[^acgt]")
    )
    if len(bad_seqs) > 0:
        raise ValueError(
            f"Invalid 'nt_seq' values in variant count file "
            f"(non-ACGT characters found in {len(bad_seqs)} rows)."
        )

    # ---- Sample column checks ----
    required_samples = exp_design.sample_names
    missing = [s for s in required_samples if s not in df.columns]
    if missing:
        raise ValueError(
            f"Sample names missing from count file columns: {missing}"
        )

    # Validate count columns: non-negative integers
    count_cols = [c for c in df.columns if c != "nt_seq"]
    for col in count_cols:
        if col not in required_samples:
            continue  # extra columns are ignored
        col_data = df[col]
        if col_data.dtype not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            # Try casting; fail if values contain non-integer data
            try:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            except Exception:
                raise ValueError(
                    f"Count column '{col}' contains non-integer values."
                )
        min_val = df[col].min()
        if min_val is not None and min_val < 0:
            raise ValueError(
                f"Count column '{col}' contains negative values."
            )

    # ---- Duplicate nt_seq check ----
    n_dupes = df["nt_seq"].is_duplicated().sum()
    if n_dupes > 0:
        raise ValueError(
            f"Duplicated 'nt_seq' values found in variant count file "
            f"({n_dupes} duplicated rows)."
        )

    # ---- Rename columns to internal DiMSum convention ----
    rename_map = {
        sn: exp_design.internal_col_name(sn)
        for sn in required_samples
    }
    # Select only nt_seq + sample columns in defined order, then rename
    df = df.select(["nt_seq"] + required_samples)
    df = df.rename(rename_map)

    # Cast count columns to UInt32 (counts are non-negative integers)
    internal_count_cols = list(rename_map.values())
    df = df.with_columns(
        [pl.col(c).fill_null(0).cast(pl.UInt32) for c in internal_count_cols]
    )

    return df
