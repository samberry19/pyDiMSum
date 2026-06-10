"""Read and validate the experimental design and related metadata files.

Mirrors:
  R/dimsum__get_experiment_design.R
  R/dimsum__check_experiment_design_countfile.R
  R/dimsum__get_barcode_design.R
  R/dimsum__get_synonymsequences.R
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

class ExperimentDesign:
    """Validated experiment design with derived replicate structure.

    Attributes
    ----------
    df : polars.DataFrame
        The experiment design table with columns:
        sample_name, experiment_replicate (=experiment), selection_id,
        selection_replicate (=biological_replicate), technical_replicate,
        pair1, pair2, generations, cell_density, selection_time, ...
    replicates : list[int]
        Sorted unique experiment_replicate values.
    """

    def __init__(
        self,
        path: Path,
        count_path: Path | None = None,
        fastq_file_dir: Path | None = None,
        allow_pair_duplicates: bool = False,
    ) -> None:
        self.path = path
        self._allow_pair_duplicates = allow_pair_duplicates
        # Normalize line endings (handle \r-only, \r\n, \n uniformly)
        raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        import io
        df = pl.read_csv(io.BytesIO(raw), separator="\t", null_values=["", "NA"])

        # Remove rows with missing sample_name
        if "sample_name" in df.columns:
            df = df.filter(pl.col("sample_name").is_not_null())

        # Backwards compatibility: copy old column names
        if "experiment" in df.columns and "experiment_replicate" not in df.columns:
            df = df.rename({"experiment": "experiment_replicate"})
        if "biological_replicate" in df.columns and "selection_replicate" not in df.columns:
            df = df.rename({"biological_replicate": "selection_replicate"})

        # DiMSum uses "experiment" internally after the temp fix; keep both
        if "experiment_replicate" in df.columns and "experiment" not in df.columns:
            df = df.with_columns(pl.col("experiment_replicate").alias("experiment"))
        if "selection_replicate" in df.columns and "biological_replicate" not in df.columns:
            df = df.with_columns(pl.col("selection_replicate").alias("biological_replicate"))

        # Add optional columns if missing
        for col in ["generations", "cell_density", "selection_time"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

        # Apply fastq_file_dir override: sets pair_directory for all rows
        if fastq_file_dir is not None:
            df = df.with_columns(
                pl.lit(str(fastq_file_dir)).alias("pair_directory")
            )

        # Clear FASTQ columns when using count file
        if count_path is not None:
            for col in ["technical_replicate", "pair1", "pair2"]:
                if col in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))

        self.df = df
        self._validate()

    def _validate(self) -> None:
        required = ["sample_name", "experiment_replicate", "selection_id"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Mandatory columns missing from experimentDesign file: {missing}"
            )
        # Check selection_id values
        valid_ids = {0, 1}
        bad = set(self.df["selection_id"].to_list()) - valid_ids
        if bad:
            raise ValueError(
                f"selection_id must be 0 (input) or 1 (output), found: {bad}"
            )
        # Check for duplicate FASTQ pairs (unless explicitly allowed)
        if not self._allow_pair_duplicates:
            pair_cols = [c for c in ("pair1", "pair2") if c in self.df.columns]
            if pair_cols:
                non_null = self.df.filter(pl.col(pair_cols[0]).is_not_null())
                if len(non_null) > 0:
                    key = pl.concat_str(pair_cols, separator="\t")
                    n_unique = non_null.select(key.alias("_key"))["_key"].n_unique()
                    if n_unique < len(non_null):
                        raise ValueError(
                            "Duplicate FASTQ pair entries found in experiment design. "
                            "Use --experiment_design_pair_duplicates to allow duplicates."
                        )

    @property
    def replicates(self) -> list[int]:
        return sorted(self.df["experiment"].unique().to_list())

    @property
    def sample_names(self) -> list[str]:
        return self.df["sample_name"].to_list()

    def input_sample_for_replicate(self, rep: int) -> str:
        """Return the sample_name for the input of a given replicate."""
        rows = self.df.filter(
            (pl.col("experiment") == rep) & (pl.col("selection_id") == 0)
        )
        names = rows["sample_name"].to_list()
        if not names:
            raise ValueError(f"No input sample found for replicate {rep}")
        return names[0]

    def output_samples_for_replicate(self, rep: int) -> list[str]:
        """Return sample_names for all output biological reps of a replicate."""
        rows = self.df.filter(
            (pl.col("experiment") == rep) & (pl.col("selection_id") == 1)
        )
        return rows["sample_name"].to_list()

    def internal_col_name(self, sample_name: str) -> str:
        """Compute the internal column name used in the wide count table.

        Mirrors dimsum__check_countfile.R sample_names construction:
          {sample_name}_e{experiment}_s{selection_id}_b{biological_replicate}_tNA_count
        """
        row = self.df.filter(pl.col("sample_name") == sample_name)
        if row.is_empty():
            raise KeyError(f"sample_name not found: {sample_name}")
        row = row.row(0, named=True)
        brep = row.get("biological_replicate")
        brep_str = str(int(brep)) if brep is not None else "NA"
        return (
            f"{sample_name}_e{int(row['experiment'])}"
            f"_s{int(row['selection_id'])}"
            f"_b{brep_str}_tNA_count"
        )


# ---------------------------------------------------------------------------
# Barcode identity
# ---------------------------------------------------------------------------

def load_barcode_identity(path: Path) -> dict[str, str]:
    """Load a barcode-to-variant identity file.

    Expected format: tab-separated with columns ``barcode`` and ``variant``
    (both A/C/G/T nucleotide sequences, case-insensitive).

    Returns a dict mapping ``barcode.lower() → variant.lower()``.

    Mirrors dimsum__check_barcodeidentityfile.R.
    """
    import io as _io
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    df = pl.read_csv(_io.BytesIO(raw), separator="\t", null_values=["", "NA"])

    mandatory = ["barcode", "variant"]
    missing = [c for c in mandatory if c not in df.columns]
    if missing:
        raise ValueError(
            f"Mandatory columns missing from barcodeIdentity file: {missing}. "
            "Required columns: 'barcode', 'variant'."
        )

    bad_rows = df.filter(
        pl.col("barcode").str.contains(r"[^ACGTacgt]") |
        pl.col("variant").str.contains(r"[^ACGTacgt]")
    )
    if len(bad_rows) > 0:
        raise ValueError(
            f"barcodeIdentity file contains non-ACGT characters in {len(bad_rows)} rows."
        )

    return {
        str(row["barcode"]).lower(): str(row["variant"]).lower()
        for row in df.iter_rows(named=True)
    }


# ---------------------------------------------------------------------------
# Synonym sequences
# ---------------------------------------------------------------------------

def load_synonym_sequences(path: Path) -> list[str]:
    """Load a plain-text synonym sequences file (one sequence per line).

    Returns a list of lower-cased nucleotide sequences.
    Mirrors dimsum__get_synonymsequences.R.
    """
    sequences = []
    with open(path) as fh:
        for line in fh:
            seq = line.strip().lower()
            if seq:
                sequences.append(seq)
    return sequences
