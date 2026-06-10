"""Stage 5 (STEAM) — merge replicate fitness and write output files.

Mirrors:
  R/dimsum__merge_fitness.R

Key improvement: the per-group singles[Pos==.. & Mut==..] lookups in R
(lines 183-186) are replaced with Polars joins.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Standard amino acid single-letter → three-letter mapping
_AA_THREE = {
    "G": "Gly", "A": "Ala", "V": "Val", "L": "Leu", "M": "Met",
    "I": "Ile", "F": "Phe", "Y": "Tyr", "W": "Trp", "K": "Lys",
    "R": "Arg", "H": "His", "D": "Asp", "E": "Glu", "S": "Ser",
    "T": "Thr", "C": "Cys", "N": "Asn", "Q": "Gln", "P": "Pro",
    "*": "Ter",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_fitness(
    all_data_df: pl.DataFrame,
    singles_df: pl.DataFrame,
    doubles_df: pl.DataFrame,
    replicates: list[int],
    sequence_type: str,
    output_path: Path,
    project_name: str,
    bayesian_double_fitness: bool = False,
) -> None:
    """Compute inverse-variance merged fitness, add single-mut info to doubles,
    and write all output files.

    Mirrors: R/dimsum__merge_fitness.R

    Parameters
    ----------
    all_data_df:
        Full variant table with per-replicate fitness{E}_uncorr and sigma{E}_uncorr.
    singles_df, doubles_df:
        Output of mutations.identify_singles / identify_doubles, with fitness/sigma
        (no _uncorr suffix — already dropped by mutations.py).
    replicates:
        List of experiment replicate integers.
    sequence_type:
        "coding" or "noncoding".
    output_path:
        Directory for output files (project_path).
    project_name:
        Used in .RData-equivalent Parquet file name.
    """
    reps = replicates
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Merge all_data ----
    all_data_df = _inverse_variance_merge(
        all_data_df, reps, fitness_suffix="_uncorr", sigma_suffix="_uncorr",
        out_fitness="fitness", out_sigma="sigma",
    )

    # ---- Merge singles ----
    if singles_df is not None and len(singles_df) > 0:
        singles_df = _inverse_variance_merge(
            singles_df, reps, fitness_suffix="", sigma_suffix="",
            out_fitness="fitness", out_sigma="sigma",
        )

    # ---- Merge doubles (uncorrected) ----
    if doubles_df is not None and len(doubles_df) > 0:
        doubles_df = _inverse_variance_merge(
            doubles_df, reps, fitness_suffix="_uncorr", sigma_suffix="_uncorr",
            out_fitness="fitness_uncorr", out_sigma="sigma_uncorr",
        )

    # ---- Rename merge_seq → nt_seq (drop existing nt_seq first to avoid duplicate) ----
    if "merge_seq" in all_data_df.columns:
        if "nt_seq" in all_data_df.columns:
            all_data_df = all_data_df.drop("nt_seq")
        all_data_df = all_data_df.rename({"merge_seq": "nt_seq"})
    if singles_df is not None and len(singles_df) > 0 and "merge_seq" in singles_df.columns:
        if "nt_seq" in singles_df.columns:
            singles_df = singles_df.drop("nt_seq")
        singles_df = singles_df.rename({"merge_seq": "nt_seq"})
    if doubles_df is not None and len(doubles_df) > 0 and "merge_seq" in doubles_df.columns:
        if "nt_seq" in doubles_df.columns:
            doubles_df = doubles_df.drop("nt_seq")
        doubles_df = doubles_df.rename({"merge_seq": "nt_seq"})

    # ---- For coding sequences: null nt_seq/Nham_nt/Nmut_codons for nonsynonymous variants ----
    if sequence_type == "coding":
        # all_data: null nt_seq for nonsynonymous error_model rows
        if "nt_seq" in all_data_df.columns and "Nham_aa" in all_data_df.columns:
            all_data_df = all_data_df.with_columns(
                pl.when(
                    pl.col("error_model").fill_null(False) &
                    (pl.col("Nham_aa").fill_null(0) > 0 | pl.col("indel").fill_null(False))
                ).then(None).otherwise(pl.col("nt_seq"))
                .alias("nt_seq")
            )
        # singles: null nt_seq for nonsynonymous rows
        if singles_df is not None and len(singles_df) > 0 and "nt_seq" in singles_df.columns and "Nham_aa" in singles_df.columns:
            singles_df = singles_df.with_columns(
                pl.when(pl.col("Nham_aa").fill_null(0) > 0)
                .then(None).otherwise(pl.col("nt_seq"))
                .alias("nt_seq")
            )
        # doubles: null nt_seq entirely
        if doubles_df is not None and len(doubles_df) > 0 and "nt_seq" in doubles_df.columns:
            doubles_df = doubles_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("nt_seq"))

    # ---- Separate synonymous ----
    if "error_model" in all_data_df.columns:
        synonymous_df = all_data_df.filter(~pl.col("error_model"))
        all_data_df = all_data_df.filter(pl.col("error_model"))
    else:
        synonymous_df = pl.DataFrame()

    # ---- Wildtype ----
    wildtype_df = all_data_df.filter(pl.col("WT").fill_null(False))

    # ---- Remove synonymous WT from all_variants for coding ----
    if sequence_type == "coding":
        all_data_df = all_data_df.filter(
            pl.col("WT").fill_null(False) |
            (pl.col("Nham_aa").fill_null(0) > 0) |
            pl.col("indel").fill_null(False)
        )

    # ---- Add singles fitness to doubles ----
    if doubles_df is not None and len(doubles_df) > 0 and singles_df is not None and len(singles_df) > 0:
        doubles_df = _join_singles_to_doubles(doubles_df, singles_df)

    # ---- Finalise output tables ----
    singles_out, singles_mavedb_out = _finalise_singles(singles_df, sequence_type)
    doubles_out = _finalise_doubles(doubles_df, sequence_type)

    # ---- Write plain text files ----
    logger.info("Writing fitness output files to %s", output_path)

    _write_table(wildtype_df, output_path / "fitness_wildtype.txt")
    _write_table(singles_out, output_path / "fitness_singles.txt")
    _write_table(doubles_out, output_path / "fitness_doubles.txt")

    if sequence_type == "coding" and synonymous_df is not None and len(synonymous_df) > 0:
        _write_table(synonymous_df, output_path / "fitness_synonymous.txt")

    # MaveDB CSV
    if singles_mavedb_out is not None and len(singles_mavedb_out) > 0:
        singles_mavedb_out.write_csv(
            str(output_path / "fitness_singles_MaveDB.csv"),
            separator=",",
        )

    # ---- Parquet output (replaces .RData) ----
    parquet_path = output_path / f"{project_name}_fitness_replicates.parquet"
    tables = {
        "all_variants": all_data_df,
        "wildtype": wildtype_df,
        "singles": singles_out,
        "doubles": doubles_out,
    }
    if sequence_type == "coding" and synonymous_df is not None and len(synonymous_df) > 0:
        tables["synonymous"] = synonymous_df

    _write_parquet_bundle(tables, parquet_path)
    logger.info("Saved Parquet fitness bundle: %s", parquet_path)


# ---------------------------------------------------------------------------
# Inverse-variance merge
# ---------------------------------------------------------------------------

def _inverse_variance_merge(
    df: pl.DataFrame,
    reps: list[int],
    fitness_suffix: str,
    sigma_suffix: str,
    out_fitness: str,
    out_sigma: str,
) -> pl.DataFrame:
    """Compute weighted mean fitness = Σ(f/σ²) / Σ(1/σ²), σ = 1/sqrt(Σ(1/σ²)).

    Mirrors: R/dimsum__merge_fitness.R:39-67
    """
    f_cols = [f"fitness{E}{fitness_suffix}" for E in reps if f"fitness{E}{fitness_suffix}" in df.columns]
    s_cols = [f"sigma{E}{sigma_suffix}" for E in reps if f"sigma{E}{sigma_suffix}" in df.columns]

    if not f_cols or not s_cols:
        return df

    # Sum of f/sigma^2, ignoring null pairs (mirrors R na.rm=TRUE)
    numerator = pl.lit(0.0)
    denominator = pl.lit(0.0)
    for fc, sc in zip(f_cols, s_cols):
        valid = pl.col(fc).is_not_null() & pl.col(sc).is_not_null()
        w = pl.when(valid).then(1.0 / (pl.col(sc) ** 2)).otherwise(pl.lit(0.0))
        numerator = numerator + pl.when(valid).then(pl.col(fc) * (1.0 / (pl.col(sc) ** 2))).otherwise(pl.lit(0.0))
        denominator = denominator + w

    df = df.with_columns([
        pl.when(denominator > 0).then(numerator / denominator).otherwise(None).alias(out_fitness),
        pl.when(denominator > 0).then(1.0 / denominator.sqrt()).otherwise(None).alias(out_sigma),
    ])
    return df


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _join_singles_to_doubles(
    doubles_df: pl.DataFrame,
    singles_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add fitness1/sigma1/fitness2/sigma2 from singles to doubles table.

    Mirrors: R/dimsum__merge_fitness.R:183-186 using proper joins.
    """
    if "Pos" not in singles_df.columns or "Mut" not in singles_df.columns:
        return doubles_df
    if "fitness" not in singles_df.columns or "sigma" not in singles_df.columns:
        return doubles_df

    s_lookup = singles_df.select(["Pos", "Mut", "fitness", "sigma"])

    doubles_df = doubles_df.join(
        s_lookup.rename({"Pos": "Pos1", "Mut": "Mut1",
                          "fitness": "fitness1", "sigma": "sigma1"}),
        on=["Pos1", "Mut1"],
        how="left",
    )
    doubles_df = doubles_df.join(
        s_lookup.rename({"Pos": "Pos2", "Mut": "Mut2",
                          "fitness": "fitness2", "sigma": "sigma2"}),
        on=["Pos2", "Mut2"],
        how="left",
    )
    return doubles_df


def _finalise_singles(
    singles_df: pl.DataFrame | None,
    sequence_type: str,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Select and order columns for singles output table."""
    if singles_df is None or len(singles_df) == 0:
        return pl.DataFrame(), None

    if sequence_type == "coding":
        # Filter to single AA substitutions only
        if "Nham_aa" in singles_df.columns:
            singles_df = singles_df.filter(pl.col("Nham_aa") == 1)
        keep = [
            "Pos", "WT_AA", "Mut", "nt_seq", "aa_seq",
            "Nham_nt", "Nham_aa", "Nmut_codons", "STOP", "STOP_readthrough",
            "mean_count", "fitness", "sigma",
        ]
        out = singles_df.select([c for c in keep if c in singles_df.columns])

        # MaveDB
        if "Pos" in out.columns and "WT_AA" in out.columns and "Mut" in out.columns:
            hgvs = [
                f"p.{_AA_THREE.get(row['WT_AA'], row['WT_AA'])}{row['Pos']}{_AA_THREE.get(row['Mut'], row['Mut'])}"
                for row in out.iter_rows(named=True)
            ]
            mavedb = pl.DataFrame({
                "hgvs_pro": hgvs,
                "score": out["fitness"].to_list(),
                "se": out["sigma"].to_list(),
            })
        else:
            mavedb = None
    else:
        keep = ["Pos", "WT_AA", "Mut", "nt_seq", "Nham_nt", "mean_count", "fitness", "sigma"]
        out = singles_df.select([c for c in keep if c in singles_df.columns])
        # Rename WT_AA → WT_nt for noncoding
        if "WT_AA" in out.columns:
            out = out.rename({"WT_AA": "WT_nt"})

        # MaveDB for noncoding
        if "Pos" in out.columns and "WT_nt" in out.columns and "Mut" in out.columns:
            hgvs = [
                f"n.{row['Pos']}{row['WT_nt'].upper()}>{row['Mut'].upper()}"
                for row in out.iter_rows(named=True)
            ]
            mavedb = pl.DataFrame({
                "hgvs_nt": hgvs,
                "score": out["fitness"].to_list(),
                "se": out["sigma"].to_list(),
            })
        else:
            mavedb = None

    return out, mavedb


def _finalise_doubles(
    doubles_df: pl.DataFrame | None,
    sequence_type: str,
) -> pl.DataFrame:
    """Select and order columns for doubles output table."""
    if doubles_df is None or len(doubles_df) == 0:
        return pl.DataFrame()

    if sequence_type == "coding":
        keep = [
            "Pos1", "Pos2", "WT_AA1", "WT_AA2", "Mut1", "Mut2",
            "nt_seq", "aa_seq", "Nham_nt", "Nham_aa", "Nmut_codons",
            "STOP", "STOP_readthrough", "mean_count",
            "fitness1", "sigma1", "fitness2", "sigma2",
            "fitness_uncorr", "sigma_uncorr",
        ]
    else:
        keep = [
            "Pos1", "Pos2", "WT_AA1", "WT_AA2", "Mut1", "Mut2",
            "nt_seq", "Nham_nt", "mean_count",
            "fitness1", "sigma1", "fitness2", "sigma2",
            "fitness_uncorr", "sigma_uncorr",
        ]

    present = [c for c in keep if c in doubles_df.columns]
    out = doubles_df.select(present)
    # Rename WT_AA1/2 → WT_nt1/2 for noncoding
    if sequence_type != "coding":
        renames = {}
        if "WT_AA1" in out.columns:
            renames["WT_AA1"] = "WT_nt1"
        if "WT_AA2" in out.columns:
            renames["WT_AA2"] = "WT_nt2"
        if renames:
            out = out.rename(renames)
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_table(df: pl.DataFrame, path: Path) -> None:
    """Write a tab-separated text file (compatible with R write.table)."""
    if df is None or len(df) == 0:
        # Write empty file with header
        path.write_text("")
        return
    df.write_csv(str(path), separator="\t", null_value="NA")


def _write_parquet_bundle(tables: dict[str, pl.DataFrame], path: Path) -> None:
    """Write all result DataFrames to a single Parquet bundle directory."""
    bundle_dir = path.with_suffix("")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df is not None and len(df) > 0:
            df.write_parquet(str(bundle_dir / f"{name}.parquet"))
