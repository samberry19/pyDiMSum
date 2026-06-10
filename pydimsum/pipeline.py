"""Main pipeline orchestrator.

Mirrors: R/dimsum.R::dimsum() — stage gating, setup, stage dispatch.

Stages:
  0  Demultiplex (cutadapt)
  1  FastQC quality assessment
  2  Trim adapters (cutadapt)
  3  Align paired-end reads (VSEARCH) + count unique variants (starcode)
  4  Build and process variant table  (STEAM)
  5  Calculate fitness and model error (STEAM)
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from pydimsum.config import RunConfig
from pydimsum.io.designs import ExperimentDesign, load_synonym_sequences
from pydimsum.steam.aggregate import (
    aggregate_aa_variants,
    aggregate_aa_variants_fitness,
)
from pydimsum.steam.fitness import (
    add_dropout_pseudocount,
    calculate_fitness,
    filter_low_counts,
    normalise_fitness_by_generations,
)
from pydimsum.steam.merge import build_variant_table
from pydimsum.steam.merge_fitness import merge_fitness
from pydimsum.steam.mutations import identify_doubles, identify_singles
from pydimsum.steam.process_variants import process_variants
from pydimsum.steam.error_model import fit_error_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(config: RunConfig) -> None:
    """Execute the pyDiMSum pipeline according to config.start_stage / stop_stage.

    For M1: only the STEAM-only (count_path) path is supported.
    """
    # Create output directories
    config.project_path.mkdir(parents=True, exist_ok=True)
    config.tmp_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== pyDiMSum pipeline ===")
    logger.info("Project: %s", config.project_path)
    logger.info("Sequence type: %s (resolved: %s)", config.sequence_type, config.sequence_type_resolved)

    # ---- Load experiment design ----
    exp_design = ExperimentDesign(config.experiment_design_path, count_path=config.count_path)
    replicates = exp_design.replicates

    if config.retained_replicates != "all":
        retained = [int(r) for r in config.retained_replicates.split(",")]
        replicates = [r for r in replicates if r in retained]

    logger.info("Replicates: %s", replicates)

    # Require >= 2 replicates for normalisation and error model
    if len(replicates) < 2:
        logger.warning(
            "Only %d replicate(s) found. "
            "Disabling fitness normalisation and error model.",
            len(replicates),
        )
        config.fitness_normalise = False
        config.fitness_error_model = False

    # ---- Synonym sequences ----
    synonym_sequences = None
    if config.synonym_sequence_path is not None:
        synonym_sequences = load_synonym_sequences(config.synonym_sequence_path)

    # ---- WRAP stages (0-3) ----
    if config.count_path is not None:
        logger.info("Count file provided — running STEAM-only (stages 4-5).")
        _run_steam(config, exp_design, replicates, synonym_sequences)
    else:
        logger.info("No count file — running full pipeline (stages 0-5).")
        _run_wrap(config, exp_design)
        # After WRAP, build per-sample count files and run STEAM
        # The aligned_pair_unique files from tally.py feed into merge.py
        # Reload experiment design with updated FASTQ metadata
        _run_steam(config, exp_design, replicates, synonym_sequences)


def _run_wrap(
    config: RunConfig,
    exp_design: ExperimentDesign,
) -> None:
    """Execute WRAP stages 0–3 (demultiplex → fastqc → trim → align → tally).

    Updates exp_design.df in-place after each stage so that downstream
    STEAM stages can locate the per-sample count files.
    """
    from pydimsum.wrap import check_binaries, BinaryNotFoundError
    from pydimsum.wrap.demultiplex import run_demultiplex
    from pydimsum.wrap.fastqc import run_fastqc
    from pydimsum.wrap.trim import run_trim
    from pydimsum.wrap.align import run_align
    from pydimsum.wrap.tally import run_tally

    # Check required binaries before starting (avoid partial runs)
    stages_to_run = [s for s in range(0, 4) if config.start_stage <= s <= config.stop_stage]
    try:
        check_binaries(stages_to_run)
    except BinaryNotFoundError as e:
        raise RuntimeError(str(e)) from e

    df = exp_design.df  # mutable reference; we reassign after each stage

    # Stage 0: Demultiplex
    if config.start_stage <= 0 <= config.stop_stage:
        barcode_design_df = _load_barcode_design(config)
        demux_outpath = config.project_path / "1_demultiplex"
        df = run_demultiplex(config, df, demux_outpath, barcode_design_df)
        exp_design.df = df
        if config.stop_stage == 0:
            return

    # Stage 1: FastQC
    if config.start_stage <= 1 <= config.stop_stage:
        fastqc_outpath = config.project_path / "2_fastqc"
        df = run_fastqc(config, df, fastqc_outpath)
        exp_design.df = df
        if config.stop_stage == 1:
            return

    # Stage 2: Trim
    if config.start_stage <= 2 <= config.stop_stage:
        trim_outpath = config.project_path / "3_trim"
        df = run_trim(config, df, trim_outpath)
        exp_design.df = df
        if config.stop_stage == 2:
            return

    # Stage 3: Align + Tally
    if config.start_stage <= 3 <= config.stop_stage:
        align_outpath = config.project_path / "4_align"
        tally_outpath = config.project_path / "5_tally"
        df = run_align(config, df, align_outpath)
        df = run_tally(config, df, tally_outpath)
        exp_design.df = df
        if config.stop_stage == 3:
            return


def _load_barcode_design(config: RunConfig) -> "pl.DataFrame | None":
    """Load barcode design file if specified, else return None."""
    if config.barcode_design_path is None:
        return None
    import io
    path = config.barcode_design_path
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return pl.read_csv(io.BytesIO(raw), separator="\t", null_values=["", "NA"])


def _run_steam(
    config: RunConfig,
    exp_design: ExperimentDesign,
    replicates: list[int],
    synonym_sequences: list[str] | None,
) -> None:
    """Execute STEAM stages 4 and 5."""

    # ============================================================
    # Stage 4: Build and process variant table
    # ============================================================
    if config.start_stage <= 4 <= config.stop_stage:
        logger.info("--- Stage 4: Process variant sequences ---")

        # Build wide count table
        variant_df = build_variant_table(config, exp_design)

        # Process variants (filter, annotate, compute Hamming distances)
        processed = process_variants(variant_df, config)
        retained = processed.retained

        logger.info(
            "Retained %d variants (%d coding, %d indels)",
            len(retained),
            len(retained.filter(pl.col("indel").fill_null(False) == False)),
            len(retained.filter(pl.col("indel").fill_null(False))),
        )

        # Write intermediates
        _write_tsv(processed.indel, config.project_path / f"{config.project_name}_indel_variant_data_merge.tsv")
        _write_tsv(processed.rejected, config.project_path / f"{config.project_name}_rejected_variant_data_merge.tsv")
        _write_tsv(retained, config.project_path / f"{config.project_name}_variant_data_merge.tsv")

        # Save for stage 5
        _variant_data = retained
    else:
        # Load from previous run (not yet implemented)
        raise NotImplementedError("Resuming from stage 5 without stage 4 data not yet supported.")

    if config.stop_stage < 5:
        return

    # ============================================================
    # Stage 5: Calculate fitness and model error
    # ============================================================
    logger.info("--- Stage 5: Calculate fitness and model error ---")

    seq_type = config.sequence_type_resolved
    wt_nt = _variant_data.filter(pl.col("WT").fill_null(False))["nt_seq"].to_list()
    if wt_nt:
        wt_nt_seq = wt_nt[0]
    else:
        wt_nt_seq = config.wt_nt_seq

    # WT AA sequence
    from pydimsum.steam.sequences import translate_sequences_fast
    wt_aa_seq = translate_sequences_fast([wt_nt_seq])[0] if seq_type == "coding" else ""

    # ---- Filter low counts ----
    nf_data = filter_low_counts(_variant_data, config, replicates)
    nf_data = nf_data.with_columns(pl.lit(True).alias("error_model"))

    # ---- AA aggregation (mixedSubstitutions + coding) ----
    if seq_type == "coding" and config.mixed_substitutions:
        nf_data = aggregate_aa_variants(nf_data, replicates, synonym_sequences)
    else:
        nf_data = nf_data.with_columns(pl.col("nt_seq").alias("merge_seq"))

    # ---- Aggregate biological output replicates (already done in merge.py) ----
    # count_e{E}_s1 is already summed; just ensure merge_seq is set

    # ---- Fit error model ----
    logger.info("Fitting error model...")
    model_result = fit_error_model(
        df=nf_data,
        replicates=replicates,
        fitness_normalise=config.fitness_normalise,
        fitness_error_model=config.fitness_error_model,
        num_cores=config.num_cores,
        seed=1234567,
    )

    # ---- Add dropout pseudocounts ----
    nf_data_pseudo = add_dropout_pseudocount(nf_data, config, replicates)

    # ---- Calculate fitness ----
    nff_data = calculate_fitness(
        df=nf_data_pseudo,
        config=config,
        replicates=replicates,
        error_model_df=model_result["error_model"],
        norm_model_df=model_result["norm_model"],
    )

    # ---- AA aggregation for fitness (mixedSubstitutions=False, coding) ----
    if seq_type == "coding" and not config.mixed_substitutions:
        nff_data = aggregate_aa_variants_fitness(nff_data, replicates)
    else:
        if seq_type != "coding":
            nff_data = nff_data.with_columns(pl.col("nt_seq").alias("merge_seq"))

    # ---- Identify WT ----
    wildtype_df = nff_data.filter(pl.col("WT").fill_null(False) & pl.col("error_model").fill_null(True))

    # ---- Identify singles and doubles ----
    logger.info("Identifying single and double mutations...")
    singles_df = identify_singles(
        df=nff_data.filter(pl.col("error_model").fill_null(True)),
        sequence_type=seq_type,
        wt_nt_seq=wt_nt_seq,
        wt_aa_seq=wt_aa_seq,
    )
    doubles_df = identify_doubles(
        df=nff_data.filter(pl.col("error_model").fill_null(True)),
        singles_df=singles_df,
        sequence_type=seq_type,
        wt_nt_seq=wt_nt_seq,
        wt_aa_seq=wt_aa_seq,
    )

    logger.info(
        "Singles: %d, Doubles: %d",
        len(singles_df) if singles_df is not None else 0,
        len(doubles_df) if doubles_df is not None else 0,
    )

    # ---- Normalise by generations (optional) ----
    gen_check = exp_design.df.filter(
        pl.col("selection_id") == 1
    )["generations"].drop_nulls()
    has_generations = len(gen_check) == len(
        exp_design.df.filter(pl.col("selection_id") == 1)
    )

    if has_generations:
        logger.info("Normalising fitness by number of generations...")
        nff_data = normalise_fitness_by_generations(
            nff_data, exp_design.df, replicates, fitness_suffix="_uncorr"
        )
        if singles_df is not None and len(singles_df) > 0:
            singles_df = normalise_fitness_by_generations(
                singles_df, exp_design.df, replicates
            )
        if doubles_df is not None and len(doubles_df) > 0:
            doubles_df = normalise_fitness_by_generations(
                doubles_df, exp_design.df, replicates, fitness_suffix="_uncorr"
            )

    # ---- Save error model and normalisation model ----
    if model_result["error_model"] is not None:
        model_result["error_model"].write_csv(
            str(config.tmp_path / "errormodel.txt"), separator="\t"
        )
    if model_result["norm_model"] is not None:
        model_result["norm_model"].write_csv(
            str(config.tmp_path / "normalisationmodel.txt"), separator="\t"
        )

    # ---- Merge fitness and write outputs ----
    logger.info("Merging fitness estimates and writing outputs...")
    merge_fitness(
        all_data_df=nff_data,
        singles_df=singles_df if (singles_df is not None and len(singles_df) > 0) else pl.DataFrame(),
        doubles_df=doubles_df if (doubles_df is not None and len(doubles_df) > 0) else pl.DataFrame(),
        replicates=replicates,
        sequence_type=seq_type,
        output_path=config.project_path,
        project_name=config.project_name,
        bayesian_double_fitness=config.bayesian_double_fitness,
    )

    logger.info("=== pyDiMSum pipeline complete ===")
    logger.info("Output files written to: %s", config.project_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tsv(df: pl.DataFrame, path: Path) -> None:
    if df is None or len(df) == 0:
        path.write_text("")
        return
    df.write_csv(str(path), separator="\t", null_value="NA")
