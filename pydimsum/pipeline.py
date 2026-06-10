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
from pydimsum.steam.library import (
    calculate_enrichment,
    process_library_variants,
    write_enrichment_outputs,
)
from pydimsum.steam.merge import build_variant_table
from pydimsum.steam.merge_fitness import merge_fitness
from pydimsum.steam.mutations import identify_doubles, identify_singles
from pydimsum.steam.process_variants import process_variants
from pydimsum.steam.error_model import fit_error_model, _fit_normalisation

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

    # Require >= 2 replicates for normalisation and error model (mutation mode)
    # In enrichment mode, <2 replicates just disables the optional scale/shift step.
    if len(replicates) < 2 and not config.enrichment_mode:
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


def _run_enrichment_steam(
    config: RunConfig,
    exp_design: ExperimentDesign,
    replicates: list[int],
) -> None:
    """Execute stages 4–5 in enrichment (library) mode.

    Sequence-agnostic: no WT assumption, no Hamming/substitution filters.
    Produces enrichment_variant_data.txt + Parquet bundle.
    """
    logger.info("=== Enrichment mode ===")

    # Stage 4: Build count table + annotate
    if config.start_stage <= 4 <= config.stop_stage:
        logger.info("--- Stage 4 (enrichment): load and annotate library ---")
        variant_df = build_variant_table(config, exp_design)
        lib_df = process_library_variants(variant_df, config)
        logger.info("Library variants: %d sequences", len(lib_df))

        _write_tsv(lib_df, config.project_path / f"{config.project_name}_library_variant_data.tsv")
    else:
        raise NotImplementedError("Resuming enrichment from stage 5 without stage 4 not yet supported.")

    if config.stop_stage < 5:
        return

    # Stage 5: Calculate enrichment
    logger.info("--- Stage 5 (enrichment): calculate enrichment scores ---")

    # Filter low counts (reuses mutation-path helper; Nham_nt=0 means only
    # integer thresholds apply — editdist:threshold form is unsupported in
    # enrichment mode and guarded in config validation)
    lib_df = filter_low_counts(lib_df, config, replicates)
    lib_df = lib_df.with_columns(pl.lit(True).alias("error_model"))

    # Dropout pseudocounts
    lib_df = add_dropout_pseudocount(lib_df, config, replicates)

    # Optional replicate scale/shift normalisation (≥2 replicates only)
    norm_model_df: pl.DataFrame | None = None
    if len(replicates) >= 2:
        try:
            # Build a temporary enrichment table for fitting (no WT filtering)
            work_for_norm = lib_df
            # Add has_all flag (needed by _fit_normalisation)
            has_all_expr = pl.lit(True)
            for E in replicates:
                has_all_expr = has_all_expr & (
                    pl.col(f"count_e{E}_s0") > 0
                ) & (
                    pl.col(f"count_e{E}_s1") > 0
                )
            work_for_norm = work_for_norm.with_columns(has_all_expr.alias("all_reads"))

            # Compute raw fitness for each replicate (for normalisation fitting)
            for E in replicates:
                s0 = f"count_e{E}_s0"
                s1 = f"count_e{E}_s1"
                work_for_norm = work_for_norm.with_columns(
                    pl.when(
                        (pl.col(s0) > 0) & (pl.col(s1) > 0)
                    ).then(
                        (pl.col(s1).cast(pl.Float64) / pl.col(s0).cast(pl.Float64)).log()
                    ).otherwise(None)
                    .alias(f"fitness{E}")
                )

            # Mark above-threshold (1st percentile — reuse same logic)
            import numpy as np
            fitness_cols = [f"fitness{E}" for E in replicates]
            above_threshold_data = work_for_norm.filter(pl.col("all_reads"))
            flat_fitness = np.concatenate([
                above_threshold_data[fc].drop_nulls().to_numpy()
                for fc in fitness_cols
            ])
            if len(flat_fitness) > 0:
                input_count_threshold = float(np.exp(-np.percentile(flat_fitness, 1)))
                above_thresh_expr = pl.lit(True)
                for E in replicates:
                    above_thresh_expr = above_thresh_expr & (
                        pl.col(f"count_e{E}_s0").cast(pl.Float64) > input_count_threshold
                    )
                work_for_norm = work_for_norm.with_columns(
                    above_thresh_expr.alias("input_above_threshold")
                )
                norm_model_df = _fit_normalisation(work_for_norm, replicates)
                logger.info("Replicate normalisation model: %s", norm_model_df.to_dict(as_series=False))
            else:
                logger.warning("No sequences with reads in all replicates — skipping normalisation fit.")
        except Exception as exc:
            logger.warning("Replicate normalisation failed (%s) — using identity.", exc)
            norm_model_df = None

    lib_df = calculate_enrichment(lib_df, config, replicates, norm_model_df)

    # Optional generation normalisation
    gen_check = exp_design.df.filter(pl.col("selection_id") == 1)["generations"].drop_nulls()
    has_generations = len(gen_check) == len(exp_design.df.filter(pl.col("selection_id") == 1))
    if has_generations:
        logger.info("Normalising enrichment by number of generations...")
        # normalise_fitness_by_generations operates on fitness{E}_uncorr columns;
        # our columns are enrichment{E}_uncorr — rename, normalise, rename back
        rename_to = {f"enrichment{E}_uncorr": f"fitness{E}_uncorr" for E in replicates}
        rename_back = {v: k for k, v in rename_to.items()}
        lib_df = lib_df.rename(rename_to)
        lib_df = normalise_fitness_by_generations(
            lib_df, exp_design.df, replicates, fitness_suffix="_uncorr"
        )
        lib_df = lib_df.rename(rename_back)

    # Save normalisation model if fitted
    if norm_model_df is not None:
        norm_model_df.write_csv(
            str(config.tmp_path / "normalisationmodel_enrichment.txt"), separator="\t"
        )

    write_enrichment_outputs(lib_df, replicates, config)
    logger.info("=== pyDiMSum enrichment pipeline complete ===")
    logger.info("Output files written to: %s", config.project_path)


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
    # Enrichment mode: bypass mutation-centric processing
    # ============================================================
    if config.enrichment_mode:
        _run_enrichment_steam(config, exp_design, replicates)
        return

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
