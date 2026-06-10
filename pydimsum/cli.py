"""pyDiMSum command-line interface.

Maps ~60 DiMSum CLI arguments (DiMSum.R:25-83) to RunConfig and runs the
pipeline.  Uses typer for modern argument handling.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from pydimsum import __version__

app = typer.Typer(
    name="pydimsum",
    help="pyDiMSum — Python reimplementation of the DiMSum DMS analysis pipeline.",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


@app.command()
def main(
    # ---- Required ----
    experiment_design_path: Path = typer.Option(
        ...,
        "--experiment_design_path", "--experimentDesignPath",
        help="Path to tab-separated experimental design file.",
    ),
    wildtype_sequence: str = typer.Option(
        ...,
        "--wildtype_sequence", "--wildtypeSequence",
        help="WT nucleotide sequence. Upper-case = variable, lower-case = constant region.",
    ),
    # ---- Input / output ----
    output_path: Path = typer.Option(
        Path("."),
        "--output_path", "--outputPath",
        help="Output directory.",
    ),
    project_name: str = typer.Option(
        "DiMSum_Project",
        "--project_name", "--projectName",
        help="Project name and subdirectory for results.",
    ),
    count_path: Optional[Path] = typer.Option(
        None,
        "--count_path", "--countPath",
        help="Path to variant count file (STEAM-only mode).",
    ),
    start_stage: int = typer.Option(0, "--start_stage", "--startStage"),
    stop_stage: int = typer.Option(5, "--stop_stage", "--stopStage"),
    num_cores: int = typer.Option(1, "--num_cores", "--numCores"),
    retain_intermediate_files: bool = typer.Option(False, "--retain_intermediate_files"),
    # ---- Sequence processing ----
    sequence_type: str = typer.Option("auto", "--sequence_type", "--sequenceType"),
    mutagenesis_type: str = typer.Option("random", "--mutagenesis_type", "--mutagenesisType"),
    permitted_sequences: Optional[str] = typer.Option(None, "--permitted_sequences", "--permittedSequences"),
    max_substitutions: int = typer.Option(2, "--max_substitutions", "--maxSubstitutions"),
    mixed_substitutions: bool = typer.Option(False, "--mixed_substitutions", "--mixedSubstitutions"),
    indels: str = typer.Option("none", "--indels"),
    reverse_complement: bool = typer.Option(False, "--reverse_complement", "--reverseComplement"),
    # ---- Fitness / analysis ----
    fitness_min_input_count_all: str = typer.Option("0", "--fitness_min_input_count_all", "--fitnessMinInputCountAll"),
    fitness_min_input_count_any: str = typer.Option("0", "--fitness_min_input_count_any", "--fitnessMinInputCountAny"),
    fitness_min_output_count_all: str = typer.Option("0", "--fitness_min_output_count_all", "--fitnessMinOutputCountAll"),
    fitness_min_output_count_any: str = typer.Option("0", "--fitness_min_output_count_any", "--fitnessMinOutputCountAny"),
    fitness_normalise: bool = typer.Option(True, "--fitness_normalise/--no_fitness_normalise", "--fitnessNormalise"),
    fitness_error_model: bool = typer.Option(True, "--fitness_error_model/--no_fitness_error_model", "--fitnessErrorModel"),
    fitness_dropout_pseudocount: int = typer.Option(0, "--fitness_dropout_pseudocount", "--fitnessDropoutPseudocount"),
    retained_replicates: str = typer.Option("all", "--retained_replicates", "--retainedReplicates"),
    # ---- Barcodes ----
    barcode_design_path: Optional[Path] = typer.Option(None, "--barcode_design_path", "--barcodeDesignPath"),
    barcode_error_rate: float = typer.Option(0.25, "--barcode_error_rate", "--barcodeErrorRate"),
    barcode_identity_path: Optional[Path] = typer.Option(None, "--barcode_identity_path", "--barcodeIdentityPath"),
    # ---- Synonym sequences ----
    synonym_sequence_path: Optional[Path] = typer.Option(None, "--synonym_sequence_path", "--synonymSequencePath"),
    # ---- Misc ----
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    version: bool = typer.Option(False, "--version", callback=None, is_eager=True),
) -> None:
    """Run the pyDiMSum pipeline."""
    if version:
        typer.echo(f"pyDiMSum {__version__}")
        raise typer.Exit()

    _setup_logging(verbose)

    from pydimsum.config import RunConfig
    from pydimsum.pipeline import run_pipeline

    try:
        config = RunConfig(
            experiment_design_path=experiment_design_path,
            wildtype_sequence=wildtype_sequence,
            output_path=output_path,
            project_name=project_name,
            count_path=count_path,
            start_stage=start_stage,
            stop_stage=stop_stage,
            num_cores=num_cores,
            retain_intermediate_files=retain_intermediate_files,
            sequence_type=sequence_type,
            mutagenesis_type=mutagenesis_type,
            permitted_sequences=permitted_sequences,
            max_substitutions=max_substitutions,
            mixed_substitutions=mixed_substitutions,
            indels=indels,
            reverse_complement=reverse_complement,
            fitness_min_input_count_all=fitness_min_input_count_all,
            fitness_min_input_count_any=fitness_min_input_count_any,
            fitness_min_output_count_all=fitness_min_output_count_all,
            fitness_min_output_count_any=fitness_min_output_count_any,
            fitness_normalise=fitness_normalise,
            fitness_error_model=fitness_error_model,
            fitness_dropout_pseudocount=fitness_dropout_pseudocount,
            retained_replicates=retained_replicates,
            barcode_design_path=barcode_design_path,
            barcode_error_rate=barcode_error_rate,
            barcode_identity_path=barcode_identity_path,
            synonym_sequence_path=synonym_sequence_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1)

    try:
        run_pipeline(config)
    except NotImplementedError as exc:
        typer.echo(f"Not implemented: {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed: %s", exc)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
