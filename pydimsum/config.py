"""RunConfig — validated configuration dataclass for pyDiMSum.

Mirrors DiMSum's ~60 CLI options (DiMSum.R:25-83 / dimsum.R:66-124).
Only STEAM-relevant options are required for M1 (count-file mode).
WRAP options default to None and are validated only when needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# IUPAC codes used in permittedSequences validation
# ---------------------------------------------------------------------------
_IUPAC_UPPER = set("ACGTRYSWKMBDHVN")
_IUPAC_LOWER = set("acgtryswkmbdhvn")
_NT_UPPER = set("ACGT")
_NT_LOWER = set("acgt")
_NT_MIXED = set("ACGTacgt")  # wildtypeSequence: upper=variable, lower=constant


def _parse_min_count_arg(value: str) -> Union[int, dict]:
    """Parse a fitnessMin*Count* argument.

    Accepts either:
      - A single integer string ("0", "10")
      - A comma-separated list of ``editdist:threshold`` pairs
        e.g. "0:5,1:10,2:30"

    Returns an int or a dict mapping Nham_nt (int) -> threshold (int).

    Mirrors dimsum__parse_minimum_read_count_arguments.R.
    """
    value = value.strip()
    if ":" not in value:
        return int(value)
    result = {}
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid count threshold format {value!r}. "
                "Expected 'integer' or 'dist:threshold,...'"
            )
        result[int(parts[0])] = int(parts[1])
    return result


@dataclass
class WTEntry:
    """One wildtype reference (mutation mode)."""

    name: str
    coded_seq: str
    """Case-coded sequence: upper = variable, lower = internal constant region."""
    permitted: str = ""
    """IUPAC string over the variable positions only."""

    @property
    def seq(self) -> str:
        """Lower-cased full sequence (variable + constant)."""
        return self.coded_seq.lower()

    @property
    def variable_seq(self) -> str:
        return "".join(b for b in self.coded_seq if b in _NT_UPPER)

    @property
    def has_constant_region(self) -> bool:
        return any(b in _NT_LOWER for b in self.coded_seq)


@dataclass
class RunConfig:
    """Validated pipeline configuration.

    Attributes mirror DiMSum's R argument names converted to snake_case.
    Defaults match DiMSum's defaults.
    """

    # ---- Required ----
    experiment_design_path: Path
    """Path to the tab-separated experimental design file."""

    wildtype_sequence: str = ""
    """WT nucleotide sequence (A/C/G/T upper-case = variable,
    lower-case = internal constant region to remove).
    Required in mutation mode; optional (defaults to "") in enrichment_mode.
    May be a comma-separated list of several WTs; the first is the primary WT
    (fitness reference).  After validation this holds the primary WT only."""

    wildtype_names: str | None = None
    """Optional comma-separated names for the WTs (default WT1, WT2, ...)."""

    # ---- Input / output ----
    output_path: Path = field(default_factory=lambda: Path("."))
    project_name: str = "DiMSum_Project"
    count_path: Path | None = None
    fastq_file_dir: Path | None = None
    fastq_file_extension: str = ".fastq"
    retain_intermediate_files: bool = False
    start_stage: int = 0
    stop_stage: int = 5
    num_cores: int = 1

    # ---- FASTQ / library ----
    gzipped: bool = True
    stranded: bool = True
    paired: bool = True
    reverse_complement: bool = False
    experiment_design_pair_duplicates: bool = False

    # ---- Sequence processing ----
    sequence_type: str = "auto"          # "auto" | "coding" | "noncoding"
    mutagenesis_type: str = "random"     # "random" | "codon"
    permitted_sequences: str | None = None
    """IUPAC string covering only the variable positions in wildtype_sequence.
    Default (None) → 'N' repeated for each variable position (all substitutions
    permitted)."""
    max_substitutions: int = 2
    mixed_substitutions: bool = False
    indels: str = "none"                 # "none" | "all" | comma-sep lengths
    trans_library: bool = False
    trans_library_reverse_complement: bool = False

    # ---- Barcodes ----
    barcode_design_path: Path | None = None
    barcode_error_rate: float = 0.25
    barcode_identity_path: Path | None = None

    # ---- Fitness / analysis ----
    fitness_min_input_count_all: str = "0"
    fitness_min_input_count_any: str = "0"
    fitness_min_output_count_all: str = "0"
    fitness_min_output_count_any: str = "0"
    fitness_normalise: bool = True
    fitness_error_model: bool = True
    fitness_dropout_pseudocount: int = 0
    retained_replicates: str = "all"

    # ---- Synonym sequences ----
    synonym_sequence_path: Path | None = None

    # ---- Enrichment / library mode ----
    enrichment_mode: bool = False
    """If True, bypass all mutation-centric filters and compute per-sequence
    log(out/in) enrichment.  Wildtype sequence is not required."""
    enrichment_normalise: str = "median"
    """Normalisation strategy for enrichment mode.
    One of: none | median | total | reference | spikein."""
    enrichment_reference_id: str | None = None
    """nt_seq string of the reference sequence (required when
    enrichment_normalise == 'reference')."""
    enrichment_spikein_ids: str | None = None
    """Comma-separated nt_seq strings for spike-in sequences
    (used when enrichment_normalise == 'spikein')."""

    # ---- Bayesian doubles (disabled / stub) ----
    bayesian_double_fitness: bool = False
    bayesian_double_fitness_lam_d: float = 0.025
    fitness_high_confidence_count: int = 10
    fitness_double_high_confidence_count: int = 50

    # ---- Cutadapt (WRAP, ignored in STEAM-only mode) ----
    cutadapt_5_first: str | None = None
    cutadapt_5_second: str | None = None
    cutadapt_3_first: str | None = None
    cutadapt_3_second: str | None = None
    cutadapt_cut_5_first: int | None = None
    cutadapt_cut_5_second: int | None = None
    cutadapt_cut_3_first: int | None = None
    cutadapt_cut_3_second: int | None = None
    cutadapt_min_length: int = 50
    cutadapt_max_length: int | None = None
    cutadapt_options: str | None = None
    cutadapt_error_rate: float = 0.2
    cutadapt_overlap: int = 3

    # ---- VSEARCH (WRAP) ----
    vsearch_min_qual: int = 30
    vsearch_max_qual: int = 41
    vsearch_max_ee: float = 0.5
    vsearch_min_ovlen: int = 10

    # ---- Internal (derived after validation) ----
    wildtypes: list[WTEntry] = field(default_factory=list, init=False, repr=False)
    _sequence_type_resolved: str = field(default="", init=False, repr=False)
    _wt_seq_upper: str = field(default="", init=False, repr=False)
    """WT sequence with constant-region bases still lower-case."""
    _wt_seq_variable: str = field(default="", init=False, repr=False)
    """WT sequence with constant-region bases removed (variable positions only)."""
    _indel_lengths: list[int] | None = field(default=None, init=False, repr=False)
    _fitness_min_input_count_all_parsed: Union[int, dict] = field(
        default=0, init=False, repr=False
    )
    _fitness_min_input_count_any_parsed: Union[int, dict] = field(
        default=0, init=False, repr=False
    )
    _fitness_min_output_count_all_parsed: Union[int, dict] = field(
        default=0, init=False, repr=False
    )
    _fitness_min_output_count_any_parsed: Union[int, dict] = field(
        default=0, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Run all validation checks and populate derived fields."""
        self._validate_paths()
        if self.enrichment_mode:
            self._validate_enrichment_mode()
        else:
            self._validate_wt_sequence()
            self._validate_sequence_type()
            self._validate_permitted_sequences()
        self._validate_indels()
        self._validate_count_thresholds()
        self._validate_misc()

    def _validate_paths(self) -> None:
        self.output_path = Path(self.output_path)
        self.experiment_design_path = Path(self.experiment_design_path)
        if not self.experiment_design_path.exists():
            raise FileNotFoundError(
                f"experimentDesignPath not found: {self.experiment_design_path}"
            )
        if self.count_path is not None:
            self.count_path = Path(self.count_path)
            if not self.count_path.exists():
                raise FileNotFoundError(
                    f"countPath not found: {self.count_path}"
                )
        if self.synonym_sequence_path is not None:
            self.synonym_sequence_path = Path(self.synonym_sequence_path)
            if not self.synonym_sequence_path.exists():
                raise FileNotFoundError(
                    f"synonymSequencePath not found: {self.synonym_sequence_path}"
                )

    def _validate_wt_sequence(self) -> None:
        if not self.wildtype_sequence and not self.enrichment_mode:
            raise ValueError("wildtypeSequence is required")
        seqs = [x.strip() for x in self.wildtype_sequence.split(",")]
        for sq in seqs:
            if not sq or not set(sq).issubset(_NT_MIXED):
                invalid = set(sq) - _NT_MIXED or {"<empty>"}
                raise ValueError(
                    f"wildtypeSequence contains invalid characters: {invalid}. "
                    "Only A/C/G/T (variable) and a/c/g/t (constant) are allowed."
                )
        if len({s_.lower() for s_ in seqs}) != len(seqs):
            raise ValueError("wildtypeSequence contains duplicate sequences")
        if self.wildtype_names:
            names = [x.strip() for x in self.wildtype_names.split(",")]
            if len(names) != len(seqs):
                raise ValueError(
                    f"wildtypeNames has {len(names)} entries but "
                    f"{len(seqs)} wildtype sequences were given"
                )
        else:
            names = [f"WT{i + 1}" for i in range(len(seqs))]
        if len(set(names)) != len(names) or not all(names):
            raise ValueError("wildtypeNames must be unique and non-empty")
        self.wildtypes = [WTEntry(n, sq) for n, sq in zip(names, seqs)]
        # Primary WT drives the single-WT convenience fields
        self.wildtype_sequence = self.wildtypes[0].coded_seq
        self._wt_seq_upper = self.wildtype_sequence  # keep case
        self._wt_seq_variable = self.wildtypes[0].variable_seq

    def _validate_enrichment_mode(self) -> None:
        """Validate enrichment-mode-specific config and set derived fields."""
        _VALID_NORM = {"none", "median", "total", "reference", "spikein"}
        if self.enrichment_normalise not in _VALID_NORM:
            raise ValueError(
                f"enrichment_normalise must be one of {_VALID_NORM}, "
                f"got {self.enrichment_normalise!r}"
            )
        if self.enrichment_normalise == "reference" and not self.enrichment_reference_id:
            raise ValueError(
                "enrichment_reference_id is required when "
                "enrichment_normalise == 'reference'"
            )
        # Sequence type: resolve without attempting WT translation
        if self.sequence_type == "auto":
            # Default to noncoding in enrichment mode (no single WT to probe)
            self._sequence_type_resolved = "noncoding"
        else:
            self._sequence_type_resolved = self.sequence_type
        # WT fields left empty — not needed
        self._wt_seq_upper = ""
        self._wt_seq_variable = ""
        # permitted_sequences not meaningful in enrichment mode
        self.permitted_sequences = None

    def _validate_sequence_type(self) -> None:
        if self.sequence_type not in ("auto", "coding", "noncoding"):
            raise ValueError(
                f"sequence_type must be 'auto', 'coding', or 'noncoding', "
                f"got {self.sequence_type!r}"
            )
        if self.sequence_type == "auto":
            # Detect per WT: try translating the lower-cased WT sequence; if it
            # produces no premature STOP -> coding.
            from Bio.Seq import Seq

            resolved = set()
            for wt in self.wildtypes:
                nt = wt.seq
                if len(nt) % 3 == 0:
                    prot = str(Seq(nt).translate())
                    # premature stop = stop before the last position
                    resolved.add("noncoding" if "*" in prot[:-1] else "coding")
                else:
                    resolved.add("noncoding")
            if len(resolved) > 1:
                raise ValueError(
                    "Wildtype sequences resolve to different sequence types "
                    "(coding and noncoding); set sequence_type explicitly."
                )
            self._sequence_type_resolved = resolved.pop()
        else:
            self._sequence_type_resolved = self.sequence_type

    def _validate_permitted_sequences(self) -> None:
        raw = self.permitted_sequences
        if raw is None:
            parts: list[str | None] = [None] * len(self.wildtypes)
        else:
            parts = [x.strip() for x in raw.split(",")]
            if len(parts) == 1 and len(self.wildtypes) > 1:
                parts = parts * len(self.wildtypes)
            if len(parts) != len(self.wildtypes):
                raise ValueError(
                    f"permittedSequences has {len(parts)} entries but "
                    f"{len(self.wildtypes)} wildtype sequences were given"
                )
        for wt, ps in zip(self.wildtypes, parts):
            n_variable = len(wt.variable_seq)
            if ps is None:
                # Default: all substitutions at every variable position
                wt.permitted = "N" * n_variable
                continue
            ps = ps.upper()
            if len(ps) != n_variable:
                raise ValueError(
                    f"permittedSequences length ({len(ps)}) must equal "
                    f"the number of variable (upper-case) positions in "
                    f"wildtypeSequence {wt.name} ({n_variable})"
                )
            if not set(ps).issubset(_IUPAC_UPPER):
                invalid = set(ps) - _IUPAC_UPPER
                raise ValueError(
                    f"permittedSequences contains invalid IUPAC characters: {invalid}"
                )
            wt.permitted = ps
        # Primary WT's permitted string drives the single-WT convenience field
        self.permitted_sequences = self.wildtypes[0].permitted

    def _validate_indels(self) -> None:
        indels = self.indels.strip().lower()
        if indels == "none":
            self._indel_lengths = None  # no indels retained
        elif indels == "all":
            self._indel_lengths = []    # empty = all lengths
        else:
            try:
                self._indel_lengths = [int(x) for x in indels.split(",")]
            except ValueError:
                raise ValueError(
                    f"indels must be 'none', 'all', or a comma-separated "
                    f"list of lengths, got {self.indels!r}"
                )

    def _validate_count_thresholds(self) -> None:
        self._fitness_min_input_count_all_parsed = _parse_min_count_arg(
            self.fitness_min_input_count_all
        )
        self._fitness_min_input_count_any_parsed = _parse_min_count_arg(
            self.fitness_min_input_count_any
        )
        self._fitness_min_output_count_all_parsed = _parse_min_count_arg(
            self.fitness_min_output_count_all
        )
        self._fitness_min_output_count_any_parsed = _parse_min_count_arg(
            self.fitness_min_output_count_any
        )

    def _validate_misc(self) -> None:
        if self.trans_library and not self.paired:
            raise ValueError(
                "trans_library requires paired=True "
                "(only paired-end trans libraries are supported)"
            )
        if not self.stranded and not self.paired:
            raise ValueError(
                "stranded=False requires paired=True "
                "(unstranded read-swap is only supported for paired-end libraries)"
            )
        if self.max_substitutions < 1:
            raise ValueError("max_substitutions must be >= 1")
        if not (0.0 <= self.barcode_error_rate <= 1.0):
            raise ValueError("barcode_error_rate must be in [0, 1]")
        if self.num_cores < 1:
            raise ValueError("num_cores must be >= 1")
        if self.mutagenesis_type not in ("random", "codon"):
            raise ValueError(
                f"mutagenesis_type must be 'random' or 'codon', "
                f"got {self.mutagenesis_type!r}"
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def project_path(self) -> Path:
        return self.output_path / self.project_name

    @property
    def tmp_path(self) -> Path:
        return self.project_path / "tmp"

    @property
    def retain_indels(self) -> bool:
        """Whether any indel variants should be retained."""
        return self._indel_lengths is not None

    @property
    def sequence_type_resolved(self) -> str:
        return self._sequence_type_resolved

    @property
    def wt_nt_seq(self) -> str:
        """Lower-cased WT nucleotide sequence (variable + constant)."""
        return self.wildtype_sequence.lower()

    @property
    def wt_variable_seq(self) -> str:
        """WT sequence with only variable positions (upper-case A/C/G/T)."""
        return self._wt_seq_variable

    @property
    def has_constant_region(self) -> bool:
        """True if the WT sequence contains internal constant region bases."""
        return any(b in _NT_LOWER for b in self.wildtype_sequence)
