# pyDiMSum

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/d24065d9-6d59-45ac-ba91-a1ee376e58ca" />

A Python reimplementation of the [DiMSum](https://github.com/lehner-lab/DiMSum) deep mutational scanning analysis pipeline (Faure et al., *Genome Biology* 2020).

## Why?

DiMSum users report that the R version can be a **memory hog and slow** on large datasets. The bottlenecks are well-understood:

| Problem | R location | Python fix |
|---------|-----------|------------|
| Per-variant Hamming (strsplit per row) | `dimsum__hamming_distance.R` | NumPy `(n,L) uint8` matrix; O(n) column-wise diff |
| Constant-region loop (column-rewrite per base) | `dimsum__remove_internal_constant_region.R` | Boolean mask slice of the matrix |
| Iterative pairwise `merge(all=T)` for counts | `dimsum_stage_merge.R:63-99` | Single Polars `pivot` |
| Repeated `data.table::copy()` | `process_merged_variants.R` | Zero-copy Polars boolean filters |
| 100 bootstraps × 20 NLS retries | `dimsum__fit_error_model_bootstrap.R` | `scipy.least_squares` with 3 multi-starts |

**Demo result (40 591 variants, 4 replicates, single core):**

| | Time |
|--|------|
| R DiMSum 1.4 | ~46 s |
| pyDiMSum | ~6 s |

## Installation

```bash
pip install -e .
```

Requirements: Python ≥ 3.11, polars, numpy, scipy, biopython, pyarrow, typer.

## Quick start

```bash
pydimsum \
  --experiment_design_path experimentDesign.txt \
  --wildtype_sequence ATGCGT... \
  --count_path countFile.txt \
  --output_path results/ \
  --project_name MyProject
```

### STEAM-only (count-file mode — no FASTQ processing required)

Provide `--count_path` to skip stages 0–3 (WRAP) and go directly from variant
counts to fitness scores. This is the fastest path and requires no external
binaries (cutadapt, VSEARCH, starcode, FastQC).

## Output files

| File | Contents |
|------|----------|
| `fitness_singles.txt` | Single-mutant fitness and sigma |
| `fitness_doubles.txt` | Double-mutant fitness (uncorrected) with per-single references |
| `fitness_wildtype.txt` | WT variant fitness |
| `fitness_synonymous.txt` | Synonymous (silent) variant fitness (coding sequences) |
| `fitness_singles_MaveDB.csv` | MaveDB-compatible CSV |
| `*_fitness_replicates/` | Parquet bundle with all tables (replaces `.RData`) |

## Architecture

```
pydimsum/
  config.py           # RunConfig dataclass (~60 DiMSum args) + validation
  pipeline.py         # Orchestrator: stage gating, STEAM-only path
  io/
    designs.py        # ExperimentDesign reader
    counts.py         # Variant count file reader/validator
  steam/
    merge.py          # Build wide count table (single-pass pivot)
    sequences.py      # Vectorized NumPy: Hamming, translate, STOP, permitted mask
    process_variants.py  # Filter/annotate variants (constant regions, too-many-subs, …)
    fitness.py        # Count filtering, fitness calculation, pseudocounts
    error_model.py    # Normalisation (L-BFGS-B) + error model (least_squares bootstrap)
    aggregate.py      # AA-level aggregation with inverse-variance weighting
    mutations.py      # Single/double mutation identification (vectorized)
    merge_fitness.py  # Inverse-variance merge, output file writing
  cli.py              # typer CLI entry point
  wrap/               # (M2) FASTQ processing: cutadapt, FastQC, VSEARCH, starcode
```

## Tests

```bash
pytest tests/ -v
```

52 tests: unit tests for sequence ops, error model, configuration; plus a full
end-to-end integration test that compares output against R DiMSum 1.4 on the
bundled Toy demo data (fitness agreement within ±0.05 absolute).

## Implementation notes

- **Counts stored as UInt32**, not R doubles — 2× lower memory per count column.
- **Single-pass pivot** replaces iterative pairwise `merge(all=T)` in `dimsum_stage_merge.R`.
- **SciPy `minimize` (L-BFGS-B)** replaces R's `nlm()` for replicate normalisation.
- **SciPy `least_squares` (TRF)** with 3 multi-starts replaces `nls()` with up to 20 retries per bootstrap.
- **`ProcessPoolExecutor`** for parallel bootstrap fitting (`--num_cores`).
- Output `.RData` replaced by **Parquet** (language-agnostic, faster to load).

## What's not yet implemented

- **WRAP stages 0–3** (demultiplex, FastQC, trim, align, tally): subprocess wrappers around cutadapt / VSEARCH / starcode. Coming in M2.
- **HTML report**: Jinja2 + matplotlib replacement for the Rmd report. Coming in M3.
- **Growth rate inference** and **barcode/trans-library** paths: M3.
- **Bayesian double-mutant fitness**: disabled (`--bayesian_double_fitness` flag not yet functional).
