# pyDiMSum

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/d24065d9-6d59-45ac-ba91-a1ee376e58ca" />

A Python reimplementation of the [DiMSum](https://github.com/lehner-lab/DiMSum) deep mutational scanning analysis pipeline (Faure et al., *Genome Biology* 2020).

## Contents

- [Why pyDiMSum?](#why-pydimsum)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage examples](#usage-examples)
  - [STEAM-only: start from a count table](#1-steam-only-start-from-a-count-table)
  - [Full pipeline: start from FASTQs](#2-full-pipeline-start-from-fastqs)
  - [Enrichment / library mode](#3-enrichment--library-mode)
  - [Growth-rate assay](#4-growth-rate-assay)
  - [Barcoded libraries](#5-barcoded-libraries)
  - [Trans-library (non-overlapping paired-end)](#6-trans-library-non-overlapping-paired-end)
  - [Coding vs non-coding sequences](#7-coding-vs-non-coding-sequences)
- [File formats](#file-formats)
  - [Experiment design file](#experiment-design-file)
  - [Variant count file](#variant-count-file)
  - [Barcode design file](#barcode-design-file)
  - [Variant identity file (barcode→variant map)](#variant-identity-file)
  - [Synonym sequences file](#synonym-sequences-file)
- [Complete flag reference](#complete-flag-reference)
  - [Input / output](#input--output)
  - [FASTQ input](#fastq-input)
  - [Adapter trimming (WRAP stage 2)](#adapter-trimming-wrap-stage-2)
  - [Alignment & quality (WRAP stage 3)](#alignment--quality-wrap-stage-3)
  - [Variant processing (STEAM stage 4)](#variant-processing-steam-stage-4)
  - [Fitness calculation (STEAM stage 5)](#fitness-calculation-steam-stage-5)
  - [Barcodes](#barcodes)
  - [Enrichment mode](#enrichment-mode)
  - [Misc](#misc)
- [Output files](#output-files)
- [Pipeline stages](#pipeline-stages)
- [Performance](#performance)
- [Architecture](#architecture)

---

## Why pyDiMSum?

DiMSum users report that the R version can be **slow and memory-hungry** on large datasets. The bottlenecks are well-understood:

| Problem | R location | Python fix |
|---------|------------|------------|
| Per-variant Hamming (strsplit per row) | `dimsum__hamming_distance.R` | NumPy `(n, L) uint8` matrix; O(n) column-wise diff |
| Constant-region loop (column rewrite per base) | `dimsum__remove_internal_constant_region.R` | Boolean mask slice of the matrix |
| Iterative pairwise `merge(all=T)` for counts | `dimsum_stage_merge.R` | Single Polars `pivot` |
| Repeated `data.table::copy()` | `process_merged_variants.R` | Zero-copy Polars boolean filters |
| 100 bootstraps × 20 NLS retries | `dimsum__fit_error_model_bootstrap.R` | `scipy.least_squares` with 3 multi-starts |

**Demo (40,591 variants, 4 replicates, single core):**

| | Time |
|--|------|
| R DiMSum 1.4 | ~46 s |
| pyDiMSum | ~6 s |

pyDiMSum also adds capabilities not yet in the R version:
- **Enrichment mode** for library-vs-library screens where a wildtype reference is not meaningful
- **HTML report** with embedded diagnostic plots (no R/Rmd required)
- **Parquet output** instead of `.RData` — loadable in Python, R, or any analytics tool

---

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.11, polars, numpy, scipy, biopython, pyarrow, typer.

**Optional — HTML report** (matplotlib plots + Jinja2 template):
```bash
pip install -e ".[report]"
```

**Optional — full WRAP pipeline** (stages 0–3) also requires these external binaries on `$PATH`:
- [cutadapt](https://cutadapt.readthedocs.io/) ≥ 3.5
- [VSEARCH](https://github.com/torognes/vsearch) ≥ 2.21
- [starcode](https://github.com/gui11aume/starcode) ≥ 1.4
- [FastQC](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/) ≥ 0.12 (optional — stage 1 only)

---

## Quick start

The fastest way to try pyDiMSum is with the Toy demo data bundled in `tests/data/`:

```bash
pydimsum \
  --experiment_design_path tests/data/experimentDesign_Toy.txt \
  --wildtype_sequence GGTAATAGCAGAGGGGGTGGAGCTGGTTTGGGAAACAATCAAGGTAGTAATATGGGTGGTGGGATGAACTTTGGTGCGTTCAGCATTAATCCAGCCATGATGGCTGCCGCCCAGGCAGCACTACAG \
  --count_path tests/data/countFile_Toy.txt \
  --output_path results/ \
  --project_name Toy
```

Output files will appear in `results/Toy/`. An HTML report is written at `results/Toy/Toy_report.html` if matplotlib and jinja2 are installed.

---

## Usage examples

### 1. STEAM-only: start from a count table

Use this when you already have per-variant read counts (e.g. from a previous DiMSum run, your own pipeline, or a collaborator). No external binaries required.

```bash
pydimsum \
  --experiment_design_path my_design.tsv \
  --wildtype_sequence ATGcccGCT \
  --count_path my_counts.tsv \
  --output_path results/ \
  --project_name MyProtein
```

In the count table, column names must match the `sample_name` values in your design file (see [Variant count file](#variant-count-file)).

**Useful options for this mode:**
```bash
  --fitness_min_input_count_all 10   # discard variants with < 10 input reads in all replicates
  --fitness_min_input_count_any 1    # or < 1 in any replicate
  --max_substitutions 2              # retain ≤ 2 amino acid (coding) or nt (noncoding) substitutions
  --retained_replicates 1,2,3        # use only these replicate numbers (default: all)
  --num_cores 4                      # parallelise bootstrap error model fitting
```

**Per-Hamming-distance count thresholds** are also supported. This lets you require higher counts for low-substitution variants (which are more informative) while keeping a lower threshold for highly-mutated ones:

```bash
  --fitness_min_input_count_all "1:100,2:10,3:5"
  # Hamming 1 variants: ≥100 reads; Hamming 2: ≥10; Hamming 3: ≥5
  # Any variant class not listed is discarded
```

---

### 2. Full pipeline: start from FASTQs

Use this when starting from raw sequencing reads. Stages 0–3 (WRAP) handle demultiplexing, QC, adapter trimming, paired-end merging, and variant tallying.

```bash
pydimsum \
  --experiment_design_path my_design.tsv \
  --wildtype_sequence ATGCGT... \
  --fastq_file_dir /data/fastq/ \
  --output_path results/ \
  --project_name MyProject \
  --cutadapt_5_first ACGTACGT \
  --cutadapt_5_second TTGGCCAA \
  --num_cores 8
```

The `--fastq_file_dir` flag sets the directory containing your FASTQ files. Alternatively, add a `pair_directory` column to your design file to specify per-row directories (useful when samples are split across folders).

**Common WRAP options:**
```bash
  --gzipped                    # FASTQs are gzipped (default: true; use --no_gzipped if not)
  --fastq_file_extension .fastq.gz  # file extension to look for (default: .fastq)
  --paired / --no_paired       # paired-end or single-end (default: paired)
  --vsearch_min_qual 30        # discard reads with any base < Q30 (default: 30)
  --vsearch_max_ee 0.5         # discard read pairs with expected errors > 0.5 (default: 0.5)
  --cutadapt_min_length 50     # discard reads shorter than this after trimming (default: 50)
  --start_stage 2              # resume from stage 2 (e.g. if stage 1 already ran)
  --stop_stage 3               # stop after stage 3 (count-table only, skip fitness)
  --retain_intermediate_files  # keep trimmed/merged FASTQs (large — off by default)
```

**Resuming a partial run:**
```bash
  --start_stage 4   # skip WRAP, start from variant processing
  --start_stage 5   # skip variant processing, start from fitness calculation
```

---

### 3. Enrichment / library mode

Use `--enrichment_mode` when your library does not have a single wildtype reference — for example, a screen comparing many different protein variants, a library of regulatory sequences, or a deep mutational scan of a disordered region where fitness is measured relative to the library median rather than a wildtype.

In this mode:
- `--wildtype_sequence` is **not required**
- Per-variant enrichment `log(output / input)` is calculated instead of WT-normalised fitness
- All mutation-centric filters (Hamming distance, constant regions, permitted sequences) are bypassed
- The WT-based error model is replaced with a Poisson count-based error estimate

```bash
pydimsum \
  --enrichment_mode \
  --experiment_design_path my_design.tsv \
  --count_path my_counts.tsv \
  --enrichment_normalise median \
  --output_path results/ \
  --project_name IDR_screen
```

**Normalisation strategies** (`--enrichment_normalise`):

| Strategy | Description |
|----------|-------------|
| `median` | Subtract the median enrichment score (centres the distribution at zero) |
| `total` | Normalise by total library read counts (size-factor normalisation) |
| `reference` | Subtract the enrichment of a specified reference sequence |
| `spikein` | Subtract the mean enrichment of a set of spike-in sequences |
| `none` | Raw log(out/in) with no normalisation |

```bash
# Median normalisation (default)
--enrichment_normalise median

# Reference sequence normalisation
--enrichment_normalise reference \
--enrichment_reference_id ATGCGTACGTAGCTACGT

# Spike-in normalisation
--enrichment_normalise spikein \
--enrichment_spikein_ids "ATGCGT,GCTAGC,TTAGGC"
```

---

### 4. Growth-rate assay

When fitness is measured in a growth competition assay (e.g. yeast doubling competition), you can provide cell density and selection time to infer growth rates directly, rather than using sequencing time points.

Add `cell_density`, `selection_time`, and (optionally) `generations` columns to your experiment design file:

```
sample_name  experiment_replicate  selection_id  ...  cell_density  selection_time
input1       1                     0             ...  0.05          
output1A     1                     1             ...  2.10          24.0
```

- `cell_density`: OD or equivalent at the time of sampling (all rows)
- `selection_time`: hours of selection (output rows only)
- `generations`: if provided instead of cell_density/selection_time, fitness is directly divided by generations rather than inferred

When all output rows have `cell_density` and `selection_time`, pyDiMSum automatically infers growth rates and uses them to normalise fitness. No extra flag is required — it detects the columns.

---

### 5. Barcoded libraries

If your library uses short barcodes that map to longer variant sequences, provide a barcode→variant mapping file:

```bash
pydimsum \
  --experiment_design_path my_design.tsv \
  --wildtype_sequence ATGCGT... \
  --count_path barcoded_counts.tsv \
  --barcode_identity_path barcode_map.tsv \
  --output_path results/ \
  --project_name Barcoded
```

The barcode identity file has two columns: `barcode` and `variant` (see [Variant identity file](#variant-identity-file)).

pyDiMSum replaces each barcode sequence in the count table with its mapped variant, then aggregates counts for barcodes that map to the same variant.

If your FASTQs contain a sample-level index that needs demultiplexing first, provide a `--barcode_design_path` (see [Barcode design file](#barcode-design-file)).

---

### 6. Trans-library (non-overlapping paired-end)

Use `--trans_library` when R1 and R2 read different molecules (rather than the two ends of the same fragment). Instead of overlap-merging, the two reads are simply concatenated.

```bash
pydimsum \
  --experiment_design_path my_design.tsv \
  --wildtype_sequence ATGCGT... \
  --fastq_file_dir /data/fastq/ \
  --trans_library \
  --output_path results/ \
  --project_name Trans
```

If R2 is on the reverse strand relative to R1, add `--trans_library_reverse_complement` to reverse-complement R2 before concatenation.

The same quality filters apply as in standard paired-end mode: reads are discarded if either read is shorter than `--cutadapt_min_length`, any base quality falls below `--vsearch_min_qual`, or combined expected errors exceed `--vsearch_max_ee`.

---

### 7. Coding vs non-coding sequences

**Auto-detection (default):** If the wildtype sequence length is divisible by 3 and the translation contains no premature stop codons, the sequence is treated as coding. Otherwise noncoding.

**Force coding:**
```bash
  --sequence_type coding
```

**Force noncoding:**
```bash
  --sequence_type noncoding
```

**Coding-sequence-specific options:**
```bash
  --mixed_substitutions       # allow variants with both synonymous and non-syn changes in the same codon
  --mutagenesis_type codon    # library was designed by codon-level mutagenesis (affects permitted mutations)
  --max_substitutions 2       # maximum amino acid substitutions (default: 2)
```

**Internal constant regions** in the wildtype sequence are specified using lower-case letters. These bases are treated as fixed (must match WT exactly) and are excised before Hamming distance calculations:

```
--wildtype_sequence ATGcgtACGtagTTG
                        ^^^   ^^^  — constant regions (lower-case)
                    ^^^   ^^^  ^^^  — variable positions (upper-case)
```

Variants where the constant-region bases differ from WT are rejected.

---

## File formats

### Experiment design file

Tab-separated plain text. One row per FASTQ file (or pair of FASTQ files). Download a template from `DiMSum/examples/example_experimentDesign.txt`.

**Required columns:**

| Column | Type | Description |
|--------|------|-------------|
| `sample_name` | string | Unique, alphanumeric name for this sample (e.g. `input1`, `output1A`). Used to match count-file column headers. |
| `experiment_replicate` | integer | Groups a matched input+output set. Variants from the same experiment replicate were grown from the same starting cell population. |
| `selection_id` | 0 or 1 | `0` = input (before selection), `1` = output (after selection). |
| `selection_replicate` | integer | (Output rows only) Distinguishes biological output replicates that share the same input. Leave blank for input rows. |
| `technical_replicate` | integer | (Optional) Marks sequencing lanes or files from the same extracted DNA. Counts are summed across technical replicates before analysis. |
| `pair1` | filename | (WRAP only) R1 FASTQ filename. |
| `pair2` | filename | (WRAP only) R2 FASTQ filename. Omit for single-end designs. |

**Optional columns for growth-rate assays:**

| Column | Rows | Description |
|--------|------|-------------|
| `generations` | output | Number of cell generations. If provided, fitness is divided by generations. |
| `cell_density` | all | OD or cell count at sampling time, used to infer growth rate. |
| `selection_time` | output | Hours of selection, used together with `cell_density`. |

**Per-sample cutadapt overrides:** Any cutadapt flag can be specified as a column name (e.g. `cutadapt5First`, `cutadaptMinLength`). Column values override the global CLI flag for that sample.

**Multiple input/output design example:**

```
sample_name  experiment_replicate  selection_id  selection_replicate  pair1              pair2
input1       1                     0                                  input_rep1_R1.fq   input_rep1_R2.fq
output1A     1                     1             1                    output1A_R1.fq     output1A_R2.fq
output1B     1                     1             2                    output1B_R1.fq     output1B_R2.fq
input2       2                     0                                  input_rep2_R1.fq   input_rep2_R2.fq
output2A     2                     1             1                    output2A_R1.fq     output2A_R2.fq
```

Here `experiment_replicate=1` and `experiment_replicate=2` are independent biological experiments (different starting populations). Within each experiment, `output1A` and `output1B` are technical or biological output replicates (both derived from `input1`). Counts are summed across selection replicates before fitness is calculated.

---

### Variant count file

Tab-separated plain text. One row per variant, one column per sample. The `sample_name` values in your experiment design file must match the column headers exactly.

```
nt_seq                   input1  output1A  input2  output2A
ATGCGTACG...             1523    892       1401    738
ATGCATACG...             87      231       94      219
```

- `nt_seq` — the full nucleotide sequence (A/C/G/T lowercase)
- One column per `sample_name` in the experiment design

Count values can be integers or floats. Missing values are treated as 0.

A template is available at `DiMSum/examples/example_variantCounts.txt`.

---

### Barcode design file

Used for multiplexed FASTQ files where sample indices need to be demultiplexed before further processing.

```
pair1               pair2               barcode    new_pair_prefix
multiplex_R1.fastq  multiplex_R2.fastq  ACGTACGT   sample1
multiplex_R1.fastq  multiplex_R2.fastq  TTGGCCAA   sample2
```

After demultiplexing, reads for `sample1` are written to `sample1_1.fastq` and `sample1_2.fastq`. These filenames (with `1.fastq`/`2.fastq` appended) must appear as `pair1`/`pair2` entries in the experiment design file.

---

### Variant identity file

Maps short DNA barcodes to their corresponding variant sequences. Used with `--barcode_identity_path`.

```
barcode       variant
AAAAAACCGT    ATGCGTACGTAGCTACGTCCTAGCGATCGATCG
TTAGGCAACT    ATGCGTACGTAGCTACGTCCAAACGATCGATCG
```

Both columns must contain only A/C/G/T characters. A template is at `DiMSum/examples/example_variantIdentity.txt`.

---

### Synonym sequences file

A list of coding nucleotide sequences (one per line, no header) that should be treated as synonymous references in addition to the wildtype. Used with `--synonym_sequence_path`.

```
ATGCGTACGTAGCTATCG
ATGCGCACGTAGCTGTCG
```

---

## Complete flag reference

### Input / output

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment_design_path` | *(required)* | Path to the tab-separated experiment design file. |
| `--wildtype_sequence` | *(required)* | WT nucleotide sequence. Upper-case = variable positions; lower-case = internal constant regions to be excised. Not required in `--enrichment_mode`. |
| `--output_path` | `.` | Directory for all output files. |
| `--project_name` | `DiMSum_Project` | Name of the project subdirectory and output file prefix. |
| `--count_path` | — | Path to a pre-computed variant count file. Providing this skips WRAP (stages 0–3). |
| `--start_stage` | `0` | Resume the pipeline from this stage (0–5). |
| `--stop_stage` | `5` | Stop after this stage. Useful for running only WRAP (`--stop_stage 3`) or inspecting variant tables before fitness (`--stop_stage 4`). |
| `--num_cores` | `1` | Number of CPU cores for parallel steps (bootstrap error model fitting, cutadapt). |
| `--retain_intermediate_files` | `false` | Keep trimmed/merged FASTQ files. These can be many GB but allow resuming from intermediate stages. |

---

### FASTQ input

| Flag | Default | Description |
|------|---------|-------------|
| `--fastq_file_dir` | — | Directory containing all FASTQ files. Overrides any `pair_directory` column in the design file. |
| `--fastq_file_extension` | `.fastq` | File extension for FASTQ files. Use `.fastq.gz` for gzipped files unless you also set `--gzipped`. |
| `--gzipped` / `--no_gzipped` | `true` | Whether FASTQ files are gzipped. |
| `--paired` / `--no_paired` | `true` | Paired-end (`true`) or single-end (`false`). |
| `--stranded` / `--no_stranded` | `true` | Whether the library is stranded. |
| `--experiment_design_pair_duplicates` | `false` | Allow the same FASTQ filename pair to appear more than once in the design file (e.g. technical replicates sequenced from the same file). |
| `--trans_library` / `--no_trans_library` | `false` | Concatenate R1+R2 instead of overlap-merging. Use when reads cover two distinct molecules that cannot overlap. |
| `--trans_library_reverse_complement` | `false` | Reverse-complement R2 before concatenation (trans-library mode only). |

---

### Adapter trimming (WRAP stage 2)

These options are passed to [cutadapt](https://cutadapt.readthedocs.io/). All can also be specified as columns in the experiment design file for per-sample control.

| Flag | Default | Description |
|------|---------|-------------|
| `--cutadapt_5_first` | — | 5′ adapter sequence for R1. Supports linked adapters: `FRONT;optional...BACK;required`. |
| `--cutadapt_5_second` | — | 5′ adapter sequence for R2. |
| `--cutadapt_3_first` | *(revcomp of `--cutadapt_5_second`)* | 3′ adapter sequence for R1. |
| `--cutadapt_3_second` | *(revcomp of `--cutadapt_5_first`)* | 3′ adapter sequence for R2. |
| `--cutadapt_cut_5_first` | — | Remove this many bases from the 5′ end of R1 *before* adapter trimming (hard clipping). |
| `--cutadapt_cut_5_second` | — | Remove this many bases from the 5′ end of R2 before trimming. |
| `--cutadapt_cut_3_first` | — | Remove this many bases from the 3′ end of R1 before trimming. |
| `--cutadapt_cut_3_second` | — | Remove this many bases from the 3′ end of R2 before trimming. |
| `--cutadapt_min_length` | `50` | Discard reads shorter than this (nt) after trimming. |
| `--cutadapt_error_rate` | `0.2` | Maximum error rate (fraction of mismatches) for adapter matching. |
| `--cutadapt_overlap` | `3` | Minimum overlap (nt) between a read and the adapter sequence. |

---

### Alignment & quality (WRAP stage 3)

| Flag | Default | Description |
|------|---------|-------------|
| `--vsearch_min_qual` | `30` | Discard reads where any base has Phred quality < this threshold. |
| `--vsearch_max_qual` | `41` | Maximum Phred score in input/output FASTQ (cannot exceed 41 without special Illumina pipelines). |
| `--vsearch_max_ee` | `0.5` | Maximum expected errors (sum of 10^(−Q/10)) across the read or read pair. |
| `--vsearch_min_ovlen` | `10` | Minimum overlap length for paired-end merging. |

---

### Variant processing (STEAM stage 4)

| Flag | Default | Description |
|------|---------|-------------|
| `--wildtype_sequence` | *(required)* | See [Input / output](#input--output). The case of each base defines variable (upper) vs constant (lower) positions. |
| `--sequence_type` | `auto` | `auto` detects coding from the WT translation; or force `coding` or `noncoding`. |
| `--mutagenesis_type` | `random` | `random` (nucleotide-level mutagenesis) or `codon` (codon-level, for codon-scanning libraries). |
| `--permitted_sequences` | all | IUPAC string covering the variable positions only. Length must equal the number of upper-case bases in `--wildtype_sequence`. Examples: `NNNN` (all substitutions), `RRRR` (purines only). |
| `--max_substitutions` | `2` | Maximum number of substitutions to retain. Amino acid substitutions are counted for coding sequences; nucleotide substitutions for noncoding. |
| `--mixed_substitutions` | `false` | If `true`, retain coding variants that have both synonymous and non-synonymous changes in the same codon (normally rejected). |
| `--indels` | `none` | Which indel variants to retain: `none`, `all`, or a comma-separated list of sequence lengths (e.g. `95,96,97,98` to retain only ±3 nt indels around a 96-nt gene). |
| `--reverse_complement` | `false` | Reverse-complement all `nt_seq` values before processing. Useful when the library was cloned on the reverse strand. |
| `--barcode_identity_path` | — | See [Barcoded libraries](#5-barcoded-libraries). |
| `--synonym_sequence_path` | — | See [Synonym sequences file](#synonym-sequences-file). |

---

### Fitness calculation (STEAM stage 5)

| Flag | Default | Description |
|------|---------|-------------|
| `--fitness_min_input_count_all` | `0` | Discard variants where the input count is below this threshold in **all** replicates. Accepts a single integer or per-Hamming thresholds like `1:100,2:10`. |
| `--fitness_min_input_count_any` | `0` | Discard variants where the input count is below this threshold in **any** replicate. |
| `--fitness_min_output_count_all` | `0` | Minimum output count in all replicates. |
| `--fitness_min_output_count_any` | `0` | Minimum output count in any replicate. |
| `--fitness_normalise` / `--no_fitness_normalise` | `true` | Fit a per-replicate scale + shift normalisation to minimise inter-replicate differences. Requires ≥ 2 replicates. |
| `--fitness_error_model` / `--no_fitness_error_model` | `true` | Fit the error model (additive + multiplicative noise terms). Requires ≥ 2 replicates. If disabled, sigma is estimated from Poisson count noise only. |
| `--fitness_dropout_pseudocount` | `0` | Pseudocount added to output counts of `0` (to handle variants that drop out completely). A value of `0.5` is often used. |
| `--retained_replicates` | `all` | Comma-separated list of experiment replicate integers to include, or `all`. Useful to exclude a outlier replicate: `--retained_replicates 1,2,4`. |

---

### Barcodes

| Flag | Default | Description |
|------|---------|-------------|
| `--barcode_design_path` | — | Path to the [barcode design file](#barcode-design-file) for demultiplexing multiplexed FASTQs (WRAP stage 0). |
| `--barcode_error_rate` | `0.25` | Maximum error rate for matching sample barcodes during demultiplexing. |
| `--barcode_identity_path` | — | Path to the [variant identity file](#variant-identity-file) mapping sequenced barcodes to variant sequences (STEAM stage 4). |

---

### Enrichment mode

| Flag | Default | Description |
|------|---------|-------------|
| `--enrichment_mode` / `--no_enrichment_mode` | `false` | Enable enrichment mode. Bypasses all mutation-centric filters; computes per-sequence `log(out/in)` rather than WT-normalised fitness. |
| `--enrichment_normalise` | `median` | Normalisation strategy: `none`, `median`, `total`, `reference`, or `spikein`. See [Enrichment / library mode](#3-enrichment--library-mode). |
| `--enrichment_reference_id` | — | `nt_seq` of the reference sequence (required when `--enrichment_normalise reference`). |
| `--enrichment_spikein_ids` | — | Comma-separated `nt_seq` values for spike-in sequences (used when `--enrichment_normalise spikein`). |

---

### Misc

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` / `-v` | `false` | Print DEBUG-level log messages. |
| `--version` | — | Print version and exit. |

---

## Output files

All files are written to `{output_path}/{project_name}/`.

**Primary outputs (always produced in mutation mode):**

| File | Description |
|------|-------------|
| `fitness_singles.txt` | Single-mutant fitness and sigma values (one row per substitution). |
| `fitness_doubles.txt` | Double-mutant fitness (uncorrected; per-single references included). |
| `fitness_wildtype.txt` | WT variant fitness values across replicates. |
| `fitness_synonymous.txt` | Synonymous variant fitness scores (coding sequences only). |
| `fitness_singles_MaveDB.csv` | [MaveDB](https://www.mavedb.org/)-compatible CSV for data deposition. |
| `*_fitness_replicates/` | Parquet bundle: `all_variants.parquet`, `singles.parquet`, `doubles.parquet`, `synonymous.parquet`, `wildtype.parquet`. Loadable in Python (`polars.read_parquet`) or R (`arrow::read_parquet`). |
| `*_report.html` | Self-contained HTML report with diagnostic plots (requires matplotlib + jinja2). |

**Enrichment mode outputs:**

| File | Description |
|------|-------------|
| `enrichment_variant_data.txt` | Per-sequence enrichment scores and errors. |
| `*_enrichment.parquet/` | Parquet bundle with the enrichment table. |

**Intermediate files (always written):**

| File | Description |
|------|-------------|
| `*_variant_data_merge.tsv` | All retained variants with counts and annotations. |
| `*_indel_variant_data_merge.tsv` | Indel variants (retained or discarded, depending on `--indels`). |
| `*_rejected_variant_data_merge.tsv` | Variants rejected by constant-region, permitted-sequence, or substitution-count filters. |
| `tmp/errormodel.txt` | Fitted error model parameters. |
| `tmp/normalisationmodel.txt` | Fitted replicate normalisation parameters. |

---

## Pipeline stages

| Stage | Name | Description |
|-------|------|-------------|
| 0 | Demultiplex | Split multiplexed FASTQ files by barcode using cutadapt. Skipped if no `--barcode_design_path` is provided. |
| 1 | FastQC | Run FastQC on raw reads to generate quality reports. |
| 2 | Trim | Remove 5′/3′ constant adapter sequences with cutadapt. Also performs hard-clipping (`--cutadapt_cut_*`) if specified. |
| 3 | Align + Tally | Merge overlapping paired-end reads with VSEARCH; filter by quality (Q, EE, length); count unique variants with starcode. |
| 4 | Process variants | Build the combined count table; annotate variants (Hamming distances, AA translation, STOP codons); apply filters (constant regions, permitted mutations, max substitutions, mixed codon). |
| 5 | Fitness | Filter low-count variants; normalise replicates; fit error model; calculate per-variant fitness and sigma; merge replicates by inverse-variance weighting. |

Use `--start_stage` and `--stop_stage` to run a subset of stages. Common patterns:

```bash
--stop_stage 3         # run WRAP only (produce count tables, skip fitness)
--start_stage 4        # re-run STEAM only (e.g. with different fitness thresholds)
--start_stage 2 --stop_stage 2   # just re-trim with different adapter sequences
```

---

## Performance

pyDiMSum scales to large DMS datasets while keeping memory low:

- Counts are stored as `UInt32` (not R doubles) — 2× lower memory per count column.
- A **single-pass Polars pivot** replaces iterative pairwise `merge(all=T)` for building the count table.
- **NumPy matrix operations** (vectorized Hamming, permitted-mask, constant-region filter) replace row-by-row string operations.
- **SciPy L-BFGS-B** replaces R's `nlm()` for replicate normalisation.
- **SciPy `least_squares` (TRF)** with 3 multi-starts replaces `nls()` with up to 20 restarts for the error model bootstrap.
- **`ProcessPoolExecutor`** parallelises bootstrap iterations across `--num_cores` cores.

---

## Architecture

```
pydimsum/
  config.py               # RunConfig dataclass (~60 options) + validation
  pipeline.py             # Stage orchestrator: WRAP dispatch, STEAM dispatch
  cli.py                  # typer entry point — all CLI flags defined here
  io/
    designs.py            # ExperimentDesign: read, validate, apply fastq_file_dir
    counts.py             # Variant count file reader
  steam/
    merge.py              # Build wide count table (single-pass Polars pivot)
    sequences.py          # Vectorized NumPy: Hamming, translate, STOP, permitted mask
    process_variants.py   # Filter/annotate variants + WT guard
    fitness.py            # Count filtering, dropout pseudocount, fitness calculation
    error_model.py        # Replicate normalisation + bootstrap error model
    aggregate.py          # AA aggregation with inverse-variance weighting
    mutations.py          # Single/double mutation identification
    merge_fitness.py      # IVW merge across replicates + output file writing
    library.py            # Enrichment mode: process + enrichment scores + output
    growth_rates.py       # Growth rate inference from cell density data
  wrap/
    demultiplex.py        # Stage 0: cutadapt demultiplex
    fastqc.py             # Stage 1: FastQC
    trim.py               # Stage 2: cutadapt adapter trimming
    align.py              # Stage 3a: VSEARCH merge/filter (+ trans-library concat)
    tally.py              # Stage 3b: starcode unique-variant counting
  report/
    plots.py              # matplotlib plot generators (8 plot types)
    html.py               # Jinja2 HTML report renderer + WRAP stats parsers
    templates/
      report.html.j2      # Report template (self-contained, inline CSS)
```

## Tests

```bash
pytest tests/ -v
```

166 tests: unit tests for sequence operations, error model, configuration, barcodes, growth rates, report generation, WRAP read processing, and trans-library; plus a full end-to-end integration test comparing output against R DiMSum 1.4 on the bundled Toy demo (fitness within ±0.05 absolute).
