"""Matplotlib plot generators for the pyDiMSum HTML report.

All functions return a ``bytes`` object (PNG encoded) suitable for embedding
as a base64 data-URI in the HTML template.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

_MUTATION_COLOURS = {
    "0 substitutions":   "#2166ac",
    "1 substitution":    "#4dac26",
    "2 substitutions":   "#d01c8b",
    "3+ substitutions":  "#f1b6da",
    "indel (retained)":  "#b2182b",
    "mixed codon":       "#e08214",
    "too many subs":     "#fdb863",
    "not permitted":     "#fee090",
    "constant region":   "#abd9e9",
    "indel (discarded)": "#74add1",
    "invalid barcode":   "#d9d9d9",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _short_name(col: str) -> str:
    """Convert 'count_e1_s0' → 'e1_in', 'count_e2_s1' → 'e2_out'."""
    col = col.replace("count_", "")
    return col.replace("_s0", "_in").replace("_s1", "_out")

# ---------------------------------------------------------------------------
# Plot 1 & 2: Variant processing stacked bar charts
# ---------------------------------------------------------------------------

def plot_variant_processing(stats: dict, count_cols: list[str]) -> bytes:
    """Stacked bar chart: reads by variant category per sample (counts)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Build per-sample category values
    categories = [
        ("0 substitutions",   "nuc_subst_dict", 0),
        ("1 substitution",    "nuc_subst_dict", 1),
        ("2 substitutions",   "nuc_subst_dict", 2),
        ("3+ substitutions",  "nuc_subst_dict", None),  # computed as remainder
        ("indel (retained)",  "nuc_indel_dict", None),
        ("mixed codon",       "nuc_mxsub_dict", None),
        ("too many subs",     "nuc_tmsub_dict", None),
        ("not permitted",     "nuc_frbdn_dict", None),
        ("constant region",   "nuc_const_dict", None),
        ("invalid barcode",   "nuc_nbarc_dict", None),
    ]

    labels = [_short_name(c) for c in count_cols]
    n = len(count_cols)

    data: dict[str, list[float]] = {cat: [0.0] * n for cat, *_ in categories}

    nuc_subst = stats.get("nuc_subst_dict", {})
    for i, col in enumerate(count_cols):
        sample_subst = nuc_subst.get(col, {})
        total_subst = sum(sample_subst.values())
        s0 = float(sample_subst.get(0, 0))
        s1 = float(sample_subst.get(1, 0))
        s2 = float(sample_subst.get(2, 0))
        s3plus = max(0.0, total_subst - s0 - s1 - s2)

        data["0 substitutions"][i] = s0
        data["1 substitution"][i] = s1
        data["2 substitutions"][i] = s2
        data["3+ substitutions"][i] = s3plus

        for cat, stat_key, _ in categories[4:]:
            data[cat][i] = float(stats.get(stat_key, {}).get(col, 0))

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9 + 2), 5))
    x = np.arange(n)
    bottom = np.zeros(n)
    patches = []
    for cat, *_ in categories:
        vals = np.array(data[cat])
        colour = _MUTATION_COLOURS.get(cat, "#cccccc")
        bar = ax.bar(x, vals, bottom=bottom, color=colour, label=cat, width=0.7)
        bottom += vals
        patches.append(mpatches.Patch(color=colour, label=cat))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Reads", fontsize=10)
    ax.set_xlabel("Sample", fontsize=10)
    ax.set_title("Variant processing — read counts by category", fontsize=11)
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def plot_variant_processing_pct(stats: dict, count_cols: list[str]) -> bytes:
    """Stacked bar chart: reads by variant category per sample (percentages)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    categories = [
        "0 substitutions", "1 substitution", "2 substitutions", "3+ substitutions",
        "indel (retained)", "mixed codon", "too many subs", "not permitted",
        "constant region", "invalid barcode",
    ]
    stat_keys = [
        ("nuc_subst_dict", 0), ("nuc_subst_dict", 1), ("nuc_subst_dict", 2),
        ("nuc_subst_dict", None),
        ("nuc_indel_dict", None), ("nuc_mxsub_dict", None), ("nuc_tmsub_dict", None),
        ("nuc_frbdn_dict", None), ("nuc_const_dict", None), ("nuc_nbarc_dict", None),
    ]

    labels = [_short_name(c) for c in count_cols]
    n = len(count_cols)
    nuc_subst = stats.get("nuc_subst_dict", {})

    raw: dict[str, list[float]] = {cat: [0.0] * n for cat in categories}
    for i, col in enumerate(count_cols):
        s = nuc_subst.get(col, {})
        total = sum(s.values())
        raw["0 substitutions"][i] = float(s.get(0, 0))
        raw["1 substitution"][i] = float(s.get(1, 0))
        raw["2 substitutions"][i] = float(s.get(2, 0))
        raw["3+ substitutions"][i] = max(0.0, total - sum(raw[c][i] for c in ["0 substitutions", "1 substitution", "2 substitutions"]))
        raw["indel (retained)"][i] = float(stats.get("nuc_indel_dict", {}).get(col, 0))
        raw["mixed codon"][i] = float(stats.get("nuc_mxsub_dict", {}).get(col, 0))
        raw["too many subs"][i] = float(stats.get("nuc_tmsub_dict", {}).get(col, 0))
        raw["not permitted"][i] = float(stats.get("nuc_frbdn_dict", {}).get(col, 0))
        raw["constant region"][i] = float(stats.get("nuc_const_dict", {}).get(col, 0))
        raw["invalid barcode"][i] = float(stats.get("nuc_nbarc_dict", {}).get(col, 0))

    totals = np.array([sum(raw[c][i] for c in categories) for i in range(n)])
    totals = np.where(totals == 0, 1, totals)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9 + 2), 5))
    x = np.arange(n)
    bottom = np.zeros(n)
    patches = []
    for cat in categories:
        vals = np.array(raw[cat]) / totals * 100
        colour = _MUTATION_COLOURS.get(cat, "#cccccc")
        ax.bar(x, vals, bottom=bottom, color=colour, label=cat, width=0.7)
        bottom += vals
        patches.append(__import__("matplotlib.patches", fromlist=["Patch"]).Patch(color=colour, label=cat))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of reads", fontsize=10)
    ax.set_xlabel("Sample", fontsize=10)
    ax.set_title("Variant processing — read percentages by category", fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png

# ---------------------------------------------------------------------------
# Plot 3: Input count distributions per Nham class
# ---------------------------------------------------------------------------

def plot_count_distributions(retained_df: "pl.DataFrame", count_cols: list[str]) -> bytes:
    """Histogram of input counts split by Nham_nt class (0, 1, 2)."""
    import matplotlib.pyplot as plt
    import polars as pl

    in_cols = [c for c in count_cols if "_s0" in c]
    if not in_cols or "Nham_nt" not in retained_df.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.tight_layout()
        png = _fig_to_png(fig)
        plt.close(fig)
        return png

    col = in_cols[0]
    nham_max = min(int(retained_df["Nham_nt"].fill_null(0).max()), 3)
    colours = ["#2166ac", "#4dac26", "#d01c8b", "#f1b6da"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for nham in range(nham_max + 1):
        sub = retained_df.filter(pl.col("Nham_nt") == nham)
        if len(sub) == 0:
            continue
        vals = sub[col].fill_null(0).cast(pl.Float64).to_numpy()
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        log_vals = np.log10(vals + 1)
        ax.hist(
            log_vals, bins=40, alpha=0.6,
            color=colours[min(nham, 3)],
            label=f"Nham={nham} (n={len(sub):,})",
            density=True,
        )

    ax.set_xlabel("log₁₀(input count + 1)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Input count distributions by Nham ({_short_name(col)})", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png

# ---------------------------------------------------------------------------
# Plot 4: Replicate fitness scatter
# ---------------------------------------------------------------------------

def plot_replicate_fitness_scatter(fitness_df: "pl.DataFrame", replicates: list[int]) -> bytes:
    """Scatter plot of per-replicate fitness values for each pair of replicates."""
    import matplotlib.pyplot as plt
    import polars as pl

    fit_cols = [f"fitness{E}_uncorr" for E in replicates if f"fitness{E}_uncorr" in fitness_df.columns]
    n = len(fit_cols)
    if n < 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Need ≥2 replicates", ha="center", va="center")
        plt.tight_layout()
        png = _fig_to_png(fig)
        plt.close(fig)
        return png

    pairs = [(fit_cols[i], fit_cols[j]) for i in range(n) for j in range(i + 1, n)]
    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, (cx, cy) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        sub = fitness_df.select([cx, cy]).drop_nulls()
        x = sub[cx].to_numpy()
        y = sub[cy].to_numpy()
        ax.hexbin(x, y, gridsize=50, cmap="YlOrRd", mincnt=1, bins="log")
        # Correlation
        if len(x) > 2:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.set_title(f"r = {r:.3f}", fontsize=9)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel(cx.replace("_uncorr", ""), fontsize=8)
        ax.set_ylabel(cy.replace("_uncorr", ""), fontsize=8)

    # Hide unused subplots
    for idx in range(len(pairs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle("Replicate fitness correlations", fontsize=11, y=1.01)
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png

# ---------------------------------------------------------------------------
# Plot 5: Error model QQ plot (leave-one-out z-scores vs N(0,1))
# ---------------------------------------------------------------------------

def plot_error_model_qqplot(fitness_df: "pl.DataFrame", replicates: list[int]) -> bytes:
    """Leave-one-out QQ plot of per-replicate z-scores against N(0,1).

    For each replicate E, z = (fitness_E_uncorr − merged_fitness) / sigma_E_uncorr,
    where merged_fitness is the IVW mean of all *other* replicates.  Only rows
    where error_model == True are included.

    Mirrors R/dimsum__error_model_qqplot.R
    """
    import matplotlib.pyplot as plt
    import polars as pl
    from scipy import stats as scipy_stats

    fit_cols = [f"fitness{E}_uncorr" for E in replicates if f"fitness{E}_uncorr" in fitness_df.columns]
    sig_cols = [f"sigma{E}_uncorr" for E in replicates if f"sigma{E}_uncorr" in fitness_df.columns]

    if len(fit_cols) < 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Need ≥2 replicates for QQ plot", ha="center", va="center")
        plt.tight_layout()
        return _fig_to_png(fig)

    # Filter to error-model rows
    if "error_model" in fitness_df.columns:
        sub = fitness_df.filter(pl.col("error_model").fill_null(False))
    else:
        sub = fitness_df

    all_z: list[float] = []
    valid_reps = [r for r in replicates if f"fitness{r}_uncorr" in fitness_df.columns and f"sigma{r}_uncorr" in fitness_df.columns]

    for E in valid_reps:
        others = [r for r in valid_reps if r != E]
        if not others:
            continue
        # IVW merge of other replicates as "true" fitness reference
        w_sum = None
        fw_sum = None
        for O in others:
            sc = f"sigma{O}_uncorr"
            fc = f"fitness{O}_uncorr"
            w = pl.when(pl.col(sc).is_not_null() & (pl.col(sc) > 0)).then(
                1.0 / (pl.col(sc) ** 2)
            ).otherwise(None)
            fw = pl.when(pl.col(sc).is_not_null() & (pl.col(sc) > 0)).then(
                pl.col(fc) / (pl.col(sc) ** 2)
            ).otherwise(None)
            w_sum = w if w_sum is None else (w_sum + w.fill_null(0))
            fw_sum = fw if fw_sum is None else (fw_sum + fw.fill_null(0))

        ref_fit = pl.when(w_sum > 0).then(fw_sum / w_sum).otherwise(None)
        fc_e = f"fitness{E}_uncorr"
        sc_e = f"sigma{E}_uncorr"

        z_col = (
            pl.when(
                pl.col(fc_e).is_not_null() & pl.col(sc_e).is_not_null() & (pl.col(sc_e) > 0)
            ).then(
                (pl.col(fc_e) - ref_fit) / pl.col(sc_e)
            ).otherwise(None)
        )
        z_vals = sub.select(z_col.alias("z"))["z"].drop_nulls().to_numpy()
        # Clip extreme z-scores to avoid distortion from outliers
        z_vals = z_vals[np.isfinite(z_vals)]
        if len(z_vals) > 0:
            all_z.extend(z_vals.tolist())

    if not all_z:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No data for QQ plot", ha="center", va="center")
        plt.tight_layout()
        return _fig_to_png(fig)

    all_z_arr = np.array(all_z)
    # Clip at ±10 for display
    all_z_arr = np.clip(all_z_arr, -10, 10)
    all_z_arr.sort()

    n = len(all_z_arr)
    # Theoretical quantiles
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = scipy_stats.norm.ppf(probs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(theoretical, all_z_arr, s=2, alpha=0.4, color="#2166ac", rasterized=True)
    lim = max(abs(theoretical[[0, -1]]).max(), abs(all_z_arr[[0, -1]]).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1.2, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Theoretical N(0,1) quantiles", fontsize=10)
    ax.set_ylabel("Observed z-scores", fontsize=10)
    ax.set_title("Error model QQ plot (leave-one-out z-scores)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Plot 6 & 7: Per-sample WRAP statistics (VSEARCH + cutadapt)
# ---------------------------------------------------------------------------

def plot_vsearch_stats(sample_stats: list[dict]) -> bytes:
    """Stacked bar chart of VSEARCH merge/filter statistics per sample.

    Parameters
    ----------
    sample_stats:
        List of dicts with keys: sample_name, Pairs, Merged, Too_short,
        No_alignment_found, Too_many_diffs, Overlap_too_short,
        Exp.errs._too_high, Min_Q_too_low.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not sample_stats:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No VSEARCH stats", ha="center", va="center")
        plt.tight_layout()
        return _fig_to_png(fig)

    categories = [
        ("Merged",              "#2166ac"),
        ("Too short",           "#fdae61"),
        ("No alignment",        "#d7191c"),
        ("Too many diffs",      "#f46d43"),
        ("Overlap too short",   "#abd9e9"),
        ("Exp. err. too high",  "#fee090"),
        ("Min Q too low",       "#d9d9d9"),
    ]
    stat_keys = [
        "Merged", "Too_short", "No_alignment_found",
        "Too_many_diffs", "Overlap_too_short", "Exp.errs._too_high", "Min_Q_too_low",
    ]

    n = len(sample_stats)
    labels = [s.get("sample_name", str(i)) for i, s in enumerate(sample_stats)]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8 + 2), 5))
    bottom = np.zeros(n)
    patches = []
    for (label, colour), key in zip(categories, stat_keys):
        vals = np.array([float(s.get(key, 0)) for s in sample_stats])
        ax.bar(x, vals, bottom=bottom, color=colour, label=label, width=0.7)
        bottom += vals
        patches.append(mpatches.Patch(color=colour, label=label))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Reads", fontsize=10)
    ax.set_xlabel("Sample", fontsize=10)
    ax.set_title("VSEARCH alignment statistics per sample", fontsize=11)
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


def plot_cutadapt_stats(sample_stats: list[dict]) -> bytes:
    """Stacked bar chart of cutadapt trimming statistics per sample.

    Parameters
    ----------
    sample_stats:
        List of dicts with keys: sample_name, total_reads,
        read1_with_adapter (optional), read2_with_adapter (optional),
        pairs_written (optional).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not sample_stats:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No cutadapt stats", ha="center", va="center")
        plt.tight_layout()
        return _fig_to_png(fig)

    n = len(sample_stats)
    labels = [s.get("sample_name", str(i)) for i, s in enumerate(sample_stats)]
    x = np.arange(n)

    totals = np.array([float(s.get("total_reads", 0)) for s in sample_stats])
    passed = np.array([float(s.get("pairs_written", s.get("total_reads", 0))) for s in sample_stats])
    filtered = np.maximum(0, totals - passed)

    colours = ["#2166ac", "#d7191c"]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8 + 2), 5))
    ax.bar(x, passed, color=colours[0], label="Passed filters", width=0.7)
    ax.bar(x, filtered, bottom=passed, color=colours[1], label="Filtered/discarded", width=0.7)
    patches = [
        mpatches.Patch(color=colours[0], label="Passed filters"),
        mpatches.Patch(color=colours[1], label="Filtered/discarded"),
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Reads", fontsize=10)
    ax.set_xlabel("Sample", fontsize=10)
    ax.set_title("Cutadapt trimming statistics per sample", fontsize=11)
    ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.8)
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Plot 8: Input count scatter matrix (diagnostics)
# ---------------------------------------------------------------------------

def plot_count_scatter_matrix(variant_df: "pl.DataFrame", count_cols: list[str]) -> bytes:
    """All-vs-all scatter matrix of log10(count+1) for input and output samples."""
    import matplotlib.pyplot as plt
    import polars as pl

    cols = [c for c in count_cols if c in variant_df.columns]
    if len(cols) < 2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Need ≥2 samples", ha="center", va="center")
        plt.tight_layout()
        png = _fig_to_png(fig)
        plt.close(fig)
        return png

    n = len(cols)
    labels = [_short_name(c) for c in cols]
    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))
    if n == 1:
        axes = np.array([[axes]])

    log_data = {}
    for c in cols:
        arr = variant_df[c].fill_null(0).cast(pl.Float64).to_numpy()
        log_data[c] = np.log10(arr + 1)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            if i == j:
                ax.hist(log_data[cols[i]], bins=40, color="#4dac26", alpha=0.7)
                ax.set_xlabel(labels[i], fontsize=7)
            elif i > j:
                x = log_data[cols[j]]
                y = log_data[cols[i]]
                ax.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1, bins="log")
                if len(x) > 2:
                    r = float(np.corrcoef(x, y)[0, 1])
                    ax.set_title(f"r={r:.2f}", fontsize=7, pad=2)
            else:
                ax.set_visible(False)

    for i in range(n):
        axes[i][0].set_ylabel(labels[i], fontsize=7)

    plt.suptitle("Sample count correlations (log₁₀(count+1))", fontsize=10, y=1.01)
    plt.tight_layout()
    png = _fig_to_png(fig)
    plt.close(fig)
    return png
