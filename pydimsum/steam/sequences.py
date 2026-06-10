"""Vectorized sequence operations using NumPy.

All functions operate on arrays of sequences, never one sequence at a time,
replacing the per-row mapply / strsplit loops in the R code.

Key design:
  - Encode a list of nt/aa strings into a 2D ``(n_seqs, seq_len)`` uint8 array
    once, then all comparisons are column-wise NumPy operations — O(n·L) but
    with no Python-level loop overhead.

Replaces:
  R/dimsum__hamming_distance.R (per-row mapply)
  R/dimsum__remove_internal_constant_region.R (per-base column loop)
  R/dimsum__identify_STOP_mutations.R
  R/dimsum__identify_permitted_mutations.R (position loop)
  Nmut_codons computation in dimsum__process_merged_variants.R:276
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

import numpy as np
from Bio.Data import CodonTable

# ---------------------------------------------------------------------------
# IUPAC expansion table
# ---------------------------------------------------------------------------
_IUPAC_EXPAND: dict[str, set[int]] = {
    "A": {ord("a")},
    "C": {ord("c")},
    "G": {ord("g")},
    "T": {ord("t")},
    "R": {ord("a"), ord("g")},
    "Y": {ord("c"), ord("t")},
    "S": {ord("c"), ord("g")},
    "W": {ord("a"), ord("t")},
    "K": {ord("g"), ord("t")},
    "M": {ord("a"), ord("c")},
    "B": {ord("c"), ord("g"), ord("t")},
    "D": {ord("a"), ord("g"), ord("t")},
    "H": {ord("a"), ord("c"), ord("t")},
    "V": {ord("a"), ord("c"), ord("g")},
    "N": {ord("a"), ord("c"), ord("g"), ord("t")},
}

# Standard codon table from BioPython
_CODON_TABLE = CodonTable.unambiguous_dna_by_name["Standard"]
_CODON_TABLE_FORWARD: dict[str, str] = dict(_CODON_TABLE.forward_table)
_CODON_TABLE_FORWARD.update({s: "*" for s in _CODON_TABLE.stop_codons})


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode(sequences: Sequence[str]) -> np.ndarray:
    """Encode a list of same-length strings into a (n, L) uint8 array.

    Each character is stored as its ASCII ordinal.  Works for both nt and aa
    sequences.

    Parameters
    ----------
    sequences:
        Iterable of strings, all the same length.

    Returns
    -------
    np.ndarray of shape (n, L) and dtype uint8.
    """
    if not sequences:
        return np.empty((0, 0), dtype=np.uint8)
    seqs = list(sequences)
    L = len(seqs[0])
    arr = np.frombuffer(("".join(seqs)).encode("ascii"), dtype=np.uint8).reshape(
        len(seqs), L
    )
    return arr.copy()  # copy because frombuffer returns read-only view


def decode_row(row: np.ndarray) -> str:
    """Decode a 1D uint8 row back to a string."""
    return row.tobytes().decode("ascii")


# ---------------------------------------------------------------------------
# Hamming distance (vectorized)
# ---------------------------------------------------------------------------

def hamming(mat: np.ndarray, ref: np.ndarray | str | bytes) -> np.ndarray:
    """Compute Hamming distance from each row of *mat* to *ref*.

    Parameters
    ----------
    mat:
        (n, L) uint8 array of sequences.
    ref:
        Single sequence as a string, bytes, or 1D uint8 array of length L.

    Returns
    -------
    np.ndarray of shape (n,) with dtype int32.

    Replaces: dimsum__hamming_distance.R — no more per-row strsplit.
    """
    if isinstance(ref, (str, bytes)):
        ref_arr = np.frombuffer(
            ref.encode("ascii") if isinstance(ref, str) else ref, dtype=np.uint8
        )
    else:
        ref_arr = np.asarray(ref, dtype=np.uint8)
    return (mat != ref_arr).sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Internal constant region removal
# ---------------------------------------------------------------------------

def variable_position_mask(wt_coded: str) -> np.ndarray:
    """Return a boolean mask of variable (upper-case) positions in *wt_coded*.

    Parameters
    ----------
    wt_coded:
        The case-coded WT sequence: upper-case = variable, lower-case = constant.

    Returns
    -------
    np.ndarray of shape (L,) dtype bool — True where variable.

    Example
    -------
    >>> variable_position_mask("aGTa")
    array([False,  True,  True, False])
    """
    arr = np.frombuffer(wt_coded.encode("ascii"), dtype=np.uint8)
    # Upper-case ASCII range: 65-90
    return (arr >= 65) & (arr <= 90)


def strip_constant_regions(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove constant-region columns from a (n, L) sequence matrix.

    Parameters
    ----------
    mat:
        (n, L) uint8 sequence matrix.
    mask:
        Boolean array of length L, True at variable positions to keep.

    Returns
    -------
    (n, n_variable) uint8 array.

    Replaces: dimsum__remove_internal_constant_region.R per-base loop.
    """
    return mat[:, mask]


def constant_region_matches_wt(
    mat: np.ndarray,
    wt_coded: str,
) -> np.ndarray:
    """Return a boolean array indicating which sequences have WT constant regions.

    Parameters
    ----------
    mat:
        (n, L) uint8 matrix of full-length (non-indel) sequences.
    wt_coded:
        Case-coded WT sequence (upper = variable, lower = constant).

    Returns
    -------
    np.ndarray of shape (n,) dtype bool.
    """
    wt_arr = np.frombuffer(wt_coded.lower().encode("ascii"), dtype=np.uint8)
    mask_const = ~variable_position_mask(wt_coded)
    if not mask_const.any():
        # No constant region at all — all sequences pass
        return np.ones(len(mat), dtype=bool)
    const_mat = mat[:, mask_const]
    wt_const = wt_arr[mask_const]
    return (const_mat == wt_const).all(axis=1)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _translate_codon(codon: str) -> str:
    """Translate a single codon (lower-case nt) to single-letter AA."""
    return _CODON_TABLE_FORWARD.get(codon, "X")  # X = unknown/ambiguous


def translate_sequences(nt_mat: np.ndarray) -> np.ndarray:
    """Translate a (n, L) lower-cased nt matrix to a (n, L//3) AA matrix.

    Parameters
    ----------
    nt_mat:
        (n, L) uint8 matrix where L must be divisible by 3.
        Characters should be lower-case a/c/g/t.

    Returns
    -------
    (n, L//3) uint8 matrix of single-letter amino acid ASCII ordinals.
    """
    n, L = nt_mat.shape
    if L % 3 != 0:
        raise ValueError(f"Sequence length {L} is not divisible by 3")
    n_codons = L // 3
    codon_mat = nt_mat.reshape(n, n_codons, 3)
    aa_arr = np.empty((n, n_codons), dtype=np.uint8)
    for j in range(n_codons):
        for i in range(n):
            codon = decode_row(codon_mat[i, j])
            aa_arr[i, j] = ord(_translate_codon(codon))
    return aa_arr


def translate_sequences_fast(nt_seqs: list[str]) -> list[str]:
    """Translate a list of nt strings to aa strings using BioPython.

    This is used when Biopython's vectorized path is faster than the pure
    NumPy loop (e.g. for very short sequences).
    """
    from Bio.Seq import Seq

    results = []
    for seq in nt_seqs:
        try:
            aa = str(Seq(seq).translate())
        except Exception:
            aa = "X" * (len(seq) // 3)
        results.append(aa)
    return results


# ---------------------------------------------------------------------------
# STOP codon detection
# ---------------------------------------------------------------------------

def detect_stop(aa_mat: np.ndarray, wt_has_terminal_stop: bool) -> tuple[np.ndarray, np.ndarray]:
    """Detect premature STOP codons and readthrough mutations.

    Parameters
    ----------
    aa_mat:
        (n, n_aa) uint8 matrix of amino acid ASCII ordinals.
    wt_has_terminal_stop:
        Whether the WT amino acid sequence ends with '*'.

    Returns
    -------
    has_premature_stop : np.ndarray shape (n,) bool
        True if the sequence contains a '*' before the last position
        (regardless of WT terminal stop).
    has_readthrough : np.ndarray shape (n,) bool
        True if WT has a terminal stop but this sequence does not end with '*'.
        All False if wt_has_terminal_stop is False.
    """
    stop_ord = ord("*")
    n, n_aa = aa_mat.shape
    if n_aa == 0:
        return np.zeros(n, bool), np.zeros(n, bool)

    # Check for '*' in all but the last column (premature stop)
    if n_aa > 1:
        has_premature_stop = (aa_mat[:, :-1] == stop_ord).any(axis=1)
    else:
        has_premature_stop = np.zeros(n, dtype=bool)

    # Readthrough: WT ends in '*' but this sequence doesn't
    if wt_has_terminal_stop:
        has_readthrough = aa_mat[:, -1] != stop_ord
    else:
        has_readthrough = np.zeros(n, dtype=bool)

    return has_premature_stop, has_readthrough


# ---------------------------------------------------------------------------
# Number of mutated codons
# ---------------------------------------------------------------------------

def n_mut_codons(nt_mat: np.ndarray, wt_nt_arr: np.ndarray) -> np.ndarray:
    """Count the number of distinct codons affected by mutations.

    Parameters
    ----------
    nt_mat:
        (n, L) uint8 nt matrix (L divisible by 3).
    wt_nt_arr:
        1D uint8 array of length L (lower-cased WT nt sequence).

    Returns
    -------
    np.ndarray of shape (n,) dtype int32.

    Replaces: per-row strsplit in dimsum__process_merged_variants.R:276.
    """
    n, L = nt_mat.shape
    n_codons = L // 3
    diff = (nt_mat != wt_nt_arr).reshape(n, n_codons, 3)  # (n, n_codons, 3)
    # True if any base in the codon differs
    codon_differs = diff.any(axis=2)  # (n, n_codons)
    return codon_differs.sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Permitted mutation mask
# ---------------------------------------------------------------------------

def permitted_mask(
    nt_mat: np.ndarray,
    permitted_sequences: str,
    wt_coded: str,
) -> np.ndarray:
    """Return a boolean array indicating which sequences have only permitted mutations.

    Parameters
    ----------
    nt_mat:
        (n, L_variable) uint8 matrix of *variable-position only* sequences
        (constant regions already stripped, lower-cased).
    permitted_sequences:
        IUPAC string of length L_variable (upper-case).
    wt_coded:
        Case-coded full WT sequence (for reference; variable positions only
        are addressed by permitted_sequences).

    Returns
    -------
    np.ndarray of shape (n,) dtype bool.

    Replaces: dimsum__identify_permitted_mutations.R loop.
    """
    n, L = nt_mat.shape
    if len(permitted_sequences) != L:
        raise ValueError(
            f"permitted_sequences length ({len(permitted_sequences)}) != "
            f"number of variable positions ({L})"
        )
    ok = np.ones(n, dtype=bool)
    for j, iupac in enumerate(permitted_sequences):
        allowed_ords = np.array(
            sorted(_IUPAC_EXPAND[iupac.upper()]), dtype=np.uint8
        )
        col = nt_mat[:, j]
        # A position is OK if its base is in the allowed set
        pos_ok = np.isin(col, allowed_ords)
        ok &= pos_ok
    return ok


# ---------------------------------------------------------------------------
# Mutation position identification (for singles / doubles tables)
# ---------------------------------------------------------------------------

def mutation_positions(
    seq_mat: np.ndarray,
    ref_arr: np.ndarray,
) -> list[np.ndarray]:
    """Return the 1-based positions where each row differs from ref.

    Parameters
    ----------
    seq_mat:
        (n, L) uint8 matrix.
    ref_arr:
        1D uint8 array of length L.

    Returns
    -------
    List of length n, where element i is a 1D int32 array of 1-based
    positions where row i differs from ref_arr.  Sorted ascending.
    """
    diff = seq_mat != ref_arr  # (n, L) bool
    result = []
    for i in range(len(seq_mat)):
        positions = np.where(diff[i])[0].astype(np.int32) + 1  # 1-based
        result.append(positions)
    return result
