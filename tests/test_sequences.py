"""Unit tests for pydimsum.steam.sequences — vectorized sequence operations."""

import numpy as np
import pytest

from pydimsum.steam.sequences import (
    constant_region_matches_wt,
    decode_row,
    detect_stop,
    encode,
    hamming,
    n_mut_codons,
    permitted_mask,
    strip_constant_regions,
    variable_position_mask,
)


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------

def test_encode_basic():
    seqs = ["acgt", "aaaa"]
    mat = encode(seqs)
    assert mat.shape == (2, 4)
    assert mat.dtype == np.uint8
    assert mat[0, 0] == ord("a")
    assert mat[1, 0] == ord("a")


def test_decode_round_trip():
    seqs = ["acgt", "tgca"]
    mat = encode(seqs)
    assert decode_row(mat[0]) == "acgt"
    assert decode_row(mat[1]) == "tgca"


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------

def test_hamming_identical():
    seqs = ["acgt"]
    mat = encode(seqs)
    result = hamming(mat, "acgt")
    assert result[0] == 0


def test_hamming_all_different():
    seqs = ["tttt"]
    mat = encode(seqs)
    result = hamming(mat, "aaaa")
    assert result[0] == 4


def test_hamming_one_mutation():
    seqs = ["acgt", "tcgt", "acgt"]
    mat = encode(seqs)
    result = hamming(mat, "acgt")
    np.testing.assert_array_equal(result, [0, 1, 0])


def test_hamming_multiple_sequences():
    seqs = ["aaaa", "aaat", "aatt", "attt"]
    mat = encode(seqs)
    result = hamming(mat, "aaaa")
    np.testing.assert_array_equal(result, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Variable position mask
# ---------------------------------------------------------------------------

def test_variable_position_mask_all_variable():
    mask = variable_position_mask("ACGT")
    np.testing.assert_array_equal(mask, [True, True, True, True])


def test_variable_position_mask_all_constant():
    mask = variable_position_mask("acgt")
    np.testing.assert_array_equal(mask, [False, False, False, False])


def test_variable_position_mask_mixed():
    mask = variable_position_mask("aGTa")
    np.testing.assert_array_equal(mask, [False, True, True, False])


# ---------------------------------------------------------------------------
# Constant region filtering
# ---------------------------------------------------------------------------

def test_constant_region_matches_wt_no_constant():
    wt = "ACGT"  # all variable
    seqs = ["acgt", "tttt"]
    mat = encode(seqs)
    result = constant_region_matches_wt(mat, wt)
    # No constant region — all pass
    np.testing.assert_array_equal(result, [True, True])


def test_constant_region_matches_wt_with_constant():
    # WT: positions 1,3 are constant (lower 'a' and 'A' → both map to 'a' after lower()),
    # positions 0 and 2 are variable (upper G, t)
    # Wait: "aGtA" → lower-case = positions 0 (a) and 2 (t); upper-case = positions 1 (G) and 3 (A)
    # So constant positions are 0 and 2 with wt chars 'a' and 't'.
    wt_coded = "aGtA"  # lower = constant at pos 0,2; upper = variable at pos 1,3
    # seqs_ok: constant positions (0,2) must match wt (a,t)
    seqs_ok = ["agta", "acta", "atta"]  # all have 'a' at pos 0 and 't' at pos 2
    mat_ok = encode(seqs_ok)
    result = constant_region_matches_wt(mat_ok, wt_coded)
    np.testing.assert_array_equal(result, [True, True, True])


def test_constant_region_filters_mutated():
    # "aGTa": lower-case = positions 0 (a) and 3 (a) are constant
    # upper-case = positions 1 (G) and 2 (T) are variable
    wt_coded = "aGTa"  # const at pos 0 (wt='a') and pos 3 (wt='a')
    # seq 0: pos0='a' ✓, pos3='a' ✓ → True
    # seq 1: pos0='a' ✓, pos3='a' ✓ → True (pos1 varies freely)
    # seq 2: pos0='a' ✓, pos3='t' ✗ → False (constant region mutated)
    seqs = ["agta", "acta", "agtT".lower()]
    # seqs[0]='agta': const=[a(0), a(3)] OK
    # seqs[1]='acta': const=[a(0), a(3)] OK
    # seqs[2]='agtt': const=[a(0), t(3)] NOT OK (pos3 = t ≠ wt a)
    seqs = ["agta", "acta", "agtt"]
    mat = encode(seqs)
    result = constant_region_matches_wt(mat, wt_coded)
    np.testing.assert_array_equal(result, [True, True, False])


# ---------------------------------------------------------------------------
# Strip constant regions
# ---------------------------------------------------------------------------

def test_strip_constant_regions():
    wt_coded = "aGTa"
    mask = variable_position_mask(wt_coded)  # [F, T, T, F]
    seqs = ["agta", "cgca"]
    mat = encode(seqs)
    stripped = strip_constant_regions(mat, mask)
    assert stripped.shape == (2, 2)
    assert chr(stripped[0, 0]) == "g"
    assert chr(stripped[0, 1]) == "t"
    assert chr(stripped[1, 0]) == "g"
    assert chr(stripped[1, 1]) == "c"


# ---------------------------------------------------------------------------
# STOP detection
# ---------------------------------------------------------------------------

def test_detect_stop_no_stop():
    # ATC = Ile, GGT = Gly, AAA = Lys → "IGK"
    aa_seqs = ["IGK"]
    mat = encode(aa_seqs)
    stop, rt = detect_stop(mat, wt_has_terminal_stop=False)
    assert not stop[0]
    assert not rt[0]


def test_detect_stop_premature():
    aa_seqs = ["I*K"]
    mat = encode(aa_seqs)
    stop, rt = detect_stop(mat, wt_has_terminal_stop=False)
    assert stop[0]


def test_detect_stop_terminal_only():
    aa_seqs = ["IGK*"]
    mat = encode(aa_seqs)
    stop, rt = detect_stop(mat, wt_has_terminal_stop=True)
    assert not stop[0]   # no premature stop
    assert not rt[0]     # ends with *


def test_detect_readthrough():
    aa_seqs = ["IGKL"]  # WT has terminal stop but this doesn't
    mat = encode(aa_seqs)
    stop, rt = detect_stop(mat, wt_has_terminal_stop=True)
    assert not stop[0]
    assert rt[0]   # readthrough


# ---------------------------------------------------------------------------
# n_mut_codons
# ---------------------------------------------------------------------------

def test_n_mut_codons_zero():
    wt_nt = "atggct"  # 2 codons: atg, gct
    seqs = ["atggct"]
    mat = encode(seqs)
    wt_arr = np.frombuffer(wt_nt.encode("ascii"), dtype=np.uint8)
    result = n_mut_codons(mat, wt_arr)
    assert result[0] == 0


def test_n_mut_codons_one_codon():
    wt_nt = "atggct"
    seqs = ["atggcc"]   # last codon: gct → gcc (synonymous but still 1 codon affected)
    mat = encode(seqs)
    wt_arr = np.frombuffer(wt_nt.encode("ascii"), dtype=np.uint8)
    result = n_mut_codons(mat, wt_arr)
    assert result[0] == 1


def test_n_mut_codons_two_codons():
    wt_nt = "atggct"
    seqs = ["ttggcc"]   # both codons mutated
    mat = encode(seqs)
    wt_arr = np.frombuffer(wt_nt.encode("ascii"), dtype=np.uint8)
    result = n_mut_codons(mat, wt_arr)
    assert result[0] == 2


# ---------------------------------------------------------------------------
# permitted_mask
# ---------------------------------------------------------------------------

def test_permitted_mask_all_n():
    # N = any base allowed
    seqs = ["acgt"]
    mat = encode(seqs)
    mask = permitted_mask(mat, "NNNN", "ACGT")
    assert mask[0] == True


def test_permitted_mask_restricted():
    # Only A allowed at position 0 (IUPAC R = A or G)
    seqs = ["acgt", "gcgt", "tcgt"]
    mat = encode(seqs)
    # R=AG, N=any, N=any, N=any
    result = permitted_mask(mat, "RNNN", "ACGT")
    # a (ok), g (ok), t (not ok — R only allows A/G)
    np.testing.assert_array_equal(result, [True, True, False])
