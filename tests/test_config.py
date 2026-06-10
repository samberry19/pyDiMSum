"""Tests for RunConfig validation."""

from pathlib import Path
import pytest

DATA_DIR = Path(__file__).parent / "data"
DESIGN = DATA_DIR / "experimentDesign_Toy.txt"
COUNT = DATA_DIR / "countFile_Toy.txt"

# WT from dimsum.R:136
WT_SEQ = "GGTAATAGCAGAGGGGGTGGAGCTGGTTTGGGAAACAATCAAGGTAGTAATATGGGTGGTGGGATGAACTTTGGTGCGTTCAGCATTAATCCAGCCATGATGGCTGCCGCCCAGGCAGCACTACAG"


def make_config(**kwargs):
    from pydimsum.config import RunConfig
    defaults = dict(
        experiment_design_path=DESIGN,
        wildtype_sequence=WT_SEQ,
        count_path=COUNT,
        output_path=Path("/tmp/pydimsum_test"),
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)


def test_basic_config_valid():
    cfg = make_config()
    assert cfg.sequence_type_resolved in ("coding", "noncoding")


def test_sequence_type_auto_detects_coding():
    # WT sequence for demo is coding (no premature stop)
    cfg = make_config()
    assert cfg.sequence_type_resolved == "coding"


def test_permitted_sequences_default_n():
    cfg = make_config()
    # Length should equal number of upper-case (variable) bases in WT
    n_variable = sum(1 for b in WT_SEQ if b.isupper())
    assert len(cfg.permitted_sequences) == n_variable
    assert all(c == "N" for c in cfg.permitted_sequences)


def test_invalid_wt_sequence():
    with pytest.raises(ValueError, match="invalid characters"):
        make_config(wildtype_sequence="ACGX")


def test_invalid_sequence_type():
    with pytest.raises(ValueError):
        make_config(sequence_type="unknown")


def test_indels_none():
    cfg = make_config(indels="none")
    assert cfg._indel_lengths is None


def test_indels_all():
    cfg = make_config(indels="all")
    assert cfg._indel_lengths == []


def test_indels_specific():
    cfg = make_config(indels="5,10,15")
    assert cfg._indel_lengths == [5, 10, 15]


def test_count_threshold_simple():
    from pydimsum.config import _parse_min_count_arg
    assert _parse_min_count_arg("10") == 10


def test_count_threshold_dict():
    from pydimsum.config import _parse_min_count_arg
    result = _parse_min_count_arg("0:5,1:10,2:30")
    assert result == {0: 5, 1: 10, 2: 30}


def test_wt_variable_seq():
    """Variable sequence should only have upper-case bases."""
    cfg = make_config()
    assert cfg.wt_variable_seq == WT_SEQ  # all upper → no constant regions in demo WT
