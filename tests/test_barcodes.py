"""Tests for barcode identity loading and debarcoding in process_variants."""

import io
import tempfile
from pathlib import Path

import polars as pl
import pytest

from pydimsum.io.designs import load_barcode_identity
from pydimsum.steam.process_variants import _debarcode_variants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tsv(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
    f.write(content)
    f.flush()
    return Path(f.name)


def _make_count_df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal count table."""
    return pl.DataFrame(rows)


class _FakeConfig:
    """Minimal config stub for _debarcode_variants."""
    def __init__(self, path: Path):
        self.barcode_identity_path = path


# ---------------------------------------------------------------------------
# load_barcode_identity
# ---------------------------------------------------------------------------

class TestLoadBarcodeIdentity:
    def test_basic_load(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_text("barcode\tvariant\nACGT\tGGGG\nTTTT\tCCCC\n")
        result = load_barcode_identity(f)
        assert result == {"acgt": "gggg", "tttt": "cccc"}

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_text("barcode\tvariant\nACGT\tGGGG\n")
        result = load_barcode_identity(f)
        assert "acgt" in result
        assert result["acgt"] == "gggg"

    def test_missing_barcode_column_raises(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_text("sequence\tvariant\nACGT\tGGGG\n")
        with pytest.raises(ValueError, match="Mandatory columns missing"):
            load_barcode_identity(f)

    def test_missing_variant_column_raises(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_text("barcode\tseq\nACGT\tGGGG\n")
        with pytest.raises(ValueError, match="Mandatory columns missing"):
            load_barcode_identity(f)

    def test_non_acgt_barcode_raises(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_text("barcode\tvariant\nACGN\tGGGG\n")
        with pytest.raises(ValueError, match="non-ACGT"):
            load_barcode_identity(f)

    def test_non_acgt_variant_raises(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_text("barcode\tvariant\nACGT\tGGXG\n")
        with pytest.raises(ValueError, match="non-ACGT"):
            load_barcode_identity(f)

    def test_empty_file_gives_empty_dict(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_text("barcode\tvariant\n")
        result = load_barcode_identity(f)
        assert result == {}

    def test_crlf_line_endings(self, tmp_path):
        f = tmp_path / "barcodes.tsv"
        f.write_bytes(b"barcode\tvariant\r\nACGT\tGGGG\r\n")
        result = load_barcode_identity(f)
        assert result == {"acgt": "gggg"}


# ---------------------------------------------------------------------------
# _debarcode_variants
# ---------------------------------------------------------------------------

class TestDebarcodeVariants:
    def _make_df_and_config(self, tmp_path, barcode_rows: list[dict],
                             count_rows: list[dict]) -> tuple:
        f = tmp_path / "barcodes.tsv"
        lines = ["barcode\tvariant"]
        for r in barcode_rows:
            lines.append(f"{r['barcode']}\t{r['variant']}")
        f.write_text("\n".join(lines) + "\n")
        config = _FakeConfig(f)
        df = pl.DataFrame(count_rows)
        return df, config

    def test_basic_debarcoding(self, tmp_path):
        df, config = self._make_df_and_config(
            tmp_path,
            [{"barcode": "AAAA", "variant": "CCCC"},
             {"barcode": "TTTT", "variant": "GGGG"}],
            [{"nt_seq": "aaaa", "count_e1_s0": 10, "count_e1_s1": 20},
             {"nt_seq": "tttt", "count_e1_s0": 5,  "count_e1_s1": 8}],
        )
        debarcoded, nobarcode = _debarcode_variants(df, config, ["count_e1_s0", "count_e1_s1"])
        assert "cccc" in debarcoded["nt_seq"].to_list()
        assert "gggg" in debarcoded["nt_seq"].to_list()
        assert len(nobarcode) == 0

    def test_invalid_barcodes_separated(self, tmp_path):
        df, config = self._make_df_and_config(
            tmp_path,
            [{"barcode": "AAAA", "variant": "CCCC"}],
            [{"nt_seq": "aaaa", "count_e1_s0": 10, "count_e1_s1": 20},
             {"nt_seq": "tttt", "count_e1_s0": 3,  "count_e1_s1": 5}],  # unknown
        )
        debarcoded, nobarcode = _debarcode_variants(df, config, ["count_e1_s0", "count_e1_s1"])
        assert len(debarcoded) == 1
        assert len(nobarcode) == 1
        assert nobarcode["nt_seq"].to_list() == ["tttt"]

    def test_aggregation_of_same_variant(self, tmp_path):
        """Two barcodes mapping to same variant → counts summed."""
        df, config = self._make_df_and_config(
            tmp_path,
            [{"barcode": "AAAA", "variant": "CCCC"},
             {"barcode": "TTTT", "variant": "CCCC"}],  # same target
            [{"nt_seq": "aaaa", "count_e1_s0": 10, "count_e1_s1": 20},
             {"nt_seq": "tttt", "count_e1_s0": 5,  "count_e1_s1": 8}],
        )
        debarcoded, nobarcode = _debarcode_variants(df, config, ["count_e1_s0", "count_e1_s1"])
        assert len(debarcoded) == 1  # only one unique variant
        row = debarcoded.row(0, named=True)
        assert row["nt_seq"] == "cccc"
        assert row["count_e1_s0"] == 15  # 10 + 5
        assert row["count_e1_s1"] == 28  # 20 + 8

    def test_no_valid_barcodes_raises(self, tmp_path):
        df, config = self._make_df_and_config(
            tmp_path,
            [{"barcode": "AAAA", "variant": "CCCC"}],
            [{"nt_seq": "tttt", "count_e1_s0": 10, "count_e1_s1": 20}],  # none match
        )
        with pytest.raises(RuntimeError, match="No valid barcodes found"):
            _debarcode_variants(df, config, ["count_e1_s0", "count_e1_s1"])

    def test_multiple_count_cols(self, tmp_path):
        df, config = self._make_df_and_config(
            tmp_path,
            [{"barcode": "AAAA", "variant": "CCCC"}],
            [{"nt_seq": "aaaa", "count_e1_s0": 10, "count_e1_s1": 20,
              "count_e2_s0": 7, "count_e2_s1": 14}],
        )
        count_cols = ["count_e1_s0", "count_e1_s1", "count_e2_s0", "count_e2_s1"]
        debarcoded, _ = _debarcode_variants(df, config, count_cols)
        row = debarcoded.row(0, named=True)
        assert row["count_e2_s0"] == 7
        assert row["count_e2_s1"] == 14
