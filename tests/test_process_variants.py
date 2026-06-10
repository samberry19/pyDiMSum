"""Tests for process_variants — WT guard and _check_variants."""

from __future__ import annotations

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# _check_variants unit tests
# ---------------------------------------------------------------------------

class TestCheckVariants:
    """Tests for the _check_variants WT presence guard."""

    def _make_df(self, n_non_wt: int = 10, include_wt: bool = True) -> pl.DataFrame:
        """Build a minimal substitution DataFrame."""
        rows = []
        count_cols = ["count_e1_s0", "count_e1_s1"]
        for i in range(n_non_wt):
            rows.append({
                "nt_seq": f"ACGT{i:04d}",
                "WT": False,
                "count_e1_s0": 100 + i,
                "count_e1_s1": 90 + i,
            })
        if include_wt:
            rows.append({
                "nt_seq": "ACGTACGT",
                "WT": True,
                "count_e1_s0": 5000,
                "count_e1_s1": 4800,
            })
        return pl.DataFrame(rows)

    def test_passes_when_wt_present_many_variants(self):
        from pydimsum.steam.process_variants import _check_variants
        df = self._make_df(n_non_wt=20, include_wt=True)
        # Should not raise
        _check_variants(df, ["count_e1_s0", "count_e1_s1"])

    def test_raises_when_wt_absent(self):
        from pydimsum.steam.process_variants import _check_variants
        df = self._make_df(n_non_wt=20, include_wt=False)
        with pytest.raises(RuntimeError, match="WT variant not found"):
            _check_variants(df, ["count_e1_s0", "count_e1_s1"])

    def test_error_message_includes_top_candidates(self):
        from pydimsum.steam.process_variants import _check_variants
        df = self._make_df(n_non_wt=10, include_wt=False)
        with pytest.raises(RuntimeError) as exc_info:
            _check_variants(df, ["count_e1_s0", "count_e1_s1"])
        msg = str(exc_info.value)
        # Should include at least one sequence from the top-5 list
        assert "ACGT" in msg

    def test_raises_when_fewer_than_2_non_wt(self):
        from pydimsum.steam.process_variants import _check_variants
        df = self._make_df(n_non_wt=1, include_wt=True)
        with pytest.raises(RuntimeError, match="Fewer than 2 non-WT"):
            _check_variants(df, ["count_e1_s0", "count_e1_s1"])

    def test_passes_exactly_2_non_wt(self):
        from pydimsum.steam.process_variants import _check_variants
        df = self._make_df(n_non_wt=2, include_wt=True)
        # Exactly 2 non-WT should pass
        _check_variants(df, ["count_e1_s0", "count_e1_s1"])

    def test_raises_zero_non_wt(self):
        from pydimsum.steam.process_variants import _check_variants
        df = self._make_df(n_non_wt=0, include_wt=True)
        with pytest.raises(RuntimeError, match="Fewer than 2"):
            _check_variants(df, ["count_e1_s0", "count_e1_s1"])

    def test_wt_null_treated_as_false(self):
        """WT=null rows should not count as WT."""
        from pydimsum.steam.process_variants import _check_variants
        df = pl.DataFrame({
            "nt_seq": ["seq1", "seq2", "seq3"],
            "WT": [None, None, None],
            "count_e1_s0": [100, 200, 300],
        })
        with pytest.raises(RuntimeError, match="WT variant not found"):
            _check_variants(df, ["count_e1_s0"])
