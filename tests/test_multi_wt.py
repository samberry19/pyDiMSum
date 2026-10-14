"""Tests for multiple wildtype (WT) sequences in mutation mode."""

from __future__ import annotations

import math
import random
from pathlib import Path

import polars as pl
import pytest

from pydimsum.config import RunConfig

DATA_DIR = Path(__file__).parent / "data"
BASES = "acgt"

# Two WTs of different length; WT_B has a lower-case (constant) 3-nt flank on each side
WT_A = "acgttgcaagcttcggatccagtcaatgca"          # 30 nt, all variable
WT_B_CODED = "gga" + "TTAGCCATCGGAACTGCATTCGAGTCCTAA" + "tcc"  # 36 nt, 30 variable
WT_B = WT_B_CODED.lower()
VAR_B = range(3, 33)                              # variable positions of WT_B


@pytest.fixture()
def design_path(tmp_path) -> Path:
    p = tmp_path / "design.txt"
    p.write_text(
        "sample_name\texperiment_replicate\tselection_id\tselection_replicate\t"
        "technical_replicate\tpair1\tpair2\n"
        "input1\t1\t0\t\t\tx_1.fq\tx_2.fq\n"
        "output1A\t1\t1\t1\t\ty_1.fq\ty_2.fq\n"
    )
    return p


def _config(design_path, wts=(WT_A.upper(), WT_B_CODED), **kw) -> RunConfig:
    return RunConfig(
        experiment_design_path=design_path,
        wildtype_sequence=",".join(wts),
        sequence_type="noncoding",
        **kw,
    )


def _mutate(seq: str, pos: int, base: str) -> str:
    return seq[:pos] + base + seq[pos + 1:]


def _variants_of(
    wt: str, rng: random.Random, n_doubles: int = 40, positions: range | None = None
) -> list[str]:
    """WT + all singles + some doubles at ``positions`` (all lower-case)."""
    positions = positions if positions is not None else range(len(wt))
    out = [wt]
    for i in positions:
        out += [_mutate(wt, i, x) for x in BASES if x != wt[i]]
    for _ in range(n_doubles):
        i, j = rng.sample(list(positions), 2)
        s = _mutate(wt, i, rng.choice([x for x in BASES if x != wt[i]]))
        out.append(_mutate(s, j, rng.choice([x for x in BASES if x != wt[j]])))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_single_wt_unchanged(self, design_path):
        cfg = _config(design_path, wts=(WT_A.upper(),))
        assert len(cfg.wildtypes) == 1
        assert cfg.wildtypes[0].name == "WT1"
        assert cfg.wildtype_sequence == WT_A.upper()
        assert cfg.permitted_sequences == "N" * 30

    def test_comma_separated_wts_and_default_names(self, design_path):
        cfg = _config(design_path)
        assert [w.name for w in cfg.wildtypes] == ["WT1", "WT2"]
        assert cfg.wildtype_sequence == WT_A.upper()     # primary
        assert cfg.wt_nt_seq == WT_A
        assert cfg.wildtypes[1].has_constant_region
        assert cfg.wildtypes[1].variable_seq == WT_B_CODED[3:-3]

    def test_custom_names(self, design_path):
        cfg = _config(design_path, wildtype_names="alpha,beta")
        assert [w.name for w in cfg.wildtypes] == ["alpha", "beta"]

    def test_name_count_mismatch(self, design_path):
        with pytest.raises(ValueError, match="wildtypeNames"):
            _config(design_path, wildtype_names="only_one")

    def test_duplicate_names(self, design_path):
        with pytest.raises(ValueError, match="unique"):
            _config(design_path, wildtype_names="x,x")

    def test_duplicate_sequences(self, design_path):
        with pytest.raises(ValueError, match="duplicate"):
            _config(design_path, wts=(WT_A.upper(), WT_A.lower()))

    def test_invalid_characters(self, design_path):
        with pytest.raises(ValueError, match="invalid characters"):
            _config(design_path, wts=(WT_A.upper(), "ACGTN"))

    def test_per_wt_permitted_sequences(self, design_path):
        cfg = _config(design_path, permitted_sequences="N" * 30 + "," + "R" * 30)
        assert cfg.wildtypes[0].permitted == "N" * 30
        assert cfg.wildtypes[1].permitted == "R" * 30
        assert cfg.permitted_sequences == "N" * 30

    def test_permitted_length_checked_per_wt(self, design_path):
        # WT_B has 30 variable positions, not 36
        with pytest.raises(ValueError, match="permittedSequences length"):
            _config(design_path, permitted_sequences="N" * 30 + "," + "N" * 36)

    def test_permitted_entry_count_mismatch(self, design_path):
        with pytest.raises(ValueError, match="entries"):
            _config(design_path, wts=(WT_A.upper(), WT_B_CODED, "ACGT"),
                    permitted_sequences="N" * 30 + "," + "N" * 30)

    def test_mixed_coding_noncoding_auto_rejected(self, design_path):
        coding = "ATGGCTGCTGCT"        # M A A A
        noncoding = "ATGTAATAGTGA"     # premature stops
        with pytest.raises(ValueError, match="different sequence types"):
            RunConfig(
                experiment_design_path=design_path,
                wildtype_sequence=f"{coding},{noncoding}",
            )


# ---------------------------------------------------------------------------
# process_variants
# ---------------------------------------------------------------------------

def _count_df(seqs_counts: dict[str, int]) -> pl.DataFrame:
    return pl.DataFrame({
        "nt_seq": list(seqs_counts),
        "count_e1_s0": list(seqs_counts.values()),
        "count_e1_s1": list(seqs_counts.values()),
    })


class TestProcessVariants:
    @pytest.fixture()
    def processed(self, design_path):
        from pydimsum.steam.process_variants import process_variants

        rng = random.Random(1)
        seqs_a = _variants_of(WT_A, rng)
        seqs_b = _variants_of(WT_B, rng, positions=VAR_B)
        # WT_B variants with a mutated constant flank must be rejected (constant region)
        bad_const = _mutate(WT_B, 0, "c" if WT_B[0] != "c" else "t")
        df = _count_df({s: 100 for s in seqs_a + seqs_b + [bad_const]})
        cfg = _config(design_path)
        return process_variants(df, cfg), seqs_a, seqs_b, bad_const

    def test_assignment(self, processed):
        res, seqs_a, seqs_b, _ = processed
        r = res.retained
        a = r.filter(pl.col("WT_id") == "WT1")
        b = r.filter(pl.col("WT_id") == "WT2")
        assert a.height == len(seqs_a)
        assert b.height == len(seqs_b)

    def test_wt_flags(self, processed):
        res, *_ = processed
        r = res.retained
        # WT = primary only; WT_ref = each WT's own sequence
        assert r.filter(pl.col("WT"))["WT_id"].to_list() == ["WT1"]
        refs = r.filter(pl.col("WT_ref"))
        assert sorted(refs["WT_id"].to_list()) == ["WT1", "WT2"]

    def test_nham_relative_to_own_wt(self, processed):
        res, *_ = processed
        r = res.retained
        for wt_id in ("WT1", "WT2"):
            g = r.filter(pl.col("WT_id") == wt_id)
            counts = g["Nham_nt"].value_counts().to_dict(as_series=False)
            by_n = dict(zip(counts["Nham_nt"], counts["count"]))
            assert by_n[0] == 1
            assert by_n[1] == 90
            assert max(by_n) == 2

    def test_constant_region_stripped_only_for_wt_b(self, processed):
        res, *_ = processed
        r = res.retained
        assert set(r.filter(pl.col("WT_id") == "WT1")["nt_seq"].str.len_chars()) == {30}
        # WT_B flanks stripped -> 30 variable positions
        assert set(r.filter(pl.col("WT_id") == "WT2")["nt_seq"].str.len_chars()) == {30}
        assert res.rejected.filter(pl.col("constant_region") == False).height == 1  # noqa: E712

    def test_wt_refs(self, processed):
        res, *_ = processed
        assert res.wt_refs["WT1"][0] == WT_A
        assert res.wt_refs["WT2"][0] == WT_B[3:-3]

    def test_missing_primary_wt_raises(self, design_path):
        from pydimsum.steam.process_variants import process_variants

        rng = random.Random(2)
        seqs_b = _variants_of(WT_B, rng, positions=VAR_B)
        df = _count_df({s: 100 for s in seqs_b})
        with pytest.raises(RuntimeError, match="primary WT"):
            process_variants(df, _config(design_path))

    def test_missing_secondary_wt_only_warns(self, design_path):
        from pydimsum.steam.process_variants import process_variants

        rng = random.Random(3)
        seqs_a = _variants_of(WT_A, rng)
        res = process_variants(_count_df({s: 100 for s in seqs_a}), _config(design_path))
        assert set(res.retained["WT_id"].to_list()) == {"WT1"}

    def test_equidistant_variant_rejected(self, design_path):
        """Same-length WTs: a sequence at equal distance from both is ambiguous."""
        from pydimsum.steam.process_variants import process_variants

        wt_c = _mutate(_mutate(WT_A, 3, "a"), 10, "g")      # differs from WT_A at 2 positions
        assert wt_c != WT_A
        # carries WT_A's base at pos 3 and WT_C's base at pos 10 -> dist 1 to each
        ambiguous = _mutate(WT_A, 10, "g")
        rng = random.Random(4)
        seqs = _variants_of(WT_A, rng) + _variants_of(wt_c, rng)
        seqs = [s for s in dict.fromkeys(seqs) if s != ambiguous]
        df = _count_df({s: 100 for s in seqs + [ambiguous]})
        cfg = _config(design_path, wts=(WT_A.upper(), wt_c.upper()))
        res = process_variants(df, cfg)
        amb = res.rejected.filter(pl.col("ambiguous_wt").fill_null(False))["nt_seq"].to_list()
        assert ambiguous in amb
        # every rejected sequence really is equidistant from both WTs
        for s in amb:
            d = [sum(a != b for a, b in zip(s, wt)) for wt in (WT_A, wt_c)]
            assert d[0] == d[1]
        # and nothing equidistant was retained
        for s in res.retained["nt_seq"].to_list():
            d = [sum(a != b for a, b in zip(s, wt)) for wt in (WT_A, wt_c)]
            assert d[0] != d[1]
        assert res.stats["nuc_ambig_dict"]["count_e1_s0"] == 100 * len(amb)

    def test_single_wt_has_no_wt_columns(self, design_path):
        from pydimsum.steam.process_variants import process_variants

        rng = random.Random(5)
        df = _count_df({s: 100 for s in _variants_of(WT_A, rng)})
        res = process_variants(df, _config(design_path, wts=(WT_A.upper(),)))
        assert "WT_id" not in res.retained.columns
        assert "WT_ref" not in res.retained.columns


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e_output(tmp_path_factory) -> Path:
    from pydimsum.pipeline import run_pipeline

    tmp = tmp_path_factory.mktemp("multi_wt_e2e")
    design = tmp / "design.txt"
    design.write_text(
        "sample_name\texperiment_replicate\tselection_id\tselection_replicate\t"
        "technical_replicate\tpair1\tpair2\n"
        "input1\t1\t0\t\t\tx_1.fq\tx_2.fq\n"
        "output1A\t1\t1\t1\t\ty_1.fq\ty_2.fq\n"
    )
    rng = random.Random(7)
    rows = []
    for wt in (WT_A, WT_B):
        for s in _variants_of(wt, rng, positions=VAR_B if wt == WT_B else None):
            n_in = rng.randint(500, 1500) if s != wt else 20000
            fit = 0.0 if s == wt else rng.uniform(-1.0, 0.2)
            # offset secondary WT so its own fitness is distinguishable from 0
            if wt == WT_B:
                fit += 0.5
            n_out = max(1, round(n_in * math.exp(fit)))
            rows.append((s, n_in, n_out))
    count_file = tmp / "counts.txt"
    count_file.write_text(
        "nt_seq\tinput1\toutput1A\n" + "".join(f"{s}\t{i}\t{o}\n" for s, i, o in rows)
    )
    cfg = RunConfig(
        experiment_design_path=design,
        wildtype_sequence=f"{WT_A.upper()},{WT_B_CODED}",
        wildtype_names="A,B",
        sequence_type="noncoding",
        count_path=count_file,
        output_path=tmp / "out",
        project_name="multi",
    )
    run_pipeline(cfg)
    return tmp / "out" / "multi"


def _read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", null_values=["NA"], infer_schema_length=10000)


class TestEndToEnd:
    def test_singles_annotated_per_wt(self, e2e_output):
        singles = _read(e2e_output / "fitness_singles.txt")
        assert set(singles["WT_id"]) == {"A", "B"}
        for wt_id, wt in (("A", WT_A), ("B", WT_B[3:-3])):
            g = singles.filter(pl.col("WT_id") == wt_id)
            assert g.height == 90
            for row in g.iter_rows(named=True):
                assert wt[row["Pos"] - 1] == row["WT_nt"].lower()
                assert row["Mut"] != row["WT_nt"].lower()

    def test_doubles_annotated_per_wt(self, e2e_output):
        doubles = _read(e2e_output / "fitness_doubles.txt")
        assert set(doubles["WT_id"]) == {"A", "B"}
        for wt_id, wt in (("A", WT_A), ("B", WT_B[3:-3])):
            for row in doubles.filter(pl.col("WT_id") == wt_id).iter_rows(named=True):
                assert wt[row["Pos1"] - 1] == row["WT_nt1"].lower()
                assert wt[row["Pos2"] - 1] == row["WT_nt2"].lower()

    def test_wildtype_table_has_all_wts(self, e2e_output):
        wt = _read(e2e_output / "fitness_wildtype.txt")
        assert sorted(wt["WT_id"].to_list()) == ["A", "B"]
        fit = dict(zip(wt["WT_id"], wt["fitness"]))
        assert fit["A"] == pytest.approx(0.0, abs=1e-6)     # primary is the reference
        assert fit["B"] == pytest.approx(0.5, abs=0.05)     # secondary on the same scale

    def test_mavedb_has_wt_id(self, e2e_output):
        mave = pl.read_csv(e2e_output / "fitness_singles_MaveDB.csv")
        assert "WT_id" in mave.columns
