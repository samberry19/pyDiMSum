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


# ---------------------------------------------------------------------------
# ExperimentDesign tests
# ---------------------------------------------------------------------------

class TestExperimentDesign:
    def _write_design(self, path: Path, extra_col: str = "") -> Path:
        """Write a minimal valid TSV design file."""
        header = "sample_name\texperiment_replicate\tselection_id\tpair1\tpair2"
        if extra_col:
            header += f"\t{extra_col}"
        rows = [
            "input1\t1\t0\tread1_A.fastq.gz\tread2_A.fastq.gz",
            "output1\t1\t1\tread1_B.fastq.gz\tread2_B.fastq.gz",
        ]
        if extra_col:
            rows = [r + "\tval" for r in rows]
        tsv = "\n".join([header] + rows) + "\n"
        design = path / "design.tsv"
        design.write_text(tsv)
        return design

    def test_fastq_file_dir_overrides_pair_directory(self, tmp_path):
        from pydimsum.io.designs import ExperimentDesign
        design = self._write_design(tmp_path)
        exp = ExperimentDesign(design, fastq_file_dir=Path("/data/fastq"))
        assert "pair_directory" in exp.df.columns
        dirs = exp.df["pair_directory"].to_list()
        assert all(d == "/data/fastq" for d in dirs)

    def test_no_fastq_file_dir_no_pair_directory_added(self, tmp_path):
        from pydimsum.io.designs import ExperimentDesign
        design = self._write_design(tmp_path)
        exp = ExperimentDesign(design)
        # pair_directory not added if not in file and fastq_file_dir not given
        if "pair_directory" in exp.df.columns:
            assert exp.df["pair_directory"].is_null().all()

    def test_duplicate_pairs_raises(self, tmp_path):
        from pydimsum.io.designs import ExperimentDesign
        header = "sample_name\texperiment_replicate\tselection_id\tpair1\tpair2"
        rows = [
            "input1\t1\t0\tread1_A.fastq.gz\tread2_A.fastq.gz",
            "input2\t2\t0\tread1_A.fastq.gz\tread2_A.fastq.gz",  # duplicate pair
        ]
        design = tmp_path / "dup.tsv"
        design.write_text("\n".join([header] + rows))
        with pytest.raises(ValueError, match="Duplicate"):
            ExperimentDesign(design)

    def test_duplicate_pairs_allowed_with_flag(self, tmp_path):
        from pydimsum.io.designs import ExperimentDesign
        header = "sample_name\texperiment_replicate\tselection_id\tpair1\tpair2"
        rows = [
            "input1\t1\t0\tread1_A.fastq.gz\tread2_A.fastq.gz",
            "input2\t2\t0\tread1_A.fastq.gz\tread2_A.fastq.gz",
        ]
        design = tmp_path / "dup.tsv"
        design.write_text("\n".join([header] + rows))
        exp = ExperimentDesign(design, allow_pair_duplicates=True)
        assert len(exp.df) == 2

    def test_empty_rows_and_trailing_column_ignored(self, tmp_path):
        """Empty rows and empty trailing columns must be silently dropped."""
        from pydimsum.io.designs import ExperimentDesign
        # Trailing tab produces an empty column; blank lines produce empty rows
        tsv = (
            "sample_name\texperiment_replicate\tselection_id\t\n"
            "input1\t1\t0\t\n"
            "output1\t1\t1\t\n"
            "\t\t\t\n"  # empty row
            "\n"
        )
        design = tmp_path / "trailing.tsv"
        design.write_text(tsv)
        exp = ExperimentDesign(design)
        assert len(exp.df) == 2
        assert "" not in exp.df.columns


class TestLoadCountFile:
    def _make_files(self, tmp_path, counts_content):
        design_tsv = "sample_name\texperiment_replicate\tselection_id\nsA\t1\t0\nsB\t1\t1\n"
        design = tmp_path / "design.tsv"
        design.write_text(design_tsv)
        counts = tmp_path / "counts.tsv"
        counts.write_text(counts_content)
        return design, counts

    def test_empty_rows_and_trailing_column_not_flagged_as_duplicates(self, tmp_path):
        """Empty rows / trailing columns must not trigger the duplicate nt_seq error."""
        from pydimsum.io.counts import load_count_file
        from pydimsum.io.designs import ExperimentDesign

        counts_content = (
            "nt_seq\tsA\tsB\t\n"
            "acgt\t10\t20\t\n"
            "tttt\t5\t15\t\n"
            "\t\t\t\n"  # empty row
            "\n"
        )
        design, counts = self._make_files(tmp_path, counts_content)
        exp = ExperimentDesign(design, count_path=counts)
        df = load_count_file(counts, exp)
        assert len(df) == 2

    def test_actual_duplicate_nt_seq_still_raises(self, tmp_path):
        """Real duplicate nt_seq values must still raise ValueError."""
        from pydimsum.io.counts import load_count_file
        from pydimsum.io.designs import ExperimentDesign

        counts_content = "nt_seq\tsA\tsB\nacgt\t10\t20\nacgt\t5\t15\n"
        design, counts = self._make_files(tmp_path, counts_content)
        exp = ExperimentDesign(design, count_path=counts)
        with pytest.raises(ValueError, match="Duplicated"):
            load_count_file(counts, exp)
