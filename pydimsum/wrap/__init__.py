"""WRAP stage helpers: binary presence checks.

Mirrors: R/dimsum.R:151-194 (binary presence/version checks).
"""

from __future__ import annotations

import shutil


# Binaries required for each WRAP stage
_STAGE_BINARIES: dict[int, list[str]] = {
    0: ["cutadapt"],
    1: ["fastqc"],
    2: ["cutadapt"],
    3: ["vsearch", "starcode"],
}


class BinaryNotFoundError(RuntimeError):
    """Raised when a required external binary is not on PATH."""


def check_binaries(stages: list[int] | None = None) -> None:
    """Check that required external binaries are present on PATH.

    Parameters
    ----------
    stages:
        List of stage numbers to check (0–3). If None, checks all WRAP stages.

    Raises
    ------
    BinaryNotFoundError
        If any required binary is missing, with an informative message.
    """
    if stages is None:
        stages = list(_STAGE_BINARIES)

    required: set[str] = set()
    for s in stages:
        required.update(_STAGE_BINARIES.get(s, []))

    missing = [b for b in sorted(required) if shutil.which(b) is None]
    if missing:
        tips = {
            "cutadapt": "pip install cutadapt  OR  conda install -c bioconda cutadapt",
            "fastqc": "conda install -c bioconda fastqc  OR  download from https://www.bioinformatics.babraham.ac.uk/projects/fastqc/",
            "vsearch": "conda install -c bioconda vsearch  OR  https://github.com/torognes/vsearch",
            "starcode": "conda install -c bioconda starcode  OR  https://github.com/gui11aume/starcode",
        }
        lines = [f"Required binaries not found on PATH: {', '.join(missing)}"]
        for b in missing:
            if b in tips:
                lines.append(f"  {b}: {tips[b]}")
        raise BinaryNotFoundError("\n".join(lines))
