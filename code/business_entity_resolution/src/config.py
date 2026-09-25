"""Central configuration: paths and hyper-parameters. Override the data root with env var BER_DATA."""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent                                   # code/business_entity_resolution
DATA_DIR = Path(os.environ.get("BER_DATA", PROJECT_DIR.parent.parent / "student_resource" / "dataset"))
ART_DIR = Path(os.environ.get("BER_ARTIFACTS", PROJECT_DIR.parent.parent / "artifacts"))
OUT_DIR = Path(os.environ.get("BER_OUTPUT", PROJECT_DIR.parent.parent / "output"))
ART_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = int(os.environ.get("BER_JOBS", max(1, (os.cpu_count() or 4) - 4)))
SEED = 42


def source_path(split: str, s: int) -> Path:
    return DATA_DIR / split / f"{split}_source{s}.tsv"


def norm_path(split: str, s: int) -> Path:
    return ART_DIR / f"{split}_s{s}_norm.tsv"
