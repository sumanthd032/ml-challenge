"""Central configuration: paths and hyper-parameters. Override the data root with env var BER_DATA."""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent                                   # code/business_entity_resolution
DATA_DIR = Path(os.environ.get("BER_DATA", PROJECT_DIR.parent.parent / "student_resource" / "dataset"))
ART_DIR = Path(os.environ.get("BER_ARTIFACTS", PROJECT_DIR.parent.parent / "artifacts"))
OUT_DIR = Path(os.environ.get("BER_OUTPUT", PROJECT_DIR.parent.parent / "output"))
FEAT_DIR = Path(os.environ.get("BER_FEATS", ART_DIR))        # large feature tables (tens of GB)
ART_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = int(os.environ.get("BER_JOBS", max(1, (os.cpu_count() or 4) - 4)))
SEED = 42


GHOST_FRAC = 0.2      # train S1s removed from the pool (distractor simulation) & used to fine-tune the bi-encoder
VALID_FRAC = 0.2      # of the remaining train S1s, used for validation


def s1_role(entity_id: str) -> str:
    """Deterministic role of a TRAIN S1 entity: 'ghost' | 'valid' | 'fit' (hash-based, reproducible)."""
    import hashlib
    h = int(hashlib.md5(entity_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < GHOST_FRAC:
        return "ghost"
    if h < GHOST_FRAC + (1 - GHOST_FRAC) * VALID_FRAC:
        return "valid"
    return "fit"


def source_path(split: str, s: int) -> Path:
    """Raw source file of `split` ('train' or 'test') and source number s (1, 2 or 3)."""
    return DATA_DIR / split / f"{split}_source{s}.tsv"


def norm_path(split: str, s: int) -> Path:
    """Normalized TSV of `split` and source s, written by preprocess.py."""
    return ART_DIR / f"{split}_s{s}_norm.tsv"
