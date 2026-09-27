"""Loading helpers for normalized sources and ground truth."""
import csv

import pandas as pd

import config


def load_norm(split: str, s: int) -> pd.DataFrame:
    """Normalized table of `split` and source s; all columns as str except `indic` (int8)."""
    df =pd.read_csv(config.norm_path(split, s), sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, escapechar="\\")
    df["indic"] = df["indic"].astype("int8")
    return df


def load_split(split: str):
    """(s1, s23): normalized Source 1, and Sources 2 and 3 concatenated in that order."""
    s1 =load_norm(split, 1)
    s23 = pd.concat([load_norm(split, 2), load_norm(split, 3)], ignore_index=True)
    return s1, s23


def load_gt_pairs() -> pd.DataFrame:
    """Ground truth as a long table of (s1_id, cand_id) positive pairs."""
    gt = pd.read_csv(config.DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    gt["cand_id"] = gt.matched_entity_ids.str.split(",")
    long = gt.explode("cand_id")
    long = long[long.cand_id.fillna("") != ""]
    return long.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]].reset_index(drop=True)
