# Architecture

```
 raw TSVs (S1, S2, S3)
        │
        ▼
 ┌──────────────────┐   learn_dict.py: native-token→English dictionary learned from train pairs
 │ 1. preprocess.py │   normalize.py : names (legal forms, junk, leetspeak, dba/t-a, website names)
 │   normalization  │                  addresses (street types EN/FR/IN, states, numbers)
 └──────────────────┘   translit.py  : rule transliteration for 9 Indic scripts (fallback)
        │  artifacts/{split}_s{n}_norm.tsv
        ▼
 ┌──────────────────┐   per country:
 │ 2. blocking.py   │   typed tokens → IDF (df ≤ 5000) → L2 → sparse cosine
 │  candidate gen.  │   top-30 S1→S2/S3  ∪  top-8 S2/S3→S1  (+ embedding kNN, planned)
 └──────────────────┘
        │  artifacts/{split}_cands_A.parquet   (= candidate_pairs.tsv content)
        ▼
 ┌──────────────────┐   39 pair similarities (rapidfuzz, IDF overlap, numbers, state)
 │ 3. features.py   │   + blocking score/ranks
 │                  │   + competition features (gap/rank vs other S1s of same candidate & vice versa)
 └──────────────────┘
        ▼
 ┌──────────────────┐   LightGBM binary classifier, trained on train split
 │ 4. train.py      │   validation = disjoint S1s, 20% "ghost" S1 removed (distractor simulation)
 └──────────────────┘
        ▼
 ┌──────────────────┐   each S2/S3 → only its best S1 (one-to-many constraint)
 │ 5. decide.py     │   per-S1 top-k maximizing expected F0.5 (empty set if Π(1-p) wins)
 └──────────────────┘
        ▼
 output/matching_results.tsv, output/candidate_pairs.tsv  → utils/validate_submission.py
```

## Code layout (`code/business_entity_resolution/src/`)
| file | role |
|---|---|
| `config.py` | paths (env overridable: `BER_DATA`, `BER_ARTIFACTS`, `BER_OUTPUT`), jobs, seed |
| `translit.py` | Indic → Latin rule transliteration |
| `learn_dict.py` | learns `indic_dict.json` from train labels |
| `normalize.py` | name/address normalization |
| `preprocess.py` | parallel normalization of all sources |
| `blocking.py`, `run_blocking.py` | candidate generation + recall report |
| `features.py` | pairwise + group features |
| `train.py` | feature table, LightGBM training, validation report |
| `decide.py` | assignment + expected-F decision |
| `metrics.py` | macro F0.5 re-implementation, blocking recall |
| `predict.py` | test inference and output writing |
| `store.py` | resumable, sharded feature store (pairs, pairwise features, competition features) |

## Scale notes
* 27M records are normalized in ~6 min on 32 processes.
* Blocking uses sparse matrix products chunked over 16 processes. Features with df > 5000 are dropped,
  which bounds the work per query.
* Features run at ~47 µs/pair per process.
