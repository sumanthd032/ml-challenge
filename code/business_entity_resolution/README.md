# Business Entity Resolution — reproducible pipeline

Links Source-2/Source-3 business records to the deduplicated Source-1 reference records.
Stages: normalization → two-pass blocking (sparse IDF tokens + fine-tuned bi-encoder kNN) → pairwise and
competition features → LightGBM → one-to-many assignment + expected-F0.5 decision.

## Environment
* Python 3.13, Windows 11 or Linux. 128 GB RAM recommended for the full data (≥64 GB minimum).
* NVIDIA GPU for the bi-encoder (tested on an RTX A5000 24 GB, CUDA 12.8 wheels). It runs on CPU, much more slowly.
* `pip install -r requirements.txt` (for CUDA torch add
  `--index-url https://download.pytorch.org/whl/cu128` for the torch line).
* Base model `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 22M params) is pulled from the
  Hugging Face hub on first use. No other external resources are used at any stage.

## Data layout
By default the code expects the challenge folder next to `code/`:
```
<root>/student_resource/dataset/{train,test}/*.tsv
<root>/code/business_entity_resolution/src/*.py
```
Override with environment variables: `BER_DATA` (dataset dir), `BER_ARTIFACTS` (intermediate files,
~60 GB), `BER_OUTPUT` (final TSVs).

## Run end-to-end (from `src/`)
```bash
python learn_dict.py                    # 1. native-script -> English token dictionary (train labels)
python preprocess.py                    # 2. normalize all 6 source files            (~6 min)
python run_blocking.py --split train    # 3a. sparse blocking, train (+recall report) (~15 min)
python run_blocking.py --split test     # 3b. sparse blocking, test                  (~12 min)
python embed.py finetune                # 4a. fine-tune bi-encoder on ghost-S1 pairs  (~8 min GPU)
python embed.py knn --split train       # 4b. encode + exact GPU kNN                  (~35 min)
python embed.py knn --split test
python train.py --tag v1                # 5. features + LightGBM + validation report
python predict.py --model lgb_v1.txt    # 6. test features, scoring, decision, outputs
python ../../../student_resource/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../student_resource/dataset/test
```
`learn_dict.py` must run before `preprocess.py`, because the normalizer loads `indic_dict.json`.

## Outputs
* `output/candidate_pairs.tsv`: exactly the pairs the LightGBM model scores (union of both blocking passes).
* `output/matching_results.tsv`: final matches, one row per test S1, always a subset of the candidates.

See `docs/` (ARCHITECTURE, DECISIONS, EDA, RESEARCH, SUBMISSIONS) for the design and history.
