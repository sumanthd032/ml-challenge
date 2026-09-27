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
~35 GB), `BER_FEATS` (feature tables and scores, ~30 GB; defaults to `BER_ARTIFACTS`), `BER_OUTPUT` (final TSVs).

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
```
The feature stage of steps 5 and 6 checkpoints to `BER_FEATS/<split>_store/` (see `store.py`). After a crash,
rerun the same command and it continues from the last finished shard. Delete the store folder to rebuild it from scratch.
```bash
python ../../../student_resource/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../student_resource/dataset/test
```
`learn_dict.py` must run before `preprocess.py`, because the normalizer loads `indic_dict.json`.

## Full pipeline of the final submission (after the v1 steps above)
Environment for all steps: `BER_FEATS=<feature dir>`, `HF_HUB_OFFLINE=1` once the base models are cached.
```bash
# cross-encoders (stage-2 features), see run_ce.sh / run_ce2.sh
python crossenc.py pairs
python crossenc.py train                                   # MiniLM-L6 -> artifacts/crossenc
python crossenc.py score --split val ; python crossenc.py score --split test
python crossenc.py train --base cross-encoder/ms-marco-MiniLM-L12-v2 --tag l12 --max-pairs 12000000
python crossenc.py score --split val --tag l12 --base cross-encoder/ms-marco-MiniLM-L12-v2
python crossenc.py score --split test --tag l12 --base cross-encoder/ms-marco-MiniLM-L12-v2
# stage 2 (India/US model): context + both cross-encoders + twin/name + density features, no ghost rows (D-020)
python stage2.py fit   --tag v1 --ce --ce2 l12 --x --dens --noghost
python stage2.py apply --tag v1 --ce --ce2 l12 --x --dens --noghost --split test
python stage2.py fit   --tag v1 --ce --x --dens            # France model (v1cexd)
# France-only pipeline (D-018): split testfr, bi-encoder + cross-encoder self-training, IDF on train scale
python france_adapt.py split
python france_adapt.py pseudo
python embed.py finetune_pseudo --split testfr --pairs <FEAT>/testfr_pseudo_pairs.parquet --model-dir biencoder_fr --epochs 2
python embed.py knn --split testfr --model-dir biencoder_fr
BER_IDF_SCALE=2 python france_adapt.py stage1
python crossenc.py score --split testfr                   # variant A: original cross-encoder
python france_adapt.py ce_pairs_nn
python crossenc.py train_pseudo --split testfr --pairs <FEAT>/testfr_ce_pairs.parquet --init crossenc --tag fr3 --lr 1e-5 --anchor 2000000
python crossenc.py score --split testfr --tag fr3         # variant D: self-trained cross-encoder
python stage2.py apply --tag v1 --ce --x --dens --split testfr
python stage2.py apply --tag v1 --ce --x --dens --split testfr --ce-sfx _fr3
python france_adapt.py blend --blend testfr_scores2_v1cexd.parquet testfr_fr3_scores2_v1cexd.parquet testfr_blendAD_scores2.parquet
# France names with the France normalization fixes (used to judge vocabulary swaps, D-021): split testfr3
BER_FR_SPLIT=testfr3 python france_adapt.py renorm
# final file (submission 12, public 0.988865): India/US thr 0.5 + compete 2; France thr 0.8, vocabulary swaps
# removed (vswap.py), high-precision categories recovered, test-measured corrections (refine.py)
python france_adapt.py assemble --iu-scores test_scores2_v1cel12xdg.parquet --iu-thr 0.5 --iu-compete 2 \
    --fr-scores testfr_blendAD_scores2.parquet --fr-thr 0.8 --vswap-norm testfr3 --fr-recover --refine \
    --out matching_results.tsv
python france_adapt.py candidates     # -> output/candidate_pairs_fa.tsv (France rows = the testfr candidate set);
                                      # copy it over output/candidate_pairs.tsv afterwards (it streams from that file)
```
`testlike.py` scores validation the way the test sees it (D-020); `decide.compete` enforces one S1 per record.
Checkpointed steps (stage-2 parts, cross-encoder score parts, embeddings) are reused by name: delete them before
rerunning a step with a changed model under the same tag.

## Outputs
* `output/candidate_pairs.tsv`: exactly the pairs the LightGBM model scores (union of both blocking passes).
* `output/matching_results.tsv`: final matches, one row per test S1, always a subset of the candidates.

See `docs/` (ARCHITECTURE, DECISIONS, EDA, RESEARCH, SUBMISSIONS) for the design and history.
