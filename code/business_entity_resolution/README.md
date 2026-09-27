# Business entity resolution: reproducible pipeline

Links Source-2 and Source-3 business records to the deduplicated Source-1 entities (Amazon ML Challenge 2026).
Stages: normalization -> two-pass blocking (IDF sparse tokens and fine-tuned bi-encoder kNN) -> stage-1 LightGBM,
which also prunes the candidate set to 5.81 pairs per entity -> cross-encoders -> stage-2 LightGBM -> one-to-many
decision with candidate competition. French entities, absent from training, go through an adapted branch
(`france_adapt.py`). Design: `../../docs/ARCHITECTURE.md`; decisions and measured effects: `../../docs/DECISIONS.md`.

## Environment
* Python 3.13, Windows 11 or Linux. Tested with 64 GB and 128 GB RAM; steps with a large in-memory peak are chunked
  to disk, and heavy steps should run one at a time.
* NVIDIA GPU for the bi-encoder and cross-encoders (tested on an RTX A5000 24 GB and an RTX PRO 4500 32 GB with
  CUDA 12.8 wheels). CPU works but is much slower.
* Install: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`
  (the CUDA build of torch is pinned as `torch==2.11.0+cu128`). On Windows, torch needs a current Visual C++
  2015-2022 runtime.
* Base models: `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2` and
  `cross-encoder/ms-marco-MiniLM-L12-v2` (Apache-2.0), downloaded from the Hugging Face hub on first use. Set
  `HF_HUB_OFFLINE=1` once they are cached. No other external data is used.

## Data layout
By default the code expects the challenge folder next to `code/`:
```
<root>/student_resource/dataset/{train,test}/*.tsv    input data
<root>/code/business_entity_resolution/src/*.py       pipeline code
<root>/artifacts/                                     normalized data, embeddings, models, logs
<root>/output/                                        candidate_pairs.tsv, matching_results.tsv, variants/
```
Environment variables override the locations: `BER_DATA` (dataset), `BER_ARTIFACTS` (intermediate files, about
30 GB), `BER_FEATS` (feature stores and score files, about 83 GB; defaults to `BER_ARTIFACTS`), `BER_OUTPUT`
(final TSVs). All paths are defined in `src/config.py`.

## Run the base pipeline (from `src/`)
```bash
python learn_dict.py                    # 1. native-script -> English token dictionary from train labels
python preprocess.py                    # 2. normalize all six source files
python run_blocking.py --split train    # 3a. pass A blocking, train split, with a recall report
python run_blocking.py --split test     # 3b. pass A blocking, test split
python embed.py finetune                # 4a. fine-tune the bi-encoder (GPU)
python embed.py knn --split train       # 4b. encode and run exact GPU kNN (pass B)
python embed.py knn --split test
python train.py --tag v1                # 5. features, stage-1 LightGBM, validation report
python predict.py --model lgb_v1.txt    # 6. test features and stage-1 scores (test_scores_lgb_v1.parquet)
```
`learn_dict.py` must run before `preprocess.py`, because the normalizer loads `indic_dict.json`. The feature stage
of steps 5 and 6 checkpoints to `BER_FEATS/<split>_store/` (`store.py`): after a crash, rerun the same command and
it continues from the last finished shard. Delete the store folder to rebuild from scratch.

## Run the final pipeline
Run heavy steps one at a time. The shell scripts `run_ce.sh`, `run_ce2.sh` and `run_s2_l12.sh` chain the
cross-encoder and stage-2 steps and skip any step whose output already exists.
```bash
# Cross-encoders (stage-2 features)
python crossenc.py pairs
python crossenc.py train                                   # MiniLM-L6 -> artifacts/crossenc
python crossenc.py score --split val
python crossenc.py score --split test
python crossenc.py train --base cross-encoder/ms-marco-MiniLM-L12-v2 --tag l12 --max-pairs 12000000
python crossenc.py score --split val --tag l12 --base cross-encoder/ms-marco-MiniLM-L12-v2
python crossenc.py score --split test --tag l12 --base cross-encoder/ms-marco-MiniLM-L12-v2

# Stage 2: India/US model without ghost rows (D-020), France model
python stage2.py fit   --tag v1 --ce --ce2 l12 --x --dens --noghost
python stage2.py apply --tag v1 --ce --ce2 l12 --x --dens --noghost --split test
python stage2.py fit   --tag v1 --ce --x --dens

# France branch (D-018): split testfr, self-trained bi-encoder and cross-encoder, IDF on train scale
python france_adapt.py split
python france_adapt.py pseudo
python embed.py finetune_pseudo --split testfr --pairs <BER_FEATS>/testfr_pseudo_pairs.parquet --model-dir biencoder_fr --epochs 2
python embed.py knn --split testfr --model-dir biencoder_fr
BER_IDF_SCALE=2 python france_adapt.py stage1
python crossenc.py score --split testfr                    # variant A: original cross-encoder
python france_adapt.py ce_pairs_nn
python crossenc.py train_pseudo --split testfr --pairs <BER_FEATS>/testfr_ce_pairs.parquet --init crossenc --tag fr3 --lr 1e-5 --anchor 2000000
python crossenc.py score --split testfr --tag fr3          # variant D: self-trained cross-encoder
python stage2.py apply --tag v1 --ce --x --dens --split testfr
python stage2.py apply --tag v1 --ce --x --dens --split testfr --ce-sfx _fr3
python france_adapt.py blend --blend testfr_scores2_v1cexd.parquet testfr_fr3_scores2_v1cexd.parquet testfr_blendAD_scores2.parquet
BER_FR_SPLIT=testfr3 python france_adapt.py renorm         # France names used to judge vocabulary swaps (D-021)
```
On Windows PowerShell, set the per-command variables first: `$env:BER_IDF_SCALE='2'`, `$env:BER_FR_SPLIT='testfr3'`
(and remove them afterwards).

## Regenerate the submission files

### candidate_pairs.tsv
```bash
python candidate_set.py                 # --min-prob 0.002 --max-per-s1 50 --out candidate_pairs.tsv
```
Reads the stage-1 scores of the test split (`test_scores_lgb_v1.parquet`, India/US) and of the France branch
(`testfr_scores_lgb_v1.parquet`), keeps pairs with probability at least 0.002 and at most 50 per entity, and writes
one row per test Source-1 entity, candidates ordered by score. Expected summary: 10.07M pairs, 5.81 per entity,
11,171 empty lists. It needs the base pipeline and the France branch up to `france_adapt.py stage1`.

### matching_results.tsv
```bash
python france_adapt.py assemble --iu-scores test_scores2_v1cel12xdg.parquet --iu-thr 0.5 --iu-compete 2 \
    --fr-scores testfr_blendAD_scores2.parquet --fr-thr 0.8 --vswap-norm testfr3 --fr-recover --refine \
    --out variants/matching_results_VRR_IUg50_FADblend_FR80.tsv
```
This writes submission 12 (public 0.988865): India/US threshold 0.5 with competition weight 2; France threshold 0.8,
vocabulary swaps removed (`vswap.py`), high-precision categories recovered, test-measured corrections (`refine.py`).
Submissions 13 to 16 add further corrections measured with the count signature (D-025 to D-030); each decision is
recorded pair by pair in `corrections/corrections.tsv` (16,882 rows). Replaying them on the assembled file
reproduces the final submission (submission 16, public 0.989334):

```bash
python apply_corrections.py --base variants/matching_results_VRR_IUg50_FADblend_FR80.tsv --out matching_results.tsv
```

### Validate
```bash
python ../../../student_resource/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../student_resource/dataset/test
```

## Outputs
* `output/candidate_pairs.tsv`: the pruned candidate set (5.81 per entity). The cross-encoders and stage 2 score
  only these pairs, so every final match is a candidate.
* `output/matching_results.tsv`: final matches, one row per test Source-1 entity, a subset of the candidates.

## Notes
* `testlike.py` scores validation the way the test sees it (no records of absent entities, D-020).
* Checkpointed steps (stage-2 parts, cross-encoder score parts, embeddings) are reused by name: delete them before
  rerunning a step with a changed model under the same tag.
