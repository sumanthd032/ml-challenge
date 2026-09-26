#!/usr/bin/env bash
# Stage 2 with both cross-encoders (L6 --ce + L12 --ce2 l12) and the --x features: model tag v1cel12x.
# Needs run_ce2.sh to have finished. Resumable like run_ce.sh: after a machine reset just run it again.
#   bash run_s2_l12.sh        (from code/business_entity_resolution/src)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-C:/Users/Admin/anaconda3/python.exe}
ART=../../../artifacts
FEAT=${BER_FEATS:-$ART}
FEAT=${FEAT//\\//}
OUT=../../../output
LOG=$ART/crossenc_l12.log
TAG=v1cel12x
FLAGS="--tag v1 --ce --ce2 l12 --x"
RES=variants/matching_results_cel12x_thr70.tsv
step() { echo "=== $1 $(date) ===" | tee -a "$LOG"; }

for s in val test; do
  [ -f "$FEAT/ce_scores_${s}_l12.parquet" ] || { echo "missing ce_scores_${s}_l12.parquet: run run_ce2.sh first"; exit 1; }
done
[ -f "$ART/lgb2_$TAG.txt" ] || { step "stage2 fit $TAG"; $PY -u stage2.py fit $FLAGS > "$ART/stage2_$TAG.log" 2>&1; }
[ -f "$FEAT/test_scores2_$TAG.parquet" ] || { step "stage2 apply $TAG"; $PY -u stage2.py apply $FLAGS >> "$ART/stage2_apply_$TAG.log" 2>&1; }
[ -f "$OUT/$RES" ] || { step "predict $TAG"; $PY -u predict.py --scores "test_scores2_$TAG.parquet" --thr 0.7 --out "$RES" > "$ART/predict_$TAG.log" 2>&1; }
step "validate $TAG"
$PY ../../../student_resource/utils/validate_submission.py --matching "$OUT/$RES" \
    --candidate "$OUT/candidate_pairs.tsv" --test-dir ../../../student_resource/dataset/test 2>&1 | tee "$ART/validate_$TAG.log"
step "stage2 $TAG done"
