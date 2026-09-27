#!/usr/bin/env bash
# Cross-encoder + stage-2 chain, resumable: after a machine reset just run it again (D-012).
# Every step is skipped when its output exists; the long steps also resume internally
# (crossenc train from crossenc_ckpt.pt, crossenc score / stage2 apply from their saved parts).
# Steps run one after another, never in parallel, to keep the RAM footprint small.
#   bash run_ce.sh            (from code/business_entity_resolution/src)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
ART=../../../artifacts
FEAT=${BER_FEATS:-$ART}
FEAT=${FEAT//\\//}
LOG=$ART/crossenc.log
step() { echo "=== $1 $(date) ===" | tee -a "$LOG"; }

[ -f "$FEAT/ce_train_pairs.parquet" ] || { step "ce pairs"; $PY -u crossenc.py pairs >> "$LOG" 2>&1; }
[ -f "$ART/crossenc/model.safetensors" ] || { step "ce train"; $PY -u crossenc.py train >> "$LOG" 2>&1; }
[ -f "$FEAT/ce_scores_val.parquet" ] || { step "ce score val"; $PY -u crossenc.py score --split val >> "$LOG" 2>&1; }
[ -f "$FEAT/ce_scores_test.parquet" ] || { step "ce score test"; $PY -u crossenc.py score --split test >> "$LOG" 2>&1; }
[ -f "$ART/lgb2_v1ce.txt" ] || { step "stage2 fit --ce"; $PY -u stage2.py fit --tag v1 --ce > "$ART/stage2_v1ce.log" 2>&1; }
[ -f "$FEAT/test_scores2_v1ce.parquet" ] || { step "stage2 apply --ce"; $PY -u stage2.py apply --tag v1 --ce >> "$ART/stage2_apply_v1.log" 2>&1; }
step "all done"
