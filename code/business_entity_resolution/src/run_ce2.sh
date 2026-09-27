#!/usr/bin/env bash
# Cross-encoder v2 (MiniLM-L12, all 11.9M training pairs), resumable like run_ce.sh: after a machine reset
# just run it again. Outputs carry the _l12 tag, so v1 files stay untouched.
#   bash run_ce2.sh            (from code/business_entity_resolution/src)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
ART=../../../artifacts
FEAT=${BER_FEATS:-$ART}
FEAT=${FEAT//\\//}
LOG=$ART/crossenc_l12.log
CE="$PY -u crossenc.py --base cross-encoder/ms-marco-MiniLM-L12-v2 --tag l12"
step() { echo "=== $1 $(date) ===" | tee -a "$LOG"; }

[ -f "$ART/crossenc_l12/model.safetensors" ] || { step "ce train"; $CE train --max-pairs 12000000 >> "$LOG" 2>&1; }
[ -f "$FEAT/ce_scores_val_l12.parquet" ] || { step "ce score val"; $CE score --split val >> "$LOG" 2>&1; }
[ -f "$FEAT/ce_scores_test_l12.parquet" ] || { step "ce score test"; $CE score --split test >> "$LOG" 2>&1; }
step "all done"
