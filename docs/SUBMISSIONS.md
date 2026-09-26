# Submission & Experiment History

Leaderboard limit: 5 submissions/day, 25–27 Sep 2026. Each row links to a git commit so any
submission can be reproduced (`git checkout <commit>` + README steps).

## Experiments (local validation, macro F0.5)
| id | date | commit | change | blocking recall | val F0.5 | P | R | notes |
|---|---|---|---|---|---|---|---|---|
| v1 | 2026-09-25 | 9a81dbc | sparse IDF + bi-encoder kNN blocking, 74 features, LightGBM (656 iters), one-to-many + thr 0.7 | 0.9956 | 0.9764 | 0.9905 | 0.9524 | expected-F0.5 alpha 1.0: 0.9755; top features emb_rank_rev, cd_emb_cos_2nd_gap |
| v1-s2 | 2026-09-25 | 7d93b88 | v1 + stage-2 context re-scorer (lgb2_v1: S1 context + similarity to confident siblings), thr 0.7 | 0.9956 | 0.9782 | 0.9927 | 0.9551 | 2-fold OOF on val S1s (stage 1 on same folds 0.9764). test: 5.83M predicted, 3.36/S1, 94.1% S1 with a match; validator PASS; file output/variants/matching_results_stage2_thr70.tsv |
| v1-ce | 2026-09-25 | bdc1f5d | v1-s2 + cross-encoder (ms-marco-MiniLM-L-6 fine-tuned 1 epoch on 6M fit-S1 pairs, val logloss 0.070) score/rank/gap as stage-2 features, thr 0.7 | 0.9956 | 0.9879 | 0.9971 | 0.9691 | thr 0.6: 0.9874, 0.8: 0.9877. CE AUC on hard band (0.3<p<0.95) 0.94 vs stage-1 0.78; stage-2 gain importance ce 0.90. test: 5.85M predicted, 3.38/S1, 94.3% S1 with a match; validator PASS; file output/variants/matching_results_ce_thr70.tsv |
| v1-cex | 2026-09-26 | 50703de | v1-ce + stage-2 name-ambiguity (8) and twin (22) features (D-013), thr 0.7 | 0.9956 | 0.9889 | 0.9972 | 0.9725 | thr 0.6: 0.9888, 0.8: 0.9888. test: 5.86M predicted, 3.38/S1, 94.3% S1 with a match; validator PASS; file output/variants/matching_results_cex_thr70.tsv |
| v1-cel12x | 2026-09-26 | 2d94a38 | v1-cex + second cross-encoder (ms-marco-MiniLM-L12 fine-tuned 1 epoch on all 11.9M fit-S1 pairs, val logloss 0.065 vs 0.070 for L6) as `--ce2 l12`, thr 0.7 | 0.9956 | 0.9896 | 0.9975 | 0.9736 | thr 0.6: 0.9895, 0.8: 0.9894 (+0.0006-0.0007 over v1-cex at every thr). Stage-2 fold logloss 0.0418/0.0422 vs 0.0441/0.0446. Gain importance ce2 0.89, ce 0.05. test: 5.87M predicted, 3.39/S1, 94.3% S1 with a match; validator PASS; file output/variants/matching_results_cel12x_thr70.tsv |
| v1-cel12x-cexFR | 2026-09-26 | 07b680d | per-country blend (D-015): v1-cel12x for India/US S1s, v1-cex for France S1s (`blend_country.py`), no retraining | 0.9956 | 0.9896 | 0.9975 | 0.9736 | val = v1-cel12x (val has no France). Expected public ~0.9817 (#3 0.981132 + India/US gain ~0.0006 seen in #4). test: 5.86M predicted, 259,452 France rows from cex; validator PASS; file output/variants/matching_results_cel12x_cexFR_thr70.tsv |

## Leaderboard submissions
| # | date | commit | experiment id | public F0.5 | notes |
|---|---|---|---|---|---|
| 1 | 2026-09-25 | 9a81dbc | v1 | 0.967016 | val 0.9764 (gap -0.0094). test: 154.7M candidate pairs, 5.81M predicted, 3.36/S1, 94.4% S1 with a match |
| 2 | 2026-09-25 | bdc1f5d | v1-ce | 0.979034 | val 0.9879 (gap -0.0089): val gains transfer ~1:1 |
| 3 | 2026-09-26 | 50703de | v1-cex | 0.981132 | val 0.9889 (gap -0.0078). Top of leaderboard 0.99056. If India/US ~ val, France (15%) ~0.935 |
| 4 | 2026-09-26 | 2d94a38 | v1-cel12x | 0.981101 | val 0.9896 (+0.0007) but public -0.000031 vs #3. Train/val have no France S1s (test: 15%), so val can't see France. Val gain holds for India +0.0008 and US +0.0006, also on candidates the CE never saw in training (no memorization). On test the new model adds +0.6% pairs in France vs +0.08% India, -0.01% US; with ~+0.0006 from India/US, France must have lost ~0.004 |
