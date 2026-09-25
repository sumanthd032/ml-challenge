# Submission & Experiment History

Leaderboard limit: 5 submissions/day, 25–27 Sep 2026. Each row links to a git commit so any
submission can be reproduced (`git checkout <commit>` + README steps).

## Experiments (local validation, macro F0.5)
| id | date | commit | change | blocking recall | val F0.5 | P | R | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-25 | 9a81dbc | v1 | _pending_ | test: 154.7M candidate pairs, 5.81M predicted, 3.36/S1 (val 3.32), 94.4% S1 with a match; validator PASS |
| v1 | 2026-09-25 | 9a81dbc | sparse IDF + bi-encoder kNN blocking, 74 features, LightGBM (656 iters), one-to-many + thr 0.7 | 0.9956 | 0.9764 | 0.9905 | 0.9524 | expected-F0.5 alpha 1.0: 0.9755; top features emb_rank_rev, cd_emb_cos_2nd_gap |

## Leaderboard submissions
| # | date | commit | experiment id | public F0.5 | notes |
|---|---|---|---|---|---|
