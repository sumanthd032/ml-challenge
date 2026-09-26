# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A two-pass blocking stage (IDF-weighted sparse token retrieval + a fine-tuned MiniLM bi-encoder with exact GPU
kNN) feeds a LightGBM matcher over ~74 language-agnostic pair and competition features; a second LightGBM stage
re-scores pairs with within-entity context, two fine-tuned MiniLM cross-encoders, name-ambiguity, "twin" and
neighbourhood-density features, and a one-to-many decision with candidate-side competition picks the matches.
The key innovation for the unseen country (France) is **label-free test-time domain adaptation**: we found that the
English, India/US-trained bi-encoder and cross-encoder collapse French businesses together, and re-adapted them by
self-training on near-certain French matches with near-twin negatives (and rehearsal on the original labels), plus
France-specific normalization fixes. This lifted the public score from 0.9825 to 0.9848+.

---

## 2. Methodology

### 2.1 Problem Analysis
* Source 1 is deduplicated; each S2/S3 record belongs to at most one S1 (true for 100% of train labels). True
  matches per S1 follow the same distribution in every country ({0: 5.6%, 1: 5.5%, 2: 17%, 3: 24%, 4: 22%, ...},
  mean 3.46), and ~40% of test S2/S3 records have no S1 (distractors).
* Name noise: typos, accents, leetspeak, dropped/added/bracketed legal forms, generic words replaced
  ("Kittle Ventures" -> "Kittle Partners" is a true match 98.9% of the time on validation), acronyms, domain names,
  social handles, "aka/DBA" names, random brand names at the same address, native-script names (9 Indic scripts).
* Address noise: abbreviations, reordering, missing components, NULL literals, truncated / zero-padded /
  suffixed house numbers, city variants, states vs regions (France: departement instead of region).
* Hard negatives are "twins": another business with a near-identical name, often at a nearby address.
* France is absent from training. Its 259k S1s sit in only 15 cities, so near-identical names share streets
  (12.7% of France S1s share their exact core name with >= 10 S1s of the same city vs 0.7% India, 0% US).

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + cross-encoders, with test-time
self-training for the unseen country.  
**Core Innovation:** diagnosing the France failure as covariate shift of the embedding features (label-free:
exclusivity violations, feature-distribution shift on obviously-true pairs) and fixing it without labels or
external data (self-trained bi-encoder with hard in-batch negatives, cross-encoder self-training with near-twin
negatives and rehearsal, IDF on the training scale, France-specific normalization).

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** within country (country agrees for 100% of true pairs). Pass A: typed tokens (name tokens,
  4-char prefixes, concatenated name, address words, numbers), IDF-weighted (df <= 5000), L2-normalized sparse
  cosine, top-30 S1->S2/S3 and top-8 S2/S3->S1. Pass B: MiniLM-L6 bi-encoder (Apache-2.0) fine-tuned with
  in-batch-negative InfoNCE on train pairs of held-out ("ghost") S1s, exact GPU top-20 forward + top-5 reverse.
  For France the bi-encoder is re-adapted by self-training (Section 4).
- **Candidate pairs generated:** ~155M test pairs (union of both passes), ~89 per S1.
- **How you ensured true matches were not lost:** bidirectional retrieval (the reverse direction exploits "each
  record has one S1"), union of lexical and semantic passes; validation blocking recall 0.9956.

---

## 4. Matching Model

**Features used:**
- Name features: ratio / token-sort / token-set / partial / WRatio, Jaro-Winkler and Levenshtein on concatenated
  names, IDF-weighted token coverage and soft coverage, alternative (DBA) names, legal-form agreement/conflict,
  exact / core equality and name ambiguity (how many S1s share the name), "twin" edit features (legal-form relation,
  missing / extra / substituted name tokens, token order).
- Address features: token-set similarity, street-token overlap, house-number equality / edit distance / relative
  difference / prefix relation, state agreement, missing components.
- Other: blocking ranks and scores, bi-encoder cosine and ranks, competition features (rank / gap of a pair among
  all S1s of the same candidate and all candidates of the same S1), stage-2 context (best / sum / rank of the
  S1's stage-1 scores, similarity to its confident "sibling" records), two cross-encoder scores (MiniLM-L6 and -L12
  fine-tuned on 6M / 11.9M train pairs, with rank / gap), neighbourhood density (same-city same-name S1 counts, city
  agreement, same-street count) and discrete pair-type codes.

**Model type:** LightGBM stage 1 (trained on train S1s), LightGBM stage 2 (trained out-of-fold on validation S1s),
cross-encoders as stage-2 features.  
**Threshold selection method:** macro F0.5 on a validation set of disjoint S1s with 20% of train S1s removed
("ghosts") so their records act as distractors like in test; flat threshold 0.7 (India/US). Decision: each record
keeps only its best S1; candidate-side competition p_i <- o_i / (1 + sum_j o_j) with o = p / (1 - p) enforces the
at-most-one-S1 rule. France threshold 0.80-0.85 chosen from leaderboard evidence (France is not in validation).

**France (unseen country) adaptation, all label-free and without external data:**
1. Diagnosis: 77% of stage-1 gain comes from two bi-encoder features whose values on obviously-true French pairs were
   2.2 standard deviations away from India/US; a label-free "exclusivity excess" (probability mass above the
   one-S1 limit) showed France 10-35x more overconfident than India/US after the cross-encoders.
2. Bi-encoder self-training on 0.7-0.8M near-certain French pairs, batches sorted by (city, name) so in-batch
   negatives are near-twins; feature IDF on the training scale.
3. Cross-encoder self-training: positives as above, negatives (B, c) where B is one of the 5 nearest other S1s of
   c's owner A; rehearsal with 2M original labelled pairs keeps its India/US calibration (val logloss 0.067).
4. France normalization: dotted legal forms (S.A.R.L.), departements mapped to regions, no US/Indian 2-letter
   state fallback. Final France scores average two adapted variants (original vs self-trained cross-encoder).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation (India/US) 0.9897 (P 0.9975, R 0.9737); public leaderboard 0.984816
  (submission 8) and above with the final France pipeline (see docs/SUBMISSIONS.md).
- **Common false positives (wrong merges):** twins with a nearby house number and one swapped generic word in dense
  same-name families; exact-name records without an address when several S1s share the name.
- **Common false negatives (missed matches):** records with no address whose name is shared by several S1s (69% of
  validation misses have an empty address), random brand names, blocking misses (0.44% of true pairs).

---

## 6. Conclusion
Language-agnostic similarity and competition features with a context-aware second stage reach 0.99 on validation
for the training countries. The decisive lesson was on the unseen country: learned embeddings silently transfer
their training domain, and a few label-free diagnostics (exclusivity violations, feature shift on obvious pairs,
count distributions) located the failure; self-training with hard negatives and rehearsal fixed much of it
without any labels.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (README.md, requirements.txt, src/). Entry points, in order:
`learn_dict.py`, `preprocess.py`, `run_blocking.py`, `embed.py finetune|knn`, `train.py`, `predict.py`,
`crossenc.py pairs|train|score`, `stage2.py fit|apply` (--ce --ce2 l12 --x --dens), `run_ce.sh`, `run_ce2.sh`,
`run_s2_l12.sh`; France adaptation: `france_adapt.py` (renorm, pseudo, stage1, ce_pairs_nn, assemble, candidates)
with `embed.py finetune_pseudo`, `crossenc.py train_pseudo --anchor`, `stage2.py apply --split testfr3`.
Decision log: docs/DECISIONS.md (D-001..D-019); experiment and submission history: docs/SUBMISSIONS.md.

### B. Additional Results
Leaderboard history: 0.967016 (stage 1) -> 0.979034 (cross-encoder) -> 0.981132 (twin features) -> 0.982487
(candidate competition, France threshold) -> 0.984816 (France adaptation) -> final France pipeline.
