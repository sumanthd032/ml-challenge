# Methodology Research — Business Entity Resolution

This note surveys entity-resolution (ER) methodology and maps each idea onto *this* dataset
(see `docs/EDA.md`). It ends with the chosen design and the reasons for it.

## 1. The problem in ER terms

* **Setting:** record linkage, one clean reference source (S1) vs two dirty sources (S2, S3).
  Multiple S2/S3 records can link to one S1 (S2/S3 contain internal duplicates).
* **Hard constraint found in EDA:** every S2/S3 record links to **at most one** S1
  (7,638,365 matched ids, all unique). This makes the decision a *one-to-many assignment*:
  each S2/S3 record either goes to its single best S1 or to nobody.
* **Metric:** macro F0.5 per S1 entity, singletons included (5.6% of train S1s).
  Per entity: `F0.5 = 1.25·TP / (0.25·n_true + n_pred)` (with `empty/empty = 1`).
  A false merge costs 4× more in the denominator than a missed match.
* **Scale:** test = 1.73M S1 × 9.97M S2+S3 → 1.7e13 raw pairs. Blocking is mandatory.

## 2. Classic pipeline (Christen 2012; Papadakis et al. 2020 survey)

`normalize → block (candidate generation) → compare (features) → classify → cluster/assign`.

### 2.1 Normalization / standardization
Canonicalize abbreviations (Rd/Road, Pvt/Private), strip legal suffixes into a separate field,
unify case/accents/punctuation, fix OCR-like noise (0↔o, 1↔l), parse numbers, transliterate
non-Latin scripts. Rules are domain knowledge, not external data, so they are allowed.

### 2.2 Blocking families
| Family | Idea | Fit here |
|---|---|---|
| Standard key blocking | exact key (e.g. PIN, normalized name) | High precision, low recall under typos; good as a cheap extra pass |
| Sorted neighbourhood | sort by key, window | Weak with token shuffles |
| Token / q-gram inverted index (meta-blocking) | records sharing rare tokens | Good for names; must prune frequent tokens |
| TF-IDF top-k (sparse cosine) | kNN in sparse char/word space | Strong, but char-trigram posting lists are huge at 10M scale |
| MinHash-LSH | Jaccard approximation | Workable but tuning-heavy; top-k kNN is simpler |
| **Dense embedding kNN** (DeepBlocker, Thirumuruganathan et al. 2021; Sparkly 2023) | bi-encoder + ANN | Handles typos, shuffles and **cross-script** names; GPU brute-force top-k is exact and fast |

Findings from the literature that matter: (i) a union of complementary blockers beats any single
one; (ii) *self-supervised / fine-tuned* bi-encoders beat off-the-shelf ones for ER blocking;
(iii) retrieval in **both directions** (S1→S2/3 top-k, and S2/3→S1 top-k) raises recall cheaply.

### 2.3 Matching
| Approach | Examples | Notes |
|---|---|---|
| Probabilistic | Fellegi–Sunter, Splink | Interpretable, but assumes conditional independence |
| Feature-based ML | Magellan, GBDT over similarity features | Strong, fast, easy to calibrate; standard in industry ER |
| Deep ER | DeepMatcher (2018), Ditto (2020, fine-tuned RoBERTa cross-encoder), HierGAT | Best accuracy on benchmarks, but cost is O(#pairs) transformer passes |
| LLM matchers | GPT-style prompting | Too slow for ~40M pairs; license/size limits (≤8B, MIT/Apache) |

Consensus from benchmark papers (e.g. Mudgal et al. 2018; Li et al. 2020): GBDT on good
similarity features is close to deep models on *structured, short* records like ours. Deep
cross-encoders win mainly on *dirty/textual* data. A **hybrid** works best: GBDT on hand-made
features plus the cosine similarity from a fine-tuned bi-encoder as one more feature.

### 2.4 Similarity features that work for names and addresses
* Edit-based: Levenshtein ratio, Jaro-Winkler, Damerau (typos, leetspeak).
* Token-based: Jaccard, token-set / token-sort ratio (word shuffles), Monge-Elkan.
* Weighted: soft TF-IDF / IDF-weighted overlap (rare tokens matter more than "Private").
* Phonetic: Metaphone/Soundex keys (transliteration variance).
* Structural: house-number equality, postal code equality, state equality, empty-field flags.
* Relational / collective: rank of the pair in both directions and the margin to the best
  competing S1. This is powerful under the one-to-many constraint.

### 2.5 Cross-script names (Hindi, Telugu, Gujarati, …)
Options: (a) rule transliteration (unidecode / ITRANS), (b) a **dictionary learned from the
training pairs** (align native tokens with English tokens of the matched S1 name), (c) multilingual
embeddings (LaBSE, multilingual-MiniLM, mE5) fine-tuned on the pairs. (b) is precise for tokens
seen in train, (a)/(c) cover unseen tokens. We use b+a in the string features and c in the
embedding.

### 2.6 Decision under F0.5 (macro, per entity)
Global thresholding is suboptimal for a per-entity macro metric. With calibrated probabilities
`p_i` for an entity's candidates, the expected-F-optimal set is a top-k by probability
(Jansche 2007; Nan et al. 2012 "optimizing F-measure"). We use the plug-in approximation
`E[F] ≈ 1.25·Σ_{top-k} p / (0.25·Σ_all p + k)`, and for `k=0`, `E[F] ≈ Π(1−p_i)`
(the probability that the entity is a true singleton). Pick the k that maximizes it.
This handles singletons in a principled way.

## 3. Distribution shift to plan for
* **France appears only in test.** No country one-hot or country-specific model. Features are
  language-agnostic similarities. French legal forms (SARL/SAS/EURL/SCI/EI/SA) and street
  abbreviations (R./BD/AV/N°/bis) are added to normalization rules (domain knowledge only).
* **More S2/S3 per S1 in test (5.76 vs 4.68).** That means likely more distractors whose true
  S1 is absent. Validation mimics this by *removing ~20% of S1s from the pool* while keeping
  their S2/S3 records as distractors.

## 4. Chosen design (summary; details in ARCHITECTURE.md)
1. Normalization with learned native-script→Latin token dictionary + rule fallback.
2. Blocking = union of (a) fine-tuned multilingual bi-encoder kNN (both directions),
   (b) rare-token name/address inverted index, (c) exact keys (concatenated name, house number + street).
3. LightGBM over ~50 pair features incl. bi-encoder cosine and competition/rank features.
4. One-to-many assignment (each S2/S3 → at most its best S1) + expected-F0.5 top-k decision per S1.
5. Optional: small cross-encoder re-ranker on the uncertain band only.

## References
* P. Christen, *Data Matching*, Springer 2012.
* G. Papadakis et al., "Blocking and Filtering Techniques for Entity Resolution: A Survey", ACM CSUR 2020.
* S. Mudgal et al., "Deep Learning for Entity Matching: A Design Space Exploration" (DeepMatcher), SIGMOD 2018.
* Y. Li et al., "Deep Entity Matching with Pre-Trained Language Models" (Ditto), VLDB 2020.
* S. Thirumuruganathan et al., "Deep Learning for Blocking in Entity Matching" (DeepBlocker), VLDB 2021.
* D. Paulsen et al., "Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching", VLDB 2023.
* I. Fellegi, A. Sunter, "A Theory for Record Linkage", JASA 1969.
* M. Jansche, "A Maximum Expected Utility Framework for Binary Sequence Labeling", ACL 2007.
* Y. Nan et al., "Optimizing F-measure: A Tale of Two Approaches", ICML 2012.
* F. Feng et al., "Language-agnostic BERT Sentence Embedding" (LaBSE), ACL 2022.
