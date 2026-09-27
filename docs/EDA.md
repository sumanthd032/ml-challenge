# EDA Findings

Scripts and raw outputs: `eda/eda0*_*.py` and `eda/eda0*_output.txt`.

## Sizes
| split | S1 | S2 | S3 | S2+S3 per S1 |
|---|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 | 4.68 |
| test  | 1,732,544 | 4,887,273 | 5,082,316 | 5.76 |

Country mix: train S1 = 60% US / 40% India. Test S1 = 47% India / 38% US / **15% France (unseen)**.
The test has more S2/S3 records per S1 in *every* country (India 5.9 vs 4.7, US 5.8 vs 4.7).
The most likely cause is that more S2/S3 records have no S1 in the test set, i.e. more distractors.

## Ground truth structure
* Singletons (no match): **5.6%** of train S1.
* Matches per S1: mean ≈ 3.5, mode 3, max 11. S2 per S1 is 0-5 and S3 per S1 is 0-6.
* **Each S2/S3 id is matched to at most one S1** (7,638,365 matched ids, all unique).
* Only 73-75% of S2/S3 records are matched to any S1. The rest are distractors.
* Country of S1 and of its matches agree in **100%** of pairs, so blocking within country is safe.
* No leakage: row order and numeric id have ~0 correlation with matches.

## Name noise
* S1 is always Latin script. S2 has **9% non-Latin names** and S3 has 4.9% (Devanagari 5.2%/2.8%, then Telugu,
  Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi). These are *transliterations* of the English name,
  e.g. `वन टेक्नोलॉजीज प्रा. लि.` = "One Technologies Pvt Ltd". Mixed-script names also occur (`First साउथ Logistics`).
* Legal suffix noise: added, dropped, reordered or bracketed (`[LLC]`, `(Limited)`, `Pvt Ltd`, `Ltd Pvt`).
* Appended generic words: Center, Services, Partners, Trading, Group; honorifics Shri/Sri/Smt/Dr/Mr.
* Word shuffles (`Up Service Rev Cleaning [It]`), duplicated words, typos (`Chbmres`), leetspeak/homoglyphs
  (`H0using`, `Va1enzuela`, `S0ns`, `c0m`), random accents (`Phármaceuticals`).
* Website/handle forms: `4220firstave.Com`, `#vmuniholdings`, `@Extracommunications`, `digitalinfrabig.com`
  (token-shuffled). Also `| www.gildasph.com` suffixes, `#77599` and `(ID: 123)` codes, `<<`/`--`/`***` prefixes.
* Trade names: `X t/a Y`, `X dba Y`, `formerly`. Also **synthetic brand names** made from syllables
  (`Xyloecto`, `Lyraectoflux`, `Onyxdova`, `Fluxvio`) that share nothing with the S1 name. Only the address can link them.
* Name exact-match (normalized, legal removed) rate among true pairs: 44-55%. Latin name token Jaccard median 0.67
  and 5th percentile 0.

## Address noise
* S1 uses state abbreviations (US) / full state (India). S3 uses full US state names. S2 uses UPPERCASE and
  abbreviations. Indian state names also appear in native script (`महाराष्ट्र`, `ಕರ್ನಾಟಕ`).
* Component reordering (`ASHBURN, 0020718 ADAMS MILL PLACE, VA`), leading zeros, `##161`, `#00430`,
  `4809B`, `NULL`/`N/A` fragments, PMB/PO Box, `Door No`, `H.no`, `Near/Opp` landmarks.
* ~3% of S2/S3 addresses are empty. House numbers overlap in 88% (US) / 96% (India) of true pairs that have numbers.
* City variants: township/county names instead of city (`Portland Twonship` vs `Tigard`), old city names
  (`Bombay`), typos (`Cheaspeake`).

## Ambiguity
* **49% of S1 names are not unique** (same normalized name as another S1). Generic combinatorial names like
  "Summit Inc" or "Housing Program" are common, so the address is required for precision.
* 40% of unmatched (distractor) S2 names equal some S1 name (hard negatives).

## France (test only)
Legal forms SARL/SAS/SASU/EURL/SCI/EI/SA, street abbreviations R./RUE/AV/BD, `bis`, `N°`, apostrophes (`D'Arras`),
département instead of région (Nord, Gironde, Loire-Atlantique), and the same synthetic-name and leetspeak noise.
