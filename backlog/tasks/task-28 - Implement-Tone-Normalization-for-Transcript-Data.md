---
id: TASK-28
title: Implement Tone Normalization for Transcript Data
status: Done
assignee:
  - '@myself'
created_date: '2026-07-01 01:35'
updated_date: '2026-07-01 01:41'
labels: []
dependencies: []
ordinal: 28000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decompose accents and diacritics using Unicode NFD normalization, extract diacritic sequences occurring after non-word-final vowels, catalog them, and replace them with numerical identifiers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement Unicode NFD normalization to separate accents/diacritics
- [x] #2 Extract all unique sequences of diacritics following non-word-final vowels in the transcripts
- [x] #3 Generate a unique numerical mapping for each unique sequence
- [x] #4 Replace the diacritic sequences in the transcript with their mapped numbers
- [x] #5 Verify correctness of the mapping and replacement logic on the dataset
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented tone normalization utility module in tone_normalize.py and integrated it into local_csv_to_wav2vec2.py. Decomposed combining accents/diacritics using Unicode NFD. Mapped unique diacritic sequences after non-word-final vowels to integers 0-8 (with 0 representing unmarked vowels) and treated grave, circumflex, and double acute accents similarly with or without colons. Dropped 3 entries containing rare/single-occurrence marks. Generated updated train/validation/test splits.
<!-- SECTION:FINAL_SUMMARY:END -->
