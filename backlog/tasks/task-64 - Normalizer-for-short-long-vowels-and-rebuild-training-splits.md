---
id: TASK-64
title: Normalizer for short/long vowels and rebuild training splits
status: Done
assignee:
  - '@myself'
created_date: '2026-07-02 15:44'
updated_date: '2026-07-02 15:45'
labels: []
dependencies: []
ordinal: 64000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a utility in tone normalization that removes tone markings from short vowels (no colons) and doubles vowels before a colon, then rebuild training data source of truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement short vowel tone stripping and long vowel doubling in tone_normalization.py
- [x] #2 Update prepare_csv.py to use this new normalization utility
- [x] #3 Rebuild training data splits (train, valid, test CSVs)
- [x] #4 Verify splits are correctly generated and contain the normalized text
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented remove_tones_and_double_vowels in transcription/utils/tone_normalization.py to strip tone/combining accents from short vowels and double long vowels (those before colons, while removing the colons). Updated transcription/training/prepare_csv.py to use this utility and pre-strip formatting asterisks so they do not interfere with vowel-colon identification. Successfully rebuilt training data splits (train, valid, and test CSVs).
<!-- SECTION:FINAL_SUMMARY:END -->
