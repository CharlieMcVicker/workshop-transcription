---
id: TASK-57
title: Retrain model with word-final vowel tone 2 and prefix checkpoints
status: Done
assignee:
  - '@myself'
created_date: '2026-07-01 22:16'
updated_date: '2026-07-01 22:19'
labels: []
dependencies: []
ordinal: 57000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the model to append '2' to word-final vowels with no tone marking. Add 'with-final-tone-' prefix to training checkpoints. Regenerate label data and rebuild the docker image.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Append '2' to word-final vowels with no tone marking in transcription labeling
- [x] #2 Prefix training checkpoints with 'with-final-tone-'
- [x] #3 Regenerate label data
- [x] #4 Rebuild docker image
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated tone_normalization.py to append '2' to word-final vowels with no tone marking. Modified train.py to add 'with-final-tone-' prefix before run ID to Hugging Face Hub checkpoint uploads when push_to_hub is enabled. Regenerated the label split CSVs using prepare_csv.py, and rebuilt the docker image with tag charliemcvicker/asr:latest.
<!-- SECTION:FINAL_SUMMARY:END -->
