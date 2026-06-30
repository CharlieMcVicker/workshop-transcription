---
id: TASK-5
title: Hardcode local data path dict in trainer_w2v2_local.py
status: Done
assignee:
  - '@myself'
created_date: '2026-06-30 21:29'
updated_date: '2026-06-30 21:30'
labels: []
dependencies: []
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the argparse setup in trainer_w2v2_local.py with a hardcoded config dict pointing to the actual local data files on disk.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Find the generated train/test/validation CSV paths on disk
- [x] #2 Replace argparse in trainer_w2v2_local.py with a config dict
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Hardcoded config dict in trainer_w2v2_local.py with CSV paths: cim-wav2vec2-train.csv, cim-wav2vec2-valid.csv, cim-wav2vec2-test.csv and audio dir sentence_audio.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the command line parsing setup in trainer_w2v2_local.py with a CONFIG dict containing local CSV/audio file paths dynamically generated in the workspace: cim-wav2vec2-train.csv, cim-wav2vec2-valid.csv, cim-wav2vec2-test.csv, and sentence_audio/.
<!-- SECTION:FINAL_SUMMARY:END -->
