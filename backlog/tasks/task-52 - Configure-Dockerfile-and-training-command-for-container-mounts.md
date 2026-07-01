---
id: TASK-52
title: Configure Dockerfile and training command for container mounts
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 21:52'
updated_date: '2026-07-01 21:53'
labels: []
dependencies: []
ordinal: 52000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update Dockerfile to reflect new data/ and src/ structures, move trainer_w2v2_local.py into src/transcription/training/train.py, and configure Docker CMD for automated training.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Move trainer_w2v2_local.py to src/transcription/training/train.py
- [x] #2 Update default CSV and audio paths in train.py to point to data/processed/
- [x] #3 Update Dockerfile to copy src/ and data/ and set CMD for training
- [x] #4 Verify Dockerfile builds
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved trainer_w2v2_local.py to src/transcription/training/train.py, updated default configuration paths to use the new data/ directory structure, and updated the Dockerfile to copy src/ and data/ directories. Set the container's default command to start training with 'python3 -m transcription.training.train' automatically when launched/mounted.
<!-- SECTION:FINAL_SUMMARY:END -->
