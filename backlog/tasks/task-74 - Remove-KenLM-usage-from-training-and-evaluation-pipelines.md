---
id: TASK-74
title: Remove KenLM usage from training and evaluation pipelines
status: Done
assignee:
  - '@agent'
created_date: '2026-07-02 20:27'
updated_date: '2026-07-02 20:29'
labels: []
dependencies: []
ordinal: 70000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove all dependencies, training, decoding, and scoring steps involving KenLM from the training pipeline and evaluation scripts, and rank checkpoints based on greedy evaluation metrics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify train.py to rank/promote checkpoints based on greedy WER/CER instead of KenLM
- [x] #2 Remove KenLM build/decoding steps from train.py, trainer_w2v2_local.py, run_training.py, and trainer.py
- [x] #3 Update evaluate_local_checkpoints.py to not load or decode with KenLM at all
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed KenLM and pyctcdecode model building, decoding, evaluation, and ranking logic from all training scripts (train.py, trainer_w2v2_local.py, run_training.py, and trainer.py). Checkpoints are now ranked and promoted purely using greedy evaluation metrics. Cleaned evaluate_local_checkpoints.py to completely remove KenLM and pyctcdecode dependencies, scoring only standard and masked greedy WER/CER.
<!-- SECTION:FINAL_SUMMARY:END -->
