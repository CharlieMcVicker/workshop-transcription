---
id: TASK-73
title: Fix KenLM test leak and evaluate local checkpoints with masked metrics
status: Done
assignee:
  - '@agent'
created_date: '2026-07-02 20:25'
updated_date: '2026-07-02 20:26'
labels: []
dependencies: []
ordinal: 69000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the post-training model selection KenLM test data leakage where the test set was leaked into the KenLM training corpus. Implement local evaluation script/tooling to score all local checkpoints on the GPU machine with both full and masked CER/WER to avoid slow Hugging Face revision downloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fix KenLM corpus construction in train.py to exclude validation and test dataset transcriptions
- [x] #2 Create an offline script or update evaluate_revisions.py to evaluate all local checkpoints in the output directory without Hugging Face downloads
- [x] #3 Support scoring both full and masked CER/WER (greedy and optionally KenLM if ARPA is available)
- [x] #4 Verify the execution of the evaluation tool locally
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Investigated post-training model selection KenLM test data leakage. Fixed the leak by only using training splits for KenLM training corpus construction in training scripts (train.py, run_training.py, trainer_w2v2_local.py, trainer.py). Created a local checkpoint evaluation script (transcription/training/evaluate_local_checkpoints.py) that loads all checkpoints locally from a folder, evaluates them offline, support full and masked CER/WER, and outputs a ranked CSV.
<!-- SECTION:FINAL_SUMMARY:END -->
