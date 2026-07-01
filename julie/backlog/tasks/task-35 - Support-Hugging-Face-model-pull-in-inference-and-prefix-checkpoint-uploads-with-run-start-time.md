---
id: TASK-35
title: >-
  Support Hugging Face model pull in inference and prefix checkpoint uploads
  with run start time
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 17:29'
updated_date: '2026-07-01 17:29'
labels: []
dependencies: []
ordinal: 35000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
1. Add option to pull the Wav2Vec2 model and processor from Hugging Face Hub in run_inference.py.\n2. In trainer_w2v2_local.py, prefix the Hugging Face Hub model ID/checkpoint name with the run's start time (e.g. YYYYMMDD-HHMMSS) when saving and uploading checkpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modify run_inference.py to check if --checkpoint and --processor are Hugging Face repo IDs or local paths, and load accordingly from HF Hub
- [x] #2 Modify trainer_w2v2_local.py to prepend a run start time prefix (YYYYMMDD_HHMMSS) to the Hugging Face Hub model ID (hub_model_id) if push_to_hub is active
- [x] #3 Verify run_inference.py can download and run inference with a model/processor from Hugging Face Hub
- [x] #4 Verify trainer_w2v2_local.py correctly formats the hub_model_id with a start time prefix
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Hugging Face model pull in run_inference.py with optional token parameter and prefixing uploads with run start time in trainer_w2v2_local.py
<!-- SECTION:FINAL_SUMMARY:END -->
