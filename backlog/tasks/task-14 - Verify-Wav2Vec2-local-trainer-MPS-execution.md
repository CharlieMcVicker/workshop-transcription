---
id: TASK-14
title: Verify Wav2Vec2 local trainer MPS execution
status: Done
assignee:
  - '@myself'
created_date: '2026-06-30 22:16'
updated_date: '2026-06-30 22:33'
labels: []
dependencies: []
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rerun a short proof-of-concept training and inference run on MPS to measure and verify speed improvements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run trainer_w2v2_local.py with minimal epochs or short run
- [x] #2 Confirm MPS backend is utilized during training/evaluation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify CONFIG in trainer_w2v2_local.py to add a 'max_steps' option.\n2. Pass 'max_steps' to TrainingArguments in trainer_w2v2_local.py.\n3. Run a short 3-step training and evaluation run using 'max_steps=3' and verify it runs on MPS.\n4. Revert/reset 'max_steps' back to default -1 after verification.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Successfully added 'max_steps' CONFIG option and passed it to TrainingArguments. Set max_steps=3 temporarily, ran proof-of-concept training and evaluation runs on MPS, confirmed MPS utilization, and reverted max_steps back to -1.
<!-- SECTION:FINAL_SUMMARY:END -->
