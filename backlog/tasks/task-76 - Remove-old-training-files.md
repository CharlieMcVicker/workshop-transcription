---
id: TASK-76
title: Remove old training files
status: Done
assignee:
  - '@myself'
created_date: '2026-07-02 20:30'
updated_date: '2026-07-02 20:33'
labels: []
dependencies: []
ordinal: 72000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove unused files in transcription/training since only transcription.training.train (train.py) is used for training.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify and remove unused files from transcription/training
- [x] #2 Verify that transcription.training.train is preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify server.py to call transcription.training.train instead of scripts/run_training.py\n2. Add comments to server.py /api/ checkpoints and transcription endpoints indicating that checkpoint paths will be messed up\n3. Update doc-2 (Vast.ai Training Guide) to use transcription.training.train\n4. Delete transcription/training/run_training.py, transcription/training/trainer.py, scripts/run_training.py, scripts/trainer_w2v2_local.py\n5. Verify that train.py can be executed correctly
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed unused training files scripts/run_training.py, scripts/trainer_w2v2_local.py, transcription/training/run_training.py, and transcription/training/trainer.py. Updated server.py to run the new transcription.training.train script entrypoint instead of scripts/run_training.py, and added warnings to the checkpoint endpoints indicating that checkpoint path resolution is out of sync.
<!-- SECTION:FINAL_SUMMARY:END -->
