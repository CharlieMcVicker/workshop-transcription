---
id: TASK-4
title: Update trainer_w2v2.py to support local files
status: Done
assignee:
  - '@myself'
created_date: '2026-06-30 21:28'
updated_date: '2026-06-30 21:28'
labels: []
dependencies: []
priority: high
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the training script trainer_w2v2.py to handle local files and CSV splits instead of Colab/Drive/GCS dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Read and analyze trainer_w2v2.py assumptions
- [x] #2 Provide plan to tweak trainer_w2v2.py for local CSVs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created trainer_w2v2_local.py
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored the Colab trainer script into a standalone trainer_w2v2_local.py that supports running locally on CSV splits, local audio file folders, and allows custom configuration via argparse. Modified dependencies to use standard python subprocess/shutil modules instead of Google Drive mounts, ipywidgets, and shell bang constructs.
<!-- SECTION:FINAL_SUMMARY:END -->
