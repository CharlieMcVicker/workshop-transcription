---
id: TASK-6
title: Add fallback compile routine for KenLM lmplz in local trainer
status: Done
assignee:
  - '@myself'
created_date: '2026-06-30 21:30'
updated_date: '2026-06-30 21:30'
labels: []
dependencies: []
priority: high
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update trainer_w2v2_local.py to automatically compile and build lmplz locally if it is not found on the system path, similar to how it did in Colab.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Detect if lmplz is missing
- [x] #2 Implement automatic local compilation step for lmplz in the workspace
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added KenLM build fallback inside output directory.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Modified trainer_w2v2_local.py to detect if lmplz is missing from the system path. If it is missing, the script will automatically clone KenLM to {output_dir}/kenlm and compile it using cmake and make so that training does not fail.
<!-- SECTION:FINAL_SUMMARY:END -->
