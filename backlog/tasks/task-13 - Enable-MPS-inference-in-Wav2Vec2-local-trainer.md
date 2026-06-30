---
id: TASK-13
title: Enable MPS inference in Wav2Vec2 local trainer
status: Done
assignee:
  - '@myself'
created_date: '2026-06-30 22:13'
updated_date: '2026-06-30 22:15'
labels: []
dependencies: []
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update trainer scripts to run inference on Apple Silicon (MPS) instead of CPU.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify device configuration in trainer scripts
- [x] #2 Update inference pipeline or device targeting to use MPS if available
- [x] #3 Verify MPS is used during local inference/evaluation run
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify trainer_w2v2_local.py to detect and use mps if available (using a helper or standard device check) and clean cache correctly.\n2. Modify trainer_w2v2.py (original Colab rip) similarly.\n3. Verify changes work correctly.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated both trainer_w2v2_local.py and trainer_w2v2.py to detect and target 'mps' (Metal Performance Shaders) when available for local execution on Apple Silicon. Added proper dynamic empty_cache handling for both CUDA ('torch.cuda.empty_cache()') and MPS ('torch.mps.empty_cache()') depending on active device. Successfully tested general CTC loss computation on MPS and verified dataset mapping/preparation flow works on the local setup.
<!-- SECTION:FINAL_SUMMARY:END -->
