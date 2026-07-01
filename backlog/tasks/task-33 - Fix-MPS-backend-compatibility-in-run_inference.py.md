---
id: TASK-33
title: Fix MPS backend compatibility in run_inference.py
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 05:48'
updated_date: '2026-07-01 05:48'
labels: []
dependencies: []
ordinal: 33000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the NotImplementedError on Apple Silicon GPU by enabling the MPS CPU fallback environment variable or falling back to CPU device.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Set PYTORCH_ENABLE_MPS_FALLBACK=1 in run_inference.py or add device fallback logic
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Set PYTORCH_ENABLE_MPS_FALLBACK='1' in os.environ at the top of run_inference.py to allow PyTorch to fall back to CPU for unsupported MPS ops.\n2. In device selection, if MPS fails during actual run, fall back to CPU.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Set PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable at the top of run_inference.py and implemented a device-safety check that falls back to CPU if execution fails on MPS. Verified that it runs correctly on Apple Silicon.
<!-- SECTION:FINAL_SUMMARY:END -->
