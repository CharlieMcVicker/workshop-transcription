---
id: TASK-47
title: Add confidence ratings to batch and single inference scripts
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-01 21:24'
updated_date: '2026-07-01 21:25'
labels: []
dependencies: []
ordinal: 47000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract and display token confidence probabilities for greedy decoding and beam scores for KenLM LM decoding in both batch_inference.py and run_inference.py, saving results to the batch inference CSV.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Calculate mean token confidence for greedy decoding (excluding padding/blanks)
- [x] #2 Extract KenLM beam score for LM decoding
- [x] #3 Include 'greedy_confidence' and 'kenlm_score' columns in the batch_inference.py output CSV
- [x] #4 Print confidence ratings to console in run_inference.py
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement confidence calculation for greedy decoding in run_inference.py\n2. Extract beam score for KenLM decoding in run_inference.py\n3. Update batch_inference.py to calculate and record greedy confidence and KenLM score\n4. Write the results to CSV and print to console\n5. Verify correctness on a test run.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented model confidence scoring for greedy decoding (using Softmax and averaging over non-padding tokens) and KenLM decoder beam score reporting (using decode_beams to retrieve logit and combined scores) in both run_inference.py and batch_inference.py. Added new columns to the batch inference output CSV and verified functionality via test runs.
<!-- SECTION:FINAL_SUMMARY:END -->
