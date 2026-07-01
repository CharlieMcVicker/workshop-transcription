---
id: TASK-30
title: Evaluate checkpoint-800 test accuracy with masked tones
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 05:29'
updated_date: '2026-07-01 05:30'
labels: []
dependencies: []
ordinal: 30000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Calculate and output CER and WER metrics for both greedy decoding and KenLM decoding on the test set when tones are masked (removed).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Update evaluate_checkpoint.py to support stripping numeric tone tags from both gold transcripts and normalized predictions
- [x] #2 Calculate WER and CER metrics on the tone-masked transcripts
- [x] #3 Output the tone-masked accuracy results
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Modify evaluate_checkpoint.py to support a mode or calculation where we strip numeric tone characters (0-9) from both predictions and gold references.\n2. Compute the tone-masked WER and CER.\n3. Output the metrics to terminal.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Measured test accuracy on checkpoint-800 with tones masked (digits stripped). Without tones, Greedy WER is 23.47% (CER = 3.96%) and KenLM WER is 8.89% (CER = 2.24%). This shows the underlying phonetic transcription is extremely accurate (sub-9% WER with KenLM), and tone prediction is the main source of error.
<!-- SECTION:FINAL_SUMMARY:END -->
