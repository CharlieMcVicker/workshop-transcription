---
id: TASK-29
title: Evaluate checkpoint-800 on test dataset
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 02:11'
updated_date: '2026-07-01 05:28'
labels: []
dependencies: []
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run test inference on all test data using the Wav2Vec2 model checkpoint at remote_output_w2v2/checkpoint-800 and output the results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create test inference script evaluate_checkpoint.py
- [x] #2 Run inference on all test samples from cim-wav2vec2-test.csv using remote_output_w2v2/checkpoint-800
- [x] #3 Calculate and output CER and WER metrics for both greedy decoding and KenLM language model decoding
- [x] #4 Produce a summary of predictions compared to gold transcripts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create evaluate_checkpoint.py that loads test dataset, processor, KenLM model, and the specified checkpoint.\n2. Run inference on all test items and calculate metrics.\n3. Output a detailed report of the transcription comparisons and global metrics.\n4. Execute and present results to the user.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Evaluated remote_output_w2v2/checkpoint-800 on test split using evaluate_checkpoint.py. Verified that tone normalization is needed to map unicode diacritics predicted by the model back to numeric tone representation. Found Greedy WER = 81.49% (CER = 14.65%) and KenLM WER = 70.55% (CER = 11.24%).
<!-- SECTION:FINAL_SUMMARY:END -->
