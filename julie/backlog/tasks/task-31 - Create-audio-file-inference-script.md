---
id: TASK-31
title: Create audio file inference script
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 05:39'
updated_date: '2026-07-01 05:39'
labels: []
dependencies: []
ordinal: 31000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an inference script that takes a single audio file path, runs inference using checkpoint-800 and KenLM, and outputs transcriptions with and without tones.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create python script run_inference.py
- [x] #2 Load checkpoint-800 and KenLM decoder in script
- [x] #3 Process input audio file, resample to 16kHz if needed, and run inference
- [x] #4 Output transcription with numeric tone tags and with tone tags stripped
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write run_inference.py using argparse.\n2. Handle loading model, processor, and KenLM ARPA file.\n3. Read input audio file using torchaudio, resample if it is not 16000Hz, and run through the model.\n4. Apply tone normalization using replace_tones and strip_tones, then output results to terminal.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created run_inference.py which handles loading model checkpoint-800, processor/tokenizer, and KenLM ARPA model. It reads an audio file path, resamples to 16kHz mono, performs greedy + KenLM decoding, normalizes predicted tones to numeric format, and outputs predictions both with and without tones.
<!-- SECTION:FINAL_SUMMARY:END -->
