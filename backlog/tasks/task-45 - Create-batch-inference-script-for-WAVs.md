---
id: TASK-45
title: Create batch inference script for WAVs
status: Done
assignee: []
created_date: '2026-07-01 21:10'
updated_date: '2026-07-01 21:13'
labels: []
dependencies: []
ordinal: 45000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a Python script to perform fast batch inference (greedy/KenLM) on a list of WAV files using the standard model/revision 5464d15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create batch_inference.py script supporting --input (file containing WAV paths or directory), --checkpoint, --processor, --revision 5464d15, and decoding modes (greedy and KenLM)
- [x] #2 Allow batch-processing of multiple audio files in a single run
- [x] #3 Print/save results efficiently
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create batch_inference.py with argparse support for inputs, checkpoint, processor, and KenLM.\n2. Handle loading a list of wav files (either from a directory or from a txt/csv file line-by-line).\n3. Load model and processor using the specified revision 5464d15.\n4. Read audio using soundfile, convert/resample, batch process with feature extractor, perform model forward pass, and run greedy / KenLM decoding.\n5. Output transcription results to stdout and save to a CSV or JSON file if requested.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented batch_inference.py to process a directory of WAV files. It supports fast greedy decoding, optional KenLM decoding, tracking of loop times, average item speed, and ETA calculation. Evaluated on 704 segments in 110.59s (~0.16s/file), and results were saved in test_results.csv.
<!-- SECTION:FINAL_SUMMARY:END -->
