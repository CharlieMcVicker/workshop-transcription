---
id: TASK-32
title: Fix audio loading backend in run_inference.py
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 05:42'
updated_date: '2026-07-01 05:42'
labels: []
dependencies: []
ordinal: 32000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change the audio loading mechanism in run_inference.py from torchaudio to soundfile to match the environment's capabilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Replace torchaudio.load with soundfile.read in run_inference.py
- [x] #2 Test the inference script with an audio file
<!-- AC:END -->



## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace torchaudio import with soundfile (or librosa) in run_inference.py.\n2. Use soundfile.read to read the audio file.\n3. Adjust waveform formatting to match the structure expected by the resampling and model (specifically, torch tensor of shape [channel, length], and resampling using torchaudio or numpy/scipy).
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced torchaudio.load with soundfile.read in run_inference.py to bypass the missing torchcodec import error. Verified and tested successfully on MPS backend using a sample audio file.
<!-- SECTION:FINAL_SUMMARY:END -->
