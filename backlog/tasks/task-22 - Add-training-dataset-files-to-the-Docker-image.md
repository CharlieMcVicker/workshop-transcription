---
id: TASK-22
title: Add training dataset files to the Docker image
status: Done
assignee:
  - '@agent'
created_date: '2026-06-30 23:37'
updated_date: '2026-07-01 00:05'
labels: []
dependencies: []
ordinal: 22000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the Dockerfile to COPY the sentence_audio directory and prepared CSV splits directly into the image, making the container completely self-contained for training runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Copy sentence_audio/ and CSV files in the Dockerfile
- [x] #2 Update default paths in trainer script or guide to align with internal container paths
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added CSVs and audio COPY commands to Dockerfile. The trainer script, trainer_w2v2_local.py, uses configurable command line arguments (e.g. --train-csv, --valid-csv, --test-csv, --audio-dir) which default to the local paths but can be overridden when executing inside the container to align with internal container paths /workspace/.*
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated Dockerfile to COPY prepared dataset CSVs and audio files directly to /workspace. Checked default configuration in local training script trainer_w2v2_local.py, which supports overriding all paths dynamically via arguments (e.g., --train-csv /workspace/cim-wav2vec2-train.csv).
<!-- SECTION:FINAL_SUMMARY:END -->
