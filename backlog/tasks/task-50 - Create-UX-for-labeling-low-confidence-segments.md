---
id: TASK-50
title: Create UX for labeling low-confidence segments
status: Done
assignee:
  - '@myself'
created_date: '2026-07-01 21:36'
updated_date: '2026-07-01 21:36'
labels: []
dependencies: []
ordinal: 50000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build a simple web interface to load batch_inference_results, sort segments by confidence, allow labeling, and export to a train CSV format.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Web interface created
- [x] #2 Can load batch inference results
- [x] #3 Sort segments by confidence
- [x] #4 Export labeled entries to CSV
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Active Labeler web app at active_labeler.py. The application serves a premium dark-themed single-page app sorting batch_inference_results.csv by confidence ascending, enabling audio playback, transcription correction, and export to train_labeled.csv.
<!-- SECTION:FINAL_SUMMARY:END -->
