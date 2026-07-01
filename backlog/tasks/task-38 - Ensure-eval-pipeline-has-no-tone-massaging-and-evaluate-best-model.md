---
id: TASK-38
title: Ensure eval pipeline has no tone massaging and evaluate best model
status: Done
assignee: []
created_date: '2026-07-01 18:03'
updated_date: '2026-07-01 18:04'
labels: []
dependencies: []
ordinal: 38000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove any tone / transcription massaging from the evaluation pipeline. Recompute accuracy with and without tone for model charliemcvicker/asr 5464d155c9cbaa04a58c74a66b298d89ba56a290
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove tone replacement or respelling logic from evaluation code
- [x] #2 Run evaluation for model charliemcvicker/asr with and without tone
- [x] #3 Document evaluation results in a clear format/artifact
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed tone replacement and transcription massaging from evaluate_checkpoint.py, and added argparse options to fetch HF models. Recomputed accuracy metrics for charliemcvicker/asr-cherokee (commit 5464d155c9cbaa04a58c74a66b298d89ba56a290) both with and without tone, and documented the results in the artifacts directory.
<!-- SECTION:FINAL_SUMMARY:END -->
