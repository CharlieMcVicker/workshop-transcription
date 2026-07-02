---
id: TASK-72
title: Load inference model from best_model.json config
status: Done
assignee:
  - '@myself'
created_date: '2026-07-02 20:19'
updated_date: '2026-07-02 20:20'
labels: []
dependencies: []
ordinal: 68000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Read repo and revision from a root best_model.json file for all inference tasks
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create best_model.json at workspace root with repo charliemcvicker/asr-cherokee and revision 5464d15
- [x] #2 Update inference files to load model using parameters from best_model.json
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created best_model.json at the workspace root containing the Hugging Face repo charliemcvicker/asr-cherokee and revision 5464d15. Implemented a model_utils helper to load the configuration, and updated all relevant inference scripts (single, batch, run, run_julie) to load model and processor based on this config by default.
<!-- SECTION:FINAL_SUMMARY:END -->
