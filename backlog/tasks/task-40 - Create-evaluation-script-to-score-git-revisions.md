---
id: TASK-40
title: Create evaluation script to score git revisions
status: Done
assignee:
  - '@myself'
created_date: '2026-07-01 18:27'
updated_date: '2026-07-01 18:48'
labels: []
dependencies: []
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a script that checks out different git revisions from revisions_to_test.csv, runs evaluation, and generates a scoring CSV with greedy/kenlm tone and notone accuracy on test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement a Python or bash script to checkout each git revision and run the evaluation
- [x] #2 Score greedy/kenlm tone and notone accuracy
- [x] #3 Output results to a scoring CSV file
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented evaluate_all_revisions.py which parses revisions_to_test.tsv and evaluates all 11 model revisions downloaded from Hugging Face Hub (charliemcvicker/asr-cherokee). Extracted WER/CER metrics for both greedy and KenLM decoding systems under both tone and no-tone (masked) configurations, saving results to revision_scores.csv.
<!-- SECTION:FINAL_SUMMARY:END -->
