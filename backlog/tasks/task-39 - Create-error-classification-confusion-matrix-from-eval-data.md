---
id: TASK-39
title: Create error classification confusion matrix from eval data
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 18:15'
updated_date: '2026-07-01 18:16'
labels: []
dependencies: []
ordinal: 39000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a confusion matrix classifying the kinds of errors made on the evaluation/test data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify eval data file containing predicted and target transcriptions
- [x] #2 Write a script to analyze differences and categorize error types
- [x] #3 Generate a confusion matrix visualization/report of these errors
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Generated dynamic programming sequence alignments for greedy and KenLM evaluation results. Categorized errors by character categories (Tones, Vowels, Consonants, Glottals, Spaces, Insertions, Deletions). Generated grouped and character-level confusion matrices, and exported beautiful heatmap visualizations and a comprehensive markdown analysis report.
<!-- SECTION:FINAL_SUMMARY:END -->
