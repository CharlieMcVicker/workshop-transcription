---
id: TASK-34
title: Use t/th system instead of t/d system for voicing/aspiration
status: Done
assignee: []
created_date: '2026-07-01 05:51'
updated_date: '2026-07-01 20:52'
labels: []
dependencies: []
ordinal: 34000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update transcription normalization/processing to use the t/th voicing/aspiration system instead of the t/d voicing/aspiration system.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify where voicing/aspiration mapping is defined
- [x] #2 Update mapping/cleaning scripts to convert/represent voicing/aspiration using t/th instead of t/d
- [x] #3 Regenerate train/valid/test datasets with the updated t/th voicing/aspiration representation
- [x] #4 Verify that generated CSV datasets are correctly updated
<!-- AC:END -->
