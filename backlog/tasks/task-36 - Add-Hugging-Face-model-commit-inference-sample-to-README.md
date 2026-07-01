---
id: TASK-36
title: Add Hugging Face model commit inference sample to README
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 17:38'
updated_date: '2026-07-01 17:39'
labels: []
dependencies: []
ordinal: 36000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the project README.md to include a sample command illustrating how to run inference using a specific commit hash (revision) of a model hosted on Hugging Face Hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify where run_inference.py is documented in README.md
- [x] #2 Add documentation/instructions and a sample command to README.md showcasing how to pass a specific Hugging Face model revision/commit using Hugging Face's revision parameter syntax
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated run_inference.py to support a --revision argument, and documented local, Hugging Face Hub, and specific commit/revision inference commands in README.md.
<!-- SECTION:FINAL_SUMMARY:END -->
