---
id: TASK-56
title: Update README with package entrypoints and usage guides
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 21:57'
updated_date: '2026-07-01 21:58'
labels: []
dependencies: []
ordinal: 56000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the project README.md to describe the new src/transcription directory layout, how to run different scripts (single/batch inference, training, active labeling server, etc.) via python module entrypoints (python -m), and Docker execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Draft README updates detailing package layout
- [x] #2 Provide examples for python -m invocations
- [x] #3 Provide examples for Docker running
- [x] #4 Commit changes to Git
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated README.md to describe the new folder structure, packaged module entrypoint invocations (e.g., python3 -m transcription.inference.single), and Docker execution instructions. Staged and committed changes to Git.
<!-- SECTION:FINAL_SUMMARY:END -->
