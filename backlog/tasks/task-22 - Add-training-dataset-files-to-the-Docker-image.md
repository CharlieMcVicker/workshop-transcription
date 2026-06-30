---
id: TASK-22
title: Add training dataset files to the Docker image
status: To Do
assignee:
  - '@user'
created_date: '2026-06-30 23:37'
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
- [ ] #1 Copy sentence_audio/ and CSV files in the Dockerfile
- [ ] #2 Update default paths in trainer script or guide to align with internal container paths
<!-- AC:END -->
