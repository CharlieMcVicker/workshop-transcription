---
id: TASK-19
title: Create Dockerfile to prebake training dependencies
status: Done
assignee:
  - '@agent'
created_date: '2026-06-30 23:29'
updated_date: '2026-06-30 23:29'
labels: []
dependencies: []
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a Dockerfile based on PyTorch CUDA that pre-installs all required ASR libraries and compiles KenLM binaries, ensuring vast.ai runs start instantly without installation overhead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create Dockerfile with PyTorch CUDA base image
- [x] #2 Install build dependencies, compile KenLM, and install python dependencies
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created a Dockerfile in the project root based on 'pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel'. It compiles KenLM binaries (adding 'lmplz' to the PATH) and installs all required python dependencies, ensuring that remote runs can launch instantly without installation overhead.
<!-- SECTION:FINAL_SUMMARY:END -->
