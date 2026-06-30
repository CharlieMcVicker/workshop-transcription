---
id: TASK-10
title: Document PyTorch source build requirements for MPS CTC loss
status: Done
assignee:
  - '@agent'
created_date: '2026-06-30 21:57'
updated_date: '2026-06-30 21:58'
labels: []
dependencies: []
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and document how to build PyTorch from source (main branch) with Apple Silicon (MPS) support for CTC loss.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research dependencies and steps required to build PyTorch from source on MacOS with MPS support
- [x] #2 Provide step-by-step instructions for building PyTorch from git repository
- [x] #3 Verify installation steps and list common pitfalls/troubleshooting
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Researched and documented the prerequisites, environment setup, cloning process, compilation flags, build process, and verification steps for compiling PyTorch from the git main branch on Apple Silicon to get MPS support for CTC loss. This has been saved to the guide document: backlog/docs/guides/pytorch-mps-build/doc-1 - PyTorch-MPS-CTC-Loss-Build-Guide.md
<!-- SECTION:FINAL_SUMMARY:END -->
