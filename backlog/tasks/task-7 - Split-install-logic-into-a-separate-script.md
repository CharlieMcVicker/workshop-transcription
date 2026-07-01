---
id: TASK-7
title: Split install logic into a separate script
status: Done
assignee:
  - '@myself'
created_date: '2026-06-30 21:30'
updated_date: '2026-06-30 21:30'
labels: []
dependencies: []
priority: high
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an setup/install script that installs pip requirements and builds dependencies like KenLM, keeping the main trainer script focused only on training.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create installation script install_requirements.sh
- [x] #2 Remove setup checks and build fallbacks from trainer_w2v2_local.py
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created install_requirements.sh and cleaned up trainer_w2v2_local.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created a separate setup script install_requirements.sh to install python dependencies and clone/build KenLM locally. Refactored trainer_w2v2_local.py to assume dependencies and KenLM (lmplz) are already installed or pre-built, printing a helpful pointer to the install script if missing.
<!-- SECTION:FINAL_SUMMARY:END -->
