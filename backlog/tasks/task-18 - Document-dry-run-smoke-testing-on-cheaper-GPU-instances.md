---
id: TASK-18
title: Document dry-run / smoke-testing on cheaper GPU instances
status: Done
assignee:
  - '@agent'
created_date: '2026-06-30 23:28'
updated_date: '2026-06-30 23:28'
labels: []
dependencies: []
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add documentation and advice for testing the trainer script on cheaper GPU instances with limit parameters to save costs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document cheap instance recommendations (e.g., RTX 3060, GTX 1080 Ti)
- [x] #2 Document using --max-steps to run a quick 5-10 step validation test
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented cost-saving testing strategies using cheaper GPU instances (RTX 3060, GTX 1080 Ti) combined with the --max-steps 10 parameter to perform end-to-end smoke testing before renting high-end GPUs.
<!-- SECTION:FINAL_SUMMARY:END -->
