---
id: TASK-37
title: Remove tone replacement and respelling from run_inference.py
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 17:52'
updated_date: '2026-07-01 17:52'
labels: []
dependencies: []
ordinal: 37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the logic that imports and calls replace_tones and strip_tones from run_inference.py, as the model now outputs proper surface forms directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remove tone_normalize and replace_tones imports from run_inference.py
- [x] #2 Remove strip_tones and replace_tones calls in both greedy and KenLM decoding logic
- [x] #3 Update stdout prints to display the direct decoded outputs without respelling / tone formatting
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the tone_normalize module imports, strip_tones function, and all associated respelling / tone formatting logic from run_inference.py to output direct surface forms from the model.
<!-- SECTION:FINAL_SUMMARY:END -->
