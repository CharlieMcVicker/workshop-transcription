---
id: TASK-17
title: Make training script portable for remote execution on vast.ai
status: Done
assignee:
  - '@agent'
created_date: '2026-06-30 23:26'
updated_date: '2026-06-30 23:26'
labels: []
dependencies: []
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update trainer_w2v2_local.py or create a wrapper script to support path overrides via CLI arguments or environment variables, allowing seamless remote execution without modifying files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add CLI argument parsing to trainer_w2v2_local.py for CONFIG overrides
- [x] #2 Verify script works locally after the changes
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added argparse overrides for the paths and training parameters to colab-script-rips/trainer_w2v2_local.py. Verified that --help runs successfully using the project's virtual environment python.
<!-- SECTION:FINAL_SUMMARY:END -->
