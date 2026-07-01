---
id: TASK-59
title: Push latest codebase to new branch on cherokee-ASR remote
status: Done
assignee:
  - '@agent'
created_date: '2026-07-01 23:33'
updated_date: '2026-07-01 23:35'
labels: []
dependencies: []
ordinal: 59000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An earlier version of our code was pulled into a repo with unrelated history (git@github.com:lily-bel/cherokee-ASR.git). We need to create a branch on this repo with our latest code to set it as our new upstream.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configure git@github.com:lily-bel/cherokee-ASR.git as a remote
- [x] #2 Create a branch containing our latest code and push it to the new remote
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the git@github.com:lily-bel/cherokee-ASR.git remote as 'upstream', created a new branch 'charlie-latest' from local main, and pushed it to 'upstream/charlie-latest'.
<!-- SECTION:FINAL_SUMMARY:END -->
