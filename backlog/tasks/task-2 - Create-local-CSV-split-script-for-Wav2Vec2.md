---
id: TASK-2
title: Create local CSV split script for Wav2Vec2
status: Done
assignee:
  - '@agent'
created_date: '2026-06-30 21:24'
updated_date: '2026-06-30 21:25'
labels: []
dependencies: []
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a Python script that reads the local sentence_audio.csv and sentence_audio/ WAV files, processes and cleans the transcripts, filters by duration, shuffles, and generates train/valid/test CSV splits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create python script local_csv_to_wav2vec2.py at the project root
- [x] #2 Implement local CSV parsing, metadata extraction (duration & filesize), transcript cleanup, filtering, and split generation
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Modified local_csv_to_wav2vec2.py to use phonetic transcription by default instead of syllabary. Regenerated train, validation, and test CSV splits using the phonetic data.
<!-- SECTION:FINAL_SUMMARY:END -->
