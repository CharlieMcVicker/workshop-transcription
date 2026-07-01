---
id: TASK-49
title: Optimize batch inference with GPU batching and parallel CPU decoding
status: Done
assignee:
  - '@antigravity'
created_date: '2026-07-01 21:27'
updated_date: '2026-07-01 21:27'
labels: []
dependencies: []
ordinal: 49000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement length bucketing, PyTorch padded batched inference on the GPU, and parallelized pyctcdecode/KenLM decoding on multiple CPU cores using multiprocessing in batch_inference.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add --batch-size and --num-workers command line arguments
- [x] #2 Sort/bucket audio files by length to form minimal-padded batches
- [x] #3 Implement GPU batched inference using collated feature representations
- [x] #4 Implement parallel CPU KenLM beam search decoding via multiprocessing Pool
- [x] #5 Validate correctness and ensure output results CSV remains fully compatible
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add --batch-size and --num-workers CLI options to batch_inference.py\n2. Rewrite audio loading logic to load and resample audios in memory or dynamically, sorting them by duration\n3. Construct padded batch inputs using Wav2Vec2Processor's pad / collate functionality\n4. Execute batched GPU forward passes to yield logit sequences\n5. Implement a multiprocessing decoding step where logit matrices are sent to a pool of CPU processes for CTC decoding and KenLM beam search\n6. Write results to CSV and print statistics\n7. Verify against sequential execution.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Optimized batch_inference.py using length sorting/bucketing to group audio files into batches with minimal padding. Implemented GPU batched inference using collated feature extractor tensors, and multiprocessed decoding (both greedy and KenLM beam search) using a multiprocessing Pool. Verified correct outputs and CSV alignment, then committed all changes.
<!-- SECTION:FINAL_SUMMARY:END -->
