---
id: TASK-15
title: Batch GPU inference and multiprocess KenLM decoding in local trainer
status: Done
assignee:
  - '@agent'
created_date: '2026-06-30 22:29'
updated_date: '2026-06-30 23:21'
labels: []
dependencies: []
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refactor the evaluation loops in trainer scripts to batch forward passes on GPU and use pyctcdecode's decode_batch with multiprocessing to parallelize KenLM decoding across CPU cores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refactor inference helper to accept batches of inputs
- [x] #2 Accumulate logits from batch inference on GPU
- [x] #3 Replace sequential decoding with processor_with_lm.decode_batch using multiprocessing
- [x] #4 Verify performance speedup during evaluation
- [x] #5 Add a tqdm progress bar for the decoding step
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add 'eval_batch_size': 16 to CONFIG in trainer_w2v2_local.py.\n2. Create a DataLoader for test_ds_prepared using DataCollatorCTCWithPadding and the configured batch size.\n3. Implement batch inference on the active device to compute and accumulate logits.\n4. Initialize a multiprocessing Pool using the 'fork' context.\n5. Chunk the logits and perform parallelized KenLM decoding via processor_with_lm.decoder.decode_batch with the pool, displaying progress via tqdm.\n6. Compute final metrics and save the results.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refactored evaluation loop in trainer_w2v2_local.py to perform batch GPU/MPS inference using a PyTorch DataLoader and DataCollatorCTCWithPadding. Also implemented parallelized KenLM decoding via processor_with_lm.decoder.decode_batch using a multiprocessing pool with fork context. Added tqdm progress bars for both inference and decoding, and verified performance.
<!-- SECTION:FINAL_SUMMARY:END -->
