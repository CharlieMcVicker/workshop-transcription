#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_all_revisions.py

Checks out each model revision from revisions_to_test.csv from Hugging Face Hub,
runs evaluation on the test dataset, and saves a scoring CSV.
"""

import os
import re
import unicodedata
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from datasets import Dataset, Audio
from pyctcdecode import build_ctcdecoder
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from tqdm import tqdm

# Import helper functions from evaluate_checkpoint if possible, otherwise redefine
from evaluate_checkpoint import (
    normalize_text,
    strip_tones,
    _try_read_csv,
    _detect_columns,
    _resolve_audio_path,
    TARGET_SAMPLE_RATE
)

def safe(s):
    return s if s.strip() else " "

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple model revisions on a test dataset.")
    parser.add_argument("--revisions-csv", type=str, default="revisions_to_test.tsv", help="Path to TSV or CSV listing revisions.")
    parser.add_argument("--test-csv", type=str, default="cim-wav2vec2-test.csv", help="Path to the test CSV file.")
    parser.add_argument("--audio-dir", type=str, default="sentence_audio", help="Directory containing audio files.")
    parser.add_argument("--checkpoint", type=str, default="charliemcvicker/asr-cherokee", help="Hugging Face repo ID to the model checkpoint.")
    parser.add_argument("--arpa", type=str, default="output_w2v2/lm-cim-4-correct.arpa", help="Path to KenLM ARPA model.")
    parser.add_argument("--output-csv", type=str, default="revision_scores.csv", help="Output path for the scoring CSV.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Hub authentication token.")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Loading revisions file: {args.revisions_csv}")
    revisions_list = []
    with open(args.revisions_csv, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    # Check if there is a header
    start_idx = 0
    if lines and ("name" in lines[0].lower() or "revision" in lines[0].lower()):
        start_idx = 1
        
    for line in lines[start_idx:]:
        parts = line.rsplit(None, 1)
        if len(parts) == 2:
            name, rev_hash = parts[0].strip(), parts[1].strip()
            # Clean trailing commas if any from CSV formats
            if name.endswith(","):
                name = name[:-1].strip()
            revisions_list.append((name, rev_hash, rev_hash))
        else:
            print(f"Skipping malformed line: {line}")

    print(f"Found {len(revisions_list)} revisions to evaluate.")

    print(f"Loading test CSV: {args.test_csv}")
    df_test = _try_read_csv(args.test_csv)
    audio_col, text_col = _detect_columns(df_test)
    print(f"Using audio column: '{audio_col}' | text column: '{text_col}'")
    
    df_test[audio_col] = df_test[audio_col].apply(lambda p: _resolve_audio_path(p, args.audio_dir))
    df_test[text_col]  = df_test[text_col].apply(normalize_text)
    df_test.dropna(subset=[audio_col, text_col], inplace=True)
    print(f"Number of test items: {len(df_test)}")

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Prepare Dataset
    print("Preparing HuggingFace dataset...")
    data_dict = {
        "audio": df_test[audio_col].tolist(),
        "sentence": df_test[text_col].tolist()
    }
    from datasets import Features, Value
    features = Features({
        "audio": Audio(sampling_rate=TARGET_SAMPLE_RATE),
        "sentence": Value("string")
    })
    test_ds = Dataset.from_dict(data_dict, features=features)

    # Let's load the processor from the default first checkpoint or default to checkpoint path to get vocabulary
    # Since vocabulary is normally identical, we can load CTC decoder once.
    first_name, first_rev, _ = revisions_list[0]
    print(f"Loading initial processor/decoder from {args.checkpoint} at revision {first_rev} to initialize LM decoder...")
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint, token=token, revision=first_rev)

    # Set up KenLM decoder
    use_lm = False
    processor_with_lm = None
    if os.path.exists(args.arpa):
        print(f"Building CTC decoder using KenLM ARPA model from {args.arpa}...")
        vocab_dict_sorted = processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab_dict_sorted.items(), key=lambda kv: kv[1])
        labels = [t for t, _ in sorted_vocab]
        labels = ["" if t == "[PAD]" else (" " if t == "|" else t) for t in labels]
        
        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=args.arpa,
            alpha=0.5,
            beta=1.0,
        )
        processor_with_lm = Wav2Vec2ProcessorWithLM(
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            decoder=decoder,
        )
        use_lm = True
    else:
        print(f"Warning: ARPA model '{args.arpa}' not found. Evaluation will run without KenLM decoding.")

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        return batch
        
    test_ds_prepared = test_ds.map(prepare_batch, remove_columns=[c for c in test_ds.column_names if c != "sentence"], num_proc=1)

    scores = []

    # Loop over all revisions
    for idx, (name, rev_hash, orig_rev_str) in enumerate(revisions_list):
        print(f"\n[{idx+1}/{len(revisions_list)}] Evaluating revision: {rev_hash} ({name})")
        
        try:
            # Load model
            model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint, token=token, revision=rev_hash)
            model.eval()
            model.to(device)
            
            # If processor needs to be loaded per-revision (just in case vocabulary changed, though unlikely)
            try:
                current_processor = Wav2Vec2Processor.from_pretrained(args.checkpoint, token=token, revision=rev_hash)
            except Exception:
                current_processor = processor

            results = []
            
            for ex in test_ds_prepared:
                input_values = torch.tensor([ex["input_values"]]).to(device)
                with torch.no_grad():
                    logits = model(input_values=input_values).logits
                
                logits_np = logits.squeeze(0).cpu().numpy()
                pred_ids = np.argmax(logits_np, axis=-1)
                hyp_greedy = current_processor.decode(pred_ids).strip()
                
                gold = ex["sentence"]
                
                row_res = {
                    "gold": gold,
                    "greedy": hyp_greedy,
                }
                
                if use_lm:
                    # Use the processor_with_lm.decoder (we assume the vocab didn't change)
                    hyp_lm = processor_with_lm.decoder.decode(logits_np).strip()
                    row_res["kenlm"] = hyp_lm
                    
                results.append(row_res)
            
            # Calculate overall metrics
            golds = [r["gold"] for r in results]
            greedies = [r["greedy"] for r in results]
            
            wer_greedy = jiwer_wer(golds, greedies)
            cer_greedy = jiwer_cer(golds, greedies)
            
            golds_masked = [strip_tones(g) for g in golds]
            greedies_masked = [strip_tones(g) for g in greedies]
            
            wer_greedy_masked = jiwer_wer(golds_masked, greedies_masked)
            cer_greedy_masked = jiwer_cer(golds_masked, greedies_masked)
            
            score_entry = {
                "name": name,
                "revision": rev_hash,
                "greedy_tone_wer": wer_greedy,
                "greedy_tone_cer": cer_greedy,
                "greedy_notone_wer": wer_greedy_masked,
                "greedy_notone_cer": cer_greedy_masked,
            }
            
            if use_lm:
                kenlms = [r["kenlm"] for r in results]
                wer_kenlm = jiwer_wer(golds, kenlms)
                cer_kenlm = jiwer_cer(golds, kenlms)
                
                kenlms_masked = [strip_tones(k) for k in kenlms]
                wer_kenlm_masked = jiwer_wer(golds_masked, kenlms_masked)
                cer_kenlm_masked = jiwer_cer(golds_masked, kenlms_masked)
                
                score_entry.update({
                    "kenlm_tone_wer": wer_kenlm,
                    "kenlm_tone_cer": cer_kenlm,
                    "kenlm_notone_wer": wer_kenlm_masked,
                    "kenlm_notone_cer": cer_kenlm_masked,
                })
                
                print(f"  Greedy: Tone WER = {wer_greedy:.4f} (CER = {cer_greedy:.4f}) | No-Tone WER = {wer_greedy_masked:.4f} (CER = {cer_greedy_masked:.4f})")
                print(f"  KenLM:  Tone WER = {wer_kenlm:.4f} (CER = {cer_kenlm:.4f}) | No-Tone WER = {wer_kenlm_masked:.4f} (CER = {cer_kenlm_masked:.4f})")
            else:
                print(f"  Greedy: Tone WER = {wer_greedy:.4f} (CER = {cer_greedy:.4f}) | No-Tone WER = {wer_greedy_masked:.4f} (CER = {cer_greedy_masked:.4f})")
                
            scores.append(score_entry)
            
        except Exception as e:
            print(f"Error evaluating revision {rev_hash}: {e}")
            
        # Clean up memory
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Save scores to CSV
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(args.output_csv, index=False)
    print(f"\nSuccessfully saved scoring results to {args.output_csv}")

if __name__ == "__main__":
    main()
