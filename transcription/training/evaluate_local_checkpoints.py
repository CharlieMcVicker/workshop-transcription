#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_local_checkpoints.py

Evaluates all local checkpoints in a directory on the test dataset.
Computes standard metrics as well as vowel-length-masked and tone-masked metrics.
Avoids downloading from Hugging Face Hub.
Does not use KenLM at all.
"""

import os
import re
import glob
import unicodedata
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset, Audio
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000
apostrophe_variants = r"[’‘ʼʻ`´‛]"
chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]'

def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(apostrophe_variants, "'", text)
    text = re.sub(chars_to_remove_regex, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_tones(text):
    # Remove all digits (0-9) representing tones
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_length(text):
    if not isinstance(text, str):
        return ""
    # Collapse any sequence of 2 or more of the same vowel into a single vowel
    return re.sub(r"([aeiouv])\1+", r"\1", text)

def strip_both(text):
    return strip_length(strip_tones(text))

def _try_read_csv(path):
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return pd.read_csv(path)

def _detect_columns(df):
    audio_candidates = ["path", "audio", "wav", "file", "filename", "filepath", "audio_path"]
    text_candidates  = ["sentence", "text", "transcription", "transcript", "label", "target"]
    cols_lower = {c.lower(): c for c in df.columns}
    
    audio_col = None
    text_col = None
    for cand in audio_candidates:
        if cand in cols_lower:
            audio_col = cols_lower[cand]
            break
    for cand in text_candidates:
        if cand in cols_lower:
            text_col = cols_lower[cand]
            break
    if audio_col is None or text_col is None:
        raise ValueError(f"Could not auto-detect columns from: {list(df.columns)}")
    return audio_col, text_col

def _resolve_audio_path(p, dataset_path):
    p = str(p).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(dataset_path, p)
    if os.path.exists(cand):
        return cand
    for sub in ["wavs", "clips", "audio", "data"]:
        cand2 = os.path.join(dataset_path, sub, os.path.basename(p))
        if os.path.exists(cand2):
            return cand2
    return p

def main():
    parser = argparse.ArgumentParser(description="Evaluate all local Wav2Vec2 checkpoints on a test dataset.")
    parser.add_argument("--checkpoints-dir", type=str, default="output_w2v2/wav2vec2-large-xlsr", help="Directory containing checkpoint directories.")
    parser.add_argument("--test-csv", type=str, default="data/processed/cim-wav2vec2-test.csv", help="Path to the test CSV file.")
    parser.add_argument("--audio-dir", type=str, default="data/processed/sentence_audio", help="Directory containing audio files.")
    parser.add_argument("--output-csv", type=str, default="data/results/local_checkpoint_scores.csv", help="Output path for the scoring CSV.")
    parser.add_argument("--processor", type=str, default=None, help="Path or HF repo ID to the processor (defaults to checkpoints-dir).")
    args = parser.parse_args()

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

    # Find checkpoints
    ckpt_dirs = glob.glob(os.path.join(args.checkpoints_dir, "checkpoint-*"))
    def _step(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpt_dirs = sorted(ckpt_dirs, key=_step)
    
    # Also evaluate the final promote model in checkpoints-dir itself if config exists
    checkpoints = []
    for d in ckpt_dirs:
        if os.path.exists(os.path.join(d, "config.json")):
            checkpoints.append((os.path.basename(d), d))
            
    if os.path.exists(os.path.join(args.checkpoints_dir, "config.json")):
        checkpoints.append(("final", args.checkpoints_dir))

    if not checkpoints:
        print(f"No valid checkpoints found in {args.checkpoints_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints/models to evaluate.")

    # Load processor
    processor_path = args.processor
    if not processor_path:
        # Try checking if checkpoints-dir or subfolders have vocab.json
        candidates = [
            args.checkpoints_dir,
            os.path.join(args.checkpoints_dir, "wav2vec2-large-xlsr"),
            checkpoints[0][1]
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "vocab.json")):
                processor_path = cand
                break
        if not processor_path:
            processor_path = args.checkpoints_dir
    
    print(f"Loading processor from {processor_path}...")
    processor = Wav2Vec2Processor.from_pretrained(processor_path)

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

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        return batch
        
    test_ds_prepared = test_ds.map(prepare_batch, remove_columns=[c for c in test_ds.column_names if c != "sentence"], num_proc=1)

    scores = []

    # Loop over all checkpoints
    for idx, (name, ckpt_path) in enumerate(checkpoints):
        print(f"\n[{idx+1}/{len(checkpoints)}] Evaluating checkpoint: {name} (path: {ckpt_path})")
        
        try:
            # Load model
            model = Wav2Vec2ForCTC.from_pretrained(ckpt_path)
            model.eval()
            model.to(device)

            results = []
            
            for ex in test_ds_prepared:
                input_values = torch.tensor([ex["input_values"]]).to(device)
                with torch.no_grad():
                    logits = model(input_values=input_values).logits
                
                logits_np = logits.squeeze(0).cpu().numpy()
                pred_ids = np.argmax(logits_np, axis=-1)
                hyp_greedy = processor.decode(pred_ids).strip()
                
                gold = ex["sentence"]
                
                row_res = {
                    "gold": gold,
                    "greedy": hyp_greedy,
                }
                
                results.append(row_res)
            
            # Calculate overall metrics
            golds = [r["gold"] for r in results]
            greedies = [r["greedy"] for r in results]
            
            def safe(s):
                return s if s.strip() else " "

            golds_safe = [safe(g) for g in golds]
            greedies_safe = [safe(g) for g in greedies]

            wer_greedy = jiwer_wer(golds_safe, greedies_safe)
            cer_greedy = jiwer_cer(golds_safe, greedies_safe)
            
            # Vowel-length masked
            golds_vowel_masked = [safe(strip_length(g)) for g in golds]
            greedies_vowel_masked = [safe(strip_length(g)) for g in greedies]
            wer_greedy_vowel_masked = jiwer_wer(golds_vowel_masked, greedies_vowel_masked)
            cer_greedy_vowel_masked = jiwer_cer(golds_vowel_masked, greedies_vowel_masked)

            # Tone masked
            golds_tone_masked = [safe(strip_tones(g)) for g in golds]
            greedies_tone_masked = [safe(strip_tones(g)) for g in greedies]
            wer_greedy_tone_masked = jiwer_wer(golds_tone_masked, greedies_tone_masked)
            cer_greedy_tone_masked = jiwer_cer(golds_tone_masked, greedies_tone_masked)

            # Both masked
            golds_both_masked = [safe(strip_both(g)) for g in golds]
            greedies_both_masked = [safe(strip_both(g)) for g in greedies]
            wer_greedy_both_masked = jiwer_wer(golds_both_masked, greedies_both_masked)
            cer_greedy_both_masked = jiwer_cer(golds_both_masked, greedies_both_masked)
            
            score_entry = {
                "checkpoint": name,
                "path": ckpt_path,
                "greedy_wer": wer_greedy,
                "greedy_cer": cer_greedy,
                "greedy_vowel_masked_wer": wer_greedy_vowel_masked,
                "greedy_vowel_masked_cer": cer_greedy_vowel_masked,
                "greedy_tone_masked_wer": wer_greedy_tone_masked,
                "greedy_tone_masked_cer": cer_greedy_tone_masked,
                "greedy_both_masked_wer": wer_greedy_both_masked,
                "greedy_both_masked_cer": cer_greedy_both_masked,
            }
            
            print(f"  Greedy: Raw WER = {wer_greedy:.4f} (CER = {cer_greedy:.4f}) | Vowel-Masked WER = {wer_greedy_vowel_masked:.4f} (CER = {cer_greedy_vowel_masked:.4f})")
            scores.append(score_entry)
            
        except Exception as e:
            print(f"Error evaluating checkpoint {name}: {e}")
            
        # Clean up memory
        if 'model' in locals():
            del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
            
    if not scores:
        print("\nNo checkpoints were successfully evaluated.")
        return

    # Save scores to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(args.output_csv, index=False)
    print(f"\nSuccessfully saved scoring results to {args.output_csv}")
    print("\nRanking table (ordered by greedy_wer):")
    print(scores_df[["checkpoint", "greedy_wer", "greedy_cer", "greedy_vowel_masked_wer", "greedy_vowel_masked_cer"]].to_string(index=False))

if __name__ == "__main__":
    main()
