#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_inference.py

Runs speech-to-text inference on a directory of WAV files using the fine-tuned Wav2Vec2 model.
"""

import os

# Enable fallback to CPU for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import re
import sys
import glob
import csv
import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder

import time

TARGET_SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(
        description="Run Wav2Vec2 ASR batch inference on a directory of audio files."
    )
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to the directory containing WAV files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="charliemcvicker/asr-cherokee",
        help="Path to model checkpoint or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="charliemcvicker/asr-cherokee",
        help="Path to saved processor or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--arpa",
        type=str,
        default=None,
        help="Path to KenLM ARPA model (optional). If omitted, only greedy decoding is done.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face Hub authentication token.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="5464d15",
        help="Specific Hugging Face Hub commit hash, branch, or tag.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_inference_results.csv",
        help="Path to the output CSV file to save results.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir_path):
        print(f"Error: Directory '{args.dir_path}' not found.", flush=True)
        sys.exit(1)

    # Find WAV files
    wav_files = []
    for ext in ("*.wav", "*.WAV"):
        wav_files.extend(glob.glob(os.path.join(args.dir_path, ext)))
    
    wav_files = sorted(list(set(wav_files)))
    if not wav_files:
        print(f"No WAV files found in '{args.dir_path}'.", flush=True)
        sys.exit(0)

    print(f"Found {len(wav_files)} WAV files to process.", flush=True)

    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if os.path.exists(args.processor):
        print(f"Loading processor from local path: {args.processor}...", flush=True)
    else:
        print(
            f"Loading processor from Hugging Face Hub: {args.processor} (revision: {args.revision})...",
            flush=True
        )
    processor = Wav2Vec2Processor.from_pretrained(
        args.processor, token=token, revision=args.revision
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading model from local path: {args.checkpoint}...", flush=True)
    else:
        print(
            f"Loading model from Hugging Face Hub: {args.checkpoint} (revision: {args.revision})...",
            flush=True
        )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.checkpoint, token=token, revision=args.revision
    )
    model.eval()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}", flush=True)
    model.to(device)

    # Initialize decoder if KenLM is requested
    decoder = None
    processor_with_lm = None
    if args.arpa:
        if os.path.exists(args.arpa):
            print(f"Building CTC decoder using KenLM ARPA model from {args.arpa}...", flush=True)
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
        else:
            print(f"Warning: ARPA model '{args.arpa}' not found. Skipping KenLM decoding.", flush=True)

    # Write headers to CSV output file immediately
    headers = ["file_path", "filename", "greedy_transcription", "greedy_confidence"]
    if processor_with_lm:
        headers.append("kenlm_transcription")
        headers.append("kenlm_logit_score")
        headers.append("kenlm_combined_score")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    start_time = time.time()
    durations = []

    for i, audio_path in enumerate(wav_files, 1):
        filename = os.path.basename(audio_path)
        item_start = time.time()

        try:
            # Load audio using soundfile
            speech_array, sample_rate = sf.read(audio_path)

            # Convert numpy array (frames, channels) -> torch tensor (channels, frames)
            waveform = torch.tensor(speech_array, dtype=torch.float32)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)  # (1, frames)
            else:
                waveform = waveform.transpose(0, 1)  # (channels, frames)

            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Squeeze to 1D
            speech = waveform.squeeze(0).numpy()

            # Feature extraction
            input_values = processor(speech, sampling_rate=TARGET_SAMPLE_RATE).input_values[0]
            input_tensor = torch.tensor(np.array([input_values])).to(device)

            # Model inference with device safety/fallback
            try:
                with torch.no_grad():
                    logits = model(input_tensor).logits
            except NotImplementedError as e:
                if device == "mps":
                    print(f"  MPS execution failed due to unsupported operators: {e}", flush=True)
                    print("  Falling back to CPU backend for this file...", flush=True)
                    device = "cpu"
                    model.to(device)
                    input_tensor = input_tensor.to(device)
                    with torch.no_grad():
                        logits = model(input_tensor).logits
                else:
                    raise e

            logits_np = logits.squeeze(0).cpu().numpy()

            # Softmax to get confidence probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_ids = np.argmax(probs, axis=-1)
            greedy_raw = processor.decode(pred_ids).strip()

            # Calculate token level confidence
            token_probs = probs[np.arange(len(pred_ids)), pred_ids]
            pad_id = getattr(processor.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = processor.tokenizer.vocab.get("[PAD]", 0)
            
            non_pad_mask = pred_ids != pad_id
            if non_pad_mask.any():
                greedy_confidence = float(np.mean(token_probs[non_pad_mask]))
            else:
                greedy_confidence = float(np.mean(token_probs))

            row = [audio_path, filename, greedy_raw, f"{greedy_confidence:.4f}"]

            # KenLM decode
            if processor_with_lm:
                beams = processor_with_lm.decoder.decode_beams(logits_np)
                if beams:
                    best_beam = beams[0]
                    lm_raw = best_beam[0].strip()
                    logit_score = float(best_beam[3])
                    combined_score = float(best_beam[4])
                else:
                    lm_raw = ""
                    logit_score = 0.0
                    combined_score = 0.0
                
                row.extend([lm_raw, f"{logit_score:.4f}", f"{combined_score:.4f}"])

            # Append to CSV output file
            with open(args.output, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            duration = time.time() - item_start
            durations.append(duration)
            avg_duration = sum(durations) / len(durations)
            remaining = len(wav_files) - i
            eta = remaining * avg_duration
            
            print(f"[{i}/{len(wav_files)}] {filename} ({duration:.2f}s, avg: {avg_duration:.2f}s, ETA: {eta:.1f}s)", flush=True)
            print(f"  Greedy: {greedy_raw} (Confidence: {greedy_confidence:.4f})", flush=True)
            if processor_with_lm:
                print(f"  KenLM:  {lm_raw} (Logit: {logit_score:.4f}, Combined: {combined_score:.4f})", flush=True)

        except Exception as e:
            print(f"Error processing {filename}: {e}", flush=True)

    total_time = time.time() - start_time
    print(f"\nBatch processing complete in {total_time:.2f}s. Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
