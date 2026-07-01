#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_inference.py

Runs speech-to-text inference on a directory of WAV files using the fine-tuned Wav2Vec2 model.
Optimized using batched GPU inference and multiprocessed CPU CTC/KenLM decoding.
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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import time
import multiprocessing

TARGET_SAMPLE_RATE = 16000

# Global variables in the worker processes to avoid serializing the decoder/processor objects
global_decoder = None
global_processor = None


def init_worker(processor_path, arpa_path, token, revision):
    """
    Initialize global decoder and processor in worker processes once.
    This avoids pickle serialization overhead and limits thread subscription.
    """
    global global_decoder, global_processor
    # Restrict internal Torch threading in workers to prevent CPU oversubscription
    torch.set_num_threads(1)
    
    from transformers import Wav2Vec2Processor
    from pyctcdecode import build_ctcdecoder
    import os

    global_processor = Wav2Vec2Processor.from_pretrained(
        processor_path, token=token, revision=revision
    )

    if arpa_path and os.path.exists(arpa_path):
        vocab_dict_sorted = global_processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab_dict_sorted.items(), key=lambda kv: kv[1])
        labels = [t for t, _ in sorted_vocab]
        labels = ["" if t == "[PAD]" else (" " if t == "|" else t) for t in labels]

        global_decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=arpa_path,
            alpha=0.5,
            beta=1.0,
        )


def decode_worker(item_data):
    """
    Worker function to decode logits.
    item_data is a tuple: (index, filename, audio_path, logits_np)
    """
    global global_decoder, global_processor
    import numpy as np
    import torch
    
    idx, filename, audio_path, logits_np = item_data
    
    # 1. Greedy Decode
    logits_tensor = torch.tensor(logits_np)
    probs = torch.nn.functional.softmax(logits_tensor, dim=-1).numpy()
    pred_ids = np.argmax(probs, axis=-1)
    greedy_raw = global_processor.decode(pred_ids).strip()

    # Calculate token level confidence
    token_probs = probs[np.arange(len(pred_ids)), pred_ids]
    pad_id = getattr(global_processor.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = global_processor.tokenizer.vocab.get("[PAD]", 0)
    
    non_pad_mask = pred_ids != pad_id
    if non_pad_mask.any():
        greedy_confidence = float(np.mean(token_probs[non_pad_mask]))
    else:
        greedy_confidence = float(np.mean(token_probs))

    # 2. KenLM Decode
    lm_raw = ""
    logit_score = 0.0
    combined_score = 0.0
    
    if global_decoder is not None:
        beams = global_decoder.decode_beams(logits_np)
        if beams:
            best_beam = beams[0]
            lm_raw = best_beam[0].strip()
            logit_score = float(best_beam[3])
            combined_score = float(best_beam[4])

    return {
        "index": idx,
        "filename": filename,
        "audio_path": audio_path,
        "greedy_raw": greedy_raw,
        "greedy_confidence": greedy_confidence,
        "lm_raw": lm_raw,
        "logit_score": logit_score,
        "combined_score": combined_score
    }


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
        default="data/results/batch_inference_results.csv",
        help="Path to the output CSV file to save results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GPU forward pass (default: 16).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel decoding (default: CPU count).",
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

    # Load and resample audio files (pre-load in memory for batching/sorting)
    # We sort by length to minimize padding overhead during batched inference.
    loaded_audios = []
    print("Loading and preparing audio files in memory...", flush=True)
    audio_load_start = time.time()
    for audio_path in wav_files:
        filename = os.path.basename(audio_path)
        try:
            speech_array, sample_rate = sf.read(audio_path)
            waveform = torch.tensor(speech_array, dtype=torch.float32)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)
            
            speech = waveform.squeeze(0).numpy()
            loaded_audios.append({
                "audio_path": audio_path,
                "filename": filename,
                "speech": speech,
                "length": len(speech)
            })
        except Exception as e:
            print(f"Error loading {filename}: {e}", flush=True)

    if not loaded_audios:
        print("No audio files successfully loaded. Exiting.", flush=True)
        sys.exit(0)

    # Sort by length
    loaded_audios.sort(key=lambda x: x["length"])
    print(f"Successfully loaded {len(loaded_audios)} files in {time.time() - audio_load_start:.2f}s.", flush=True)

    # Create batches
    batches = [loaded_audios[i : i + args.batch_size] for i in range(0, len(loaded_audios), args.batch_size)]
    
    # Store logits data for multiprocessing
    # Items: (global_index, filename, audio_path, logits_np)
    logits_data = []
    global_idx = 0

    gpu_start_time = time.time()
    print(f"Running batched GPU inference (batch size: {args.batch_size}, total batches: {len(batches)})...", flush=True)
    for batch_idx, batch in enumerate(batches, 1):
        speech_list = [item["speech"] for item in batch]
        
        inputs = processor(
            speech_list,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        try:
            with torch.no_grad():
                if attention_mask is not None:
                    batch_logits = model(input_values, attention_mask=attention_mask).logits
                else:
                    batch_logits = model(input_values).logits
        except NotImplementedError as e:
            if device == "mps":
                print("  MPS execution failed. Falling back to CPU backend for this batch...", flush=True)
                device = "cpu"
                model.to(device)
                input_values = input_values.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with torch.no_grad():
                    if attention_mask is not None:
                        batch_logits = model(input_values, attention_mask=attention_mask).logits
                    else:
                        batch_logits = model(input_values).logits
            else:
                raise e

        # Extract and unpad logits for each item
        for item_idx, item in enumerate(batch):
            input_len = len(item["speech"])
            logit_len = int(model._get_feat_extract_output_lengths(input_len))
            logits_np = batch_logits[item_idx, :logit_len].cpu().numpy()
            logits_data.append((global_idx, item["filename"], item["audio_path"], logits_np))
            global_idx += 1

    print(f"GPU inference completed in {time.time() - gpu_start_time:.2f}s.", flush=True)

    # Initialize multiprocessing Pool for parallel decoding
    num_workers = args.num_workers or multiprocessing.cpu_count()
    print(f"Initializing multiprocessing pool with {num_workers} workers...", flush=True)
    
    # Initialize child processes by loading processor and CTC decoder (ARPA) once per worker
    pool = multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.processor, args.arpa, token, args.revision)
    )

    print(f"Starting parallel decoding of {len(logits_data)} items...", flush=True)
    decode_start_time = time.time()
    
    results = pool.map(decode_worker, logits_data)
    pool.close()
    pool.join()
    
    print(f"Decoding completed in {time.time() - decode_start_time:.2f}s.", flush=True)

    # Sort results back to original directory reading order
    wav_order = {path: idx for idx, path in enumerate(wav_files)}
    results.sort(key=lambda x: wav_order.get(x["audio_path"], 0))

    # Write results to CSV
    headers = ["file_path", "filename", "greedy_transcription", "greedy_confidence"]
    if args.arpa:
        headers.append("kenlm_transcription")
        headers.append("kenlm_logit_score")
        headers.append("kenlm_combined_score")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for res in results:
            row = [res["audio_path"], res["filename"], res["greedy_raw"], f"{res['greedy_confidence']:.4f}"]
            if args.arpa:
                row.extend([res["lm_raw"], f"{res['logit_score']:.4f}", f"{res['combined_score']:.4f}"])
            writer.writerow(row)

    total_time = time.time() - audio_load_start
    print(f"\nBatch processing complete in {total_time:.2f}s. Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
