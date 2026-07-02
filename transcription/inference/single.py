#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inference.py

Runs speech-to-text inference on a single audio file using the fine-tuned Wav2Vec2 model.
"""

import os

# Enable fallback to CPU for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import re
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder

from transcription.utils.model_utils import get_best_model_config

TARGET_SAMPLE_RATE = 16000


def main():
    model_config = get_best_model_config()
    default_repo = model_config["repo"]
    default_revision = model_config["revision"]

    parser = argparse.ArgumentParser(
        description="Run Wav2Vec2 ASR inference on a single audio file."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input audio file (WAV, FLAC, MP3, etc.).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_repo,
        help="Path to model checkpoint or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=default_repo,
        help="Path to saved processor or Hugging Face Hub repo ID.",
    )
    parser.add_argument(
        "--arpa",
        type=str,
        default="output_w2v2/lm-cim-4-correct.arpa",
        help="Path to KenLM ARPA model.",
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
        default=default_revision,
        help="Specific Hugging Face Hub commit hash, branch, or tag.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file '{args.audio_path}' not found.")
        sys.exit(1)

    token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    if os.path.exists(args.processor):
        print(f"Loading processor from local path: {args.processor}...")
    else:
        print(
            f"Loading processor from Hugging Face Hub: {args.processor} (revision: {args.revision})..."
        )
    processor = Wav2Vec2Processor.from_pretrained(
        args.processor, token=token, revision=args.revision
    )

    if os.path.exists(args.checkpoint):
        print(f"Loading model from local path: {args.checkpoint}...")
    else:
        print(
            f"Loading model from Hugging Face Hub: {args.checkpoint} (revision: {args.revision})..."
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
    print(f"Using device: {device}")
    model.to(device)

    # Load audio using soundfile
    print(f"Loading audio file: {args.audio_path}...")
    speech_array, sample_rate = sf.read(args.audio_path)

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
        print(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz...")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Squeeze to 1D
    speech = waveform.squeeze(0).numpy()

    # Feature extraction
    input_values = processor(speech, sampling_rate=TARGET_SAMPLE_RATE).input_values[0]
    input_tensor = torch.tensor([input_values]).to(device)

    # Model inference with device safety/fallback
    try:
        with torch.no_grad():
            logits = model(input_tensor).logits
    except NotImplementedError as e:
        if device == "mps":
            print(f"MPS execution failed due to unsupported operators: {e}")
            print("Falling back to CPU backend...")
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

    print("\n" + "=" * 60)
    print("GREEDY DECODING PREDICTIONS:")
    print(f"  Transcription: {greedy_raw}")
    print(f"  Confidence:    {greedy_confidence:.4f} ({greedy_confidence:.2%})")
    print("=" * 60 + "\n")

    # KenLM decode
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

        # Run LM decode using decode_beams to get confidence/scores
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

        print("=" * 60)
        print("KENLM DECODING PREDICTIONS:")
        print(f"  Transcription:  {lm_raw}")
        print(f"  Logit Score:    {logit_score:.4f}")
        print(f"  Combined Score: {combined_score:.4f}")
        print("=" * 60 + "\n")
    else:
        print(f"Warning: ARPA model '{args.arpa}' not found. Skipping KenLM decoding.")


if __name__ == "__main__":
    main()
