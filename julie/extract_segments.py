#!/usr/bin/env python3
import sys
import os
import argparse
import time
import json
import numpy as np
from pydub import AudioSegment

def get_energy_profile(audio, step_ms=10):
    """
    Computes the dBFS energy profile for the audio segment in step_ms increments.
    """
    num_steps = len(audio) // step_ms
    dbfs_profile = np.zeros(num_steps, dtype=np.float32)
    
    for i in range(num_steps):
        chunk = audio[i * step_ms : (i + 1) * step_ms]
        dbfs_profile[i] = chunk.dBFS
        
    return dbfs_profile

def segment_audio_from_profile(dbfs_profile, total_duration_ms, step_ms=10, min_silence_len=500, silence_thresh=-40, keep_silence=100):
    """
    Finds nonsilent segments directly from the precomputed dBFS energy profile.
    """
    min_silence_steps = min_silence_len // step_ms
    keep_silence_steps = keep_silence // step_ms
    
    is_silent = dbfs_profile < silence_thresh
    
    silent_runs = []
    in_silent_run = False
    run_start = 0
    
    for i, silent in enumerate(is_silent):
        if silent:
            if not in_silent_run:
                in_silent_run = True
                run_start = i
        else:
            if in_silent_run:
                in_silent_run = False
                run_len = i - run_start
                if run_len >= min_silence_steps:
                    silent_runs.append((run_start, i))
                    
    if in_silent_run:
        run_len = len(is_silent) - run_start
        if run_len >= min_silence_steps:
            silent_runs.append((run_start, len(is_silent)))
            
    nonsilent_ranges = []
    last_end = 0
    for start, end in silent_runs:
        if start > last_end:
            nonsilent_ranges.append((last_end, start))
        last_end = end
    if last_end < len(is_silent):
        nonsilent_ranges.append((last_end, len(is_silent)))
        
    segments = []
    for start_step, end_step in nonsilent_ranges:
        start_ms = start_step * step_ms
        end_ms = end_step * step_ms
        
        pad_start = max(0, start_ms - keep_silence)
        pad_end = min(total_duration_ms, end_ms + keep_silence)
        segments.append({
            'start': pad_start,
            'end': pad_end,
            'duration': pad_end - pad_start
        })
        
    return segments

def main():
    parser = argparse.ArgumentParser(description="Extract audio segments and write a manifest JSON.")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--out-dir", default="segments", help="Directory to save extracted segments (default: segments)")
    parser.add_argument("--thresh", type=int, default=-30, help="Silence threshold in dBFS (default: -30)")
    parser.add_argument("--min-silence", type=int, default=500, help="Minimum silence length in ms (default: 500)")
    parser.add_argument("--keep-silence", type=int, default=200, help="Keep silence padding in ms (default: 200)")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: file {args.audio_path} does not exist.")
        sys.exit(1)
        
    # Set up output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading and decoding audio: {args.audio_path}")
    t0 = time.time()
    audio = AudioSegment.from_file(args.audio_path)
    total_len = len(audio)
    print(f"Loaded {total_len / 1000.0:.2f}s of audio in {time.time() - t0:.2f}s")
    
    print("Computing energy profile (dBFS)...")
    t0 = time.time()
    dbfs_profile = get_energy_profile(audio, step_ms=10)
    print(f"Energy profile computed in {time.time() - t0:.2f}s")
    
    print(f"Segmenting audio (thresh={args.thresh}, min_silence={args.min_silence}, keep_silence={args.keep_silence})...")
    segments = segment_audio_from_profile(
        dbfs_profile,
        total_len,
        step_ms=10,
        min_silence_len=args.min_silence,
        silence_thresh=args.thresh,
        keep_silence=args.keep_silence
    )
    
    print(f"Found {len(segments)} segments. Extracting to disk...")
    
    # Base name for output segments
    base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    
    manifest = []
    
    for idx, seg in enumerate(segments):
        start_ms = seg['start']
        end_ms = seg['end']
        duration_ms = seg['duration']
        
        # Slicing pydub AudioSegment
        chunk = audio[start_ms:end_ms]
        
        # Define output filename
        out_filename = f"{base_name}_segment_{idx:04d}_{start_ms}_{end_ms}.wav"
        out_filepath = os.path.join(args.out_dir, out_filename)
        
        # Export segment
        chunk.export(out_filepath, format="wav")
        
        # Add to manifest
        manifest.append({
            "id": idx,
            "filename": out_filename,
            "filepath": out_filepath,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "start_seconds": start_ms / 1000.0,
            "end_seconds": end_ms / 1000.0,
            "duration_seconds": duration_ms / 1000.0
        })
        
    # Write manifest.json
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Extracted {len(segments)} segments to '{args.out_dir}/'")
    print(f"Saved manifest to '{manifest_path}'")

if __name__ == "__main__":
    main()
