#!/usr/bin/env python3
import sys
import os
import argparse
import time
import numpy as np
from pydub import AudioSegment

def get_energy_profile(audio, step_ms=10):
    """
    Computes the dBFS energy profile for the audio segment in step_ms increments.
    Returns a numpy array of dBFS values for each step.
    """
    # pydub calculates dbfs relative to max possible amplitude
    # We iterate over the audio segment in step_ms increments
    num_steps = len(audio) // step_ms
    dbfs_profile = np.zeros(num_steps, dtype=np.float32)
    
    # Pre-extract frame properties for manual calculation to avoid pydub overhead
    # dbfs = 20 * log10(rms / max_possible_amplitude)
    # pydub's chunk.dbfs does this. Let's do it efficiently.
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
    
    # Boolean mask: True if signal is silent (energy < threshold)
    # We handle negative infinity (absolute silence) safely
    is_silent = dbfs_profile < silence_thresh
    
    # We want to identify continuous runs of silence that are >= min_silence_steps.
    # To do this, we can label the silent/nonsilent segments.
    # An easy way is to run a status tracker:
    # Find contiguous silent blocks:
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
    # Handle end of array
    if in_silent_run:
        run_len = len(is_silent) - run_start
        if run_len >= min_silence_steps:
            silent_runs.append((run_start, len(is_silent)))
            
    # Now invert the silent runs to find speaking segments.
    # We start with the entire range [0, len(is_silent)]
    nonsilent_ranges = []
    last_end = 0
    for start, end in silent_runs:
        if start > last_end:
            nonsilent_ranges.append((last_end, start))
        last_end = end
    if last_end < len(is_silent):
        nonsilent_ranges.append((last_end, len(is_silent)))
        
    # Map steps back to milliseconds and apply keep_silence padding
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

def compute_metrics(segments, total_duration_ms):
    if not segments:
        return {
            'percent_segmented': 0.0,
            'avg_len': 0.0,
            'min_len': 0.0,
            'max_len': 0.0,
            'median_len': 0.0,
            'count': 0
        }
        
    durations = [seg['duration'] for seg in segments]
    
    # Calculate non-overlapping segmented duration
    intervals = sorted([(seg['start'], seg['end']) for seg in segments])
    union_duration = 0
    if intervals:
        curr_start, curr_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= curr_end:
                curr_end = max(curr_end, end)
            else:
                union_duration += (curr_end - curr_start)
                curr_start, curr_end = start, end
        union_duration += (curr_end - curr_start)
    else:
        union_duration = 0
        
    percent_segmented = (union_duration / total_duration_ms) * 100.0
    avg_len = np.mean(durations) / 1000.0  # seconds
    min_len = np.min(durations) / 1000.0  # seconds
    max_len = np.max(durations) / 1000.0  # seconds
    median_len = np.median(durations) / 1000.0  # seconds
    
    return {
        'percent_segmented': percent_segmented,
        'avg_len': avg_len,
        'min_len': min_len,
        'max_len': max_len,
        'median_len': median_len,
        'count': len(segments)
    }

def print_table(results):
    header = (
        "| Thresh (dBFS) | Min Sil (ms) | Keep Sil (ms) | Seg Count | % Segmented | Avg Len (s) | Min Len (s) | Median Len (s) | Max Len (s) |"
    )
    separator = (
        "|---------------|--------------|---------------|-----------|-------------|-------------|-------------|----------------|-------------|"
    )
    print(header)
    print(separator)
    for r in results:
        m = r['metrics']
        print(
            f"| {r['silence_thresh']:<13} | {r['min_silence_len']:<12} | {r['keep_silence']:<13} | "
            f"{m['count']:<9} | {m['percent_segmented']:<11.2f}% | {m['avg_len']:<11.2f} | "
            f"{m['min_len']:<11.2f} | {m['median_len']:<14.2f} | {m['max_len']:<11.2f} |"
        )

def main():
    parser = argparse.ArgumentParser(description="Segment audio files and evaluate hyperparameters.")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--sweep", action="store_true", help="Perform a hyperparameter sweep")
    parser.add_argument("--thresh", type=int, default=-40, help="Silence threshold in dBFS (default: -40)")
    parser.add_argument("--min-silence", type=int, default=500, help="Minimum silence length in ms (default: 500)")
    parser.add_argument("--keep-silence", type=int, default=100, help="Keep silence padding in ms (default: 100)")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: file {args.audio_path} does not exist.")
        sys.exit(1)
        
    print(f"Loading and decoding audio: {args.audio_path}")
    t0 = time.time()
    audio = AudioSegment.from_file(args.audio_path)
    total_len = len(audio)
    print(f"Loaded {total_len / 1000.0:.2f}s of audio in {time.time() - t0:.2f}s")
    
    print("Computing energy profile (dBFS)...")
    t0 = time.time()
    dbfs_profile = get_energy_profile(audio, step_ms=10)
    print(f"Energy profile computed in {time.time() - t0:.2f}s")
    
    if args.sweep:
        thresholds = [-50, -45, -40, -35, -30]
        min_silence_lens = [300, 500, 1000]
        keep_silences = [100, 200]
        
        results = []
        t0 = time.time()
        for thresh in thresholds:
            for min_sil in min_silence_lens:
                for keep_sil in keep_silences:
                    segments = segment_audio_from_profile(
                        dbfs_profile,
                        total_len,
                        step_ms=10,
                        min_silence_len=min_sil,
                        silence_thresh=thresh,
                        keep_silence=keep_sil
                    )
                    metrics = compute_metrics(segments, total_len)
                    results.append({
                        'silence_thresh': thresh,
                        'min_silence_len': min_sil,
                        'keep_silence': keep_sil,
                        'metrics': metrics
                    })
        sweep_time = time.time() - t0
        print(f"\n### Hyperparameter Sweep Results (Completed {len(results)} configurations in {sweep_time:.4f}s)\n")
        print_table(results)
    else:
        # Single run
        segments = segment_audio_from_profile(
            dbfs_profile,
            total_len,
            step_ms=10,
            min_silence_len=args.min_silence,
            silence_thresh=args.thresh,
            keep_silence=args.keep_silence
        )
        metrics = compute_metrics(segments, total_len)
        results = [{
            'silence_thresh': args.thresh,
            'min_silence_len': args.min_silence,
            'keep_silence': args.keep_silence,
            'metrics': metrics
        }]
        print("\n### Segmentation Results\n")
        print_table(results)

if __name__ == "__main__":
    main()
