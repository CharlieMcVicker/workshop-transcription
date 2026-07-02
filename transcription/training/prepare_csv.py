#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local CSV to Wav2Vec2 Dataset splits generator.
Transforms local sentence CSV file and a directory of WAV files
into shuffled and filtered Train / Validation / Test CSV splits for Wav2Vec2.
"""

import os
import argparse
import random
import wave
import contextlib
import csv
from transcription.utils.tone_normalization import remove_tones_and_double_vowels



def clean_transcription(text):
    """
    Cleans up transcription text by removing punctuation/formatting,
    collapsing multiple spaces, and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Characters to clean up (including asterisks used for formatting/bolding words in sentences_audio.csv)
    punctuation = [
        "[", "]", "\"", "(", ")", ".", "\u0f7b", "_", "|", "》", "?", "!",
        "/", ",", "-", "?", "<", "…", ">", "*"
    ]
    for p in punctuation:
        text = text.replace(p, " ")
    
    # Collapse multiple spaces and lowercase the result
    return " ".join(text.split()).lower()


def get_audio_metadata(filepath):
    """
    Extracts the duration (in seconds) and file size (in bytes) of a WAV file.
    """
    try:
        filesize = os.path.getsize(filepath)
        with contextlib.closing(wave.open(filepath, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return filesize, duration
    except Exception as e:
        print(f"Warning: Failed to process audio file '{filepath}': {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Convert local sentence CSV and WAV directory to Wav2Vec2 training splits."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/sentence_audio.csv",
        help="Path to the input CSV file containing metadata (default: data/processed/sentence_audio.csv)"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/processed/sentence_audio",
        help="Path to the directory containing audio files (default: data/processed/sentence_audio)"
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="phonetic",
        help="Column name in the CSV to use for transcription text (default: phonetic)"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="data/processed/cim-wav2vec2",
        help="Prefix for the generated train/valid/test split CSV files (default: data/processed/cim-wav2vec2)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum audio duration (seconds) to include in the splits (default: 15.0)"
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[80.0, 10.0, 10.0],
        help="Train, Validation, and Test split percentages summing to 100 (default: 80 10 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when shuffling dataset (default: 42)"
    )

    args = parser.parse_args()

    # Verify input CSV exists
    if not os.path.isfile(args.csv):
        print(f"Error: Input CSV file '{args.csv}' not found.")
        return

    # Verify audio directory exists
    if not os.path.isdir(args.audio_dir):
        print(f"Error: Audio directory '{args.audio_dir}' not found.")
        return

    # Verify split sum
    if sum(args.split) != 100.0:
        print(f"Error: Splitting percentages {args.split} must sum up to exactly 100.")
        return

    print(f"Reading metadata from '{args.csv}'...")
    
    samples = []
    
    with open(args.csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check column headers
        if "audio" not in reader.fieldnames:
            print("Error: Input CSV must contain an 'audio' column specifying filename.")
            return
        if args.text_col not in reader.fieldnames:
            print(f"Error: Transcription column '{args.text_col}' not found in CSV headers: {reader.fieldnames}")
            return

        for row in reader:
            samples.append(row)

    print("Analyzing audio files and extracting metadata...")
    valid_samples = []
    missing_count = 0
    empty_transcripts = 0
    duration_filtered = 0
    dropped_tone_count = 0

    for row in samples:
        filename = row['audio']
        filepath = os.path.join(args.audio_dir, filename)
        
        # 1. Check audio file exists
        if not os.path.isfile(filepath):
            missing_count += 1
            continue

        size, dur = get_audio_metadata(filepath)
        
        if size is None or dur is None:
            continue

        # 2. Reformat transcription
        raw_text = row[args.text_col]
        if isinstance(raw_text, str):
            raw_text = raw_text.replace("*", "")
        norm_text, should_drop = remove_tones_and_double_vowels(raw_text)
        if should_drop:
            dropped_tone_count += 1
            continue
            
        cleaned_text = clean_transcription(norm_text)
        
        if cleaned_text == "":
            empty_transcripts += 1
            continue

        # 3. Filter by duration
        if dur > args.max_duration:
            duration_filtered += 1
            continue

        valid_samples.append({
            'path': filepath,
            'sentence': cleaned_text
        })

    if missing_count > 0:
        print(f"Note: {missing_count} audio files referenced in CSV were not found in '{args.audio_dir}'.")
    if dropped_tone_count > 0:
        print(f"Note: {dropped_tone_count} samples dropped due to rare tone/diacritic marks.")
    if empty_transcripts > 0:
        print(f"Note: {empty_transcripts} empty transcripts removed.")
    if duration_filtered > 0:
        print(f"Note: {duration_filtered} samples dropped for exceeding maximum duration of {args.max_duration}s.")


    # Shuffle the dataset
    print(f"Shuffling dataset with seed {args.seed}...")
    random.seed(args.seed)
    random.shuffle(valid_samples)

    total_samples = len(valid_samples)
    if total_samples == 0:
        print("Error: No samples remaining after processing and filtering.")
        return

    # Compute partition splits
    train_pct, valid_pct, _ = args.split
    train_end = int(round(total_samples * (train_pct / 100.0)))
    valid_end = int(round(total_samples * ((train_pct + valid_pct) / 100.0)))

    train_data = valid_samples[:train_end]
    valid_data = valid_samples[train_end:valid_end]
    test_data = valid_samples[valid_end:]

    print(f"\nFinal dataset breakdown ({total_samples} total samples):")
    print(f" - Train:      {len(train_data)} samples ({train_pct}%)")
    print(f" - Validation: {len(valid_data)} samples ({valid_pct}%)")
    print(f" - Test:       {len(test_data)} samples ({args.split[2]}%)")

    # Export split files
    train_csv = f"{args.output_prefix}-train.csv"
    valid_csv = f"{args.output_prefix}-valid.csv"
    test_csv = f"{args.output_prefix}-test.csv"

    def write_csv(data, filename):
        with open(filename, mode='w', encoding='utf-8', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(['path', 'sentence'])
            for item in data:
                writer.writerow([item['path'], item['sentence']])

    print(f"\nWriting output CSV files...")
    write_csv(train_data, train_csv)
    write_csv(valid_data, valid_csv)
    write_csv(test_data, test_csv)

    print(f"Generated: {train_csv}")
    print(f"Generated: {valid_csv}")
    print(f"Generated: {test_csv}")
    print("Done!")


if __name__ == "__main__":
    main()
