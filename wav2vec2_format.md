# Wav2Vec2 Dataset Format and Structure

This document outlines the file structure and data format generated at the end of the ASR transcription data pipeline by [from_gsheet_to_wav2vec2_files.py](file:///Users/charlesmcvicker/code/workshop-transcription/from_gsheet_to_wav2vec2_files.py).

## Directory Structure

The script outputs dataset files inside the sandbox folder within your Google Drive installation folder:

```
[installationFolder]/
└── [destinationSandbox]/
    ├── wav/                          # Folder containing individual segmented .wav files
    │   ├── [SPEAKER]-[PREFIX]-001.wav
    │   ├── [SPEAKER]-[PREFIX]-002.wav
    │   └── ...
    ├── [filePrefix]-train.csv        # Training split metadata file
    ├── [filePrefix]-valid.csv        # Validation split metadata file
    └── [filePrefix]-test.csv         # Testing split metadata file
```

- **`installationFolder`**: Top-level directory (e.g., `202606-cim-asr`).
- **`destinationSandbox`**: Individual sandbox directory (e.g., `sandbox-user`).
- **`filePrefix`**: Configured prefix for output files (defaults to `cim-wav2vec2` for Wav2Vec2).

---

## File Formats

Depending on the `software` option chosen when running the script, the output CSV files are structured differently.

### 1. Wav2Vec2 Format (`software="wav2vec2"`)

Output files:
- `[filePrefix]-train.csv`
- `[filePrefix]-valid.csv`
- `[filePrefix]-test.csv`

#### CSV Header
```csv
path,sentence
```

#### Field Specifications
* **`path`**: The absolute path to the `.wav` audio file on Google Drive (e.g. `/content/drive/MyDrive/202606-cim-asr/sandbox-user/wav/MSC-RRMSAvaikiP24-001.wav`).
* **`sentence`**: The cleaned, reformatted transcription corresponding to the audio segment.

#### Reformatting/Cleaning Rules applied to `sentence`
The transcript is normalized using the following cleaning steps:
1. Punctuation and special characters are removed: `[`, `]`, `"`, `(`, `)`, `.`, `\u0f7b` (Tibetan vowel sign), `_`, `|`, `》`, `?`, `!`, `/`, `,`, `-`, `?`, `<`, `…`, `>`.
2. Multiple spaces are collapsed to a single space.
3. Whitespace at the beginning and end of the transcript is trimmed.
4. Text is converted entirely to lowercase.

---

### 2. DeepSpeech Format (`software="ds"`)

Output files:
- `[filePrefix]-train.csv`
- `[filePrefix]-valid.csv`
- `[filePrefix]-test.csv`

#### CSV Header
```csv
wav_filename,wav_filesize,transcript
```

#### Field Specifications
* **`wav_filename`**: The absolute path to the `.wav` audio file (similar to `path` in Wav2Vec2).
* **`wav_filesize`**: The size of the `.wav` audio file in bytes.
* **`transcript`**: The cleaned, normalized transcription.

---

## Data Split & Filtering

The script filters rows from the source Google Sheet before partitioning them:
* **Code-Switching filter**: Ignores rows marked as code-switched (`codeSwitch == "1"`) if `useCodeSwitchedData` is disabled.
* **Doubtful Data filter**: Ignores rows marked as needing review (`needsFurtherCheck == "1"`) if `useDoutbfulData` is disabled.
* **Empty Transcripts**: Automatically filters out any rows where the transcription is empty.
* **Duration filter**: Limits audio segments to `maxWavDuration` (default 15 seconds) to prevent CUDA out-of-memory errors during model training.

The remaining filtered items are randomly shuffled and split based on configured percentages (default is **80% training**, **10% validation**, and **10% testing**).
