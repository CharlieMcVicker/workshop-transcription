# Workshop Transcription & Wav2Vec2 Dataset Preparation

This repository contains tools for processing audio transcription files and preparing datasets for training speech recognition models like **Wav2Vec2** or **DeepSpeech**.

## Repository Structure

```
.
├── colab-script-rips/            # Localized training & legacy Google Colab scripts
│   ├── install_requirements.sh   # Installs workspace dependencies in .venv & compiles KenLM
│   ├── trainer_w2v2_local.py     # Localized adaptation of Wav2Vec2 trainer (fully offline)
│   ├── from_elan_to_wav_and_gsheet.py
│   ├── from_gsheet_to_wav2vec2_files.py
│   └── trainer_w2v2.py
│
├── sentence_audio/               # Folder containing individual segmented audio files (.wav) - (.gitignored)
│
├── sentence_audio.csv            # Source metadata CSV linking audio filenames to transcriptions
├── local_csv_to_wav2vec2.py      # Main pipeline script: converts local CSV and WAV files to training splits
│
├── cim-wav2vec2-train.csv        # Generated training split (path, sentence)
├── cim-wav2vec2-valid.csv        # Generated validation split (path, sentence)
├── cim-wav2vec2-test.csv         # Generated test split (path, sentence)
├── wav2vec2_format.md            # Detailed format specification of the output dataset splits
└── output_w2v2/                  # Training output artifacts (model checkpoints, ARPA files, results logs)
```

---

## Workspace Setup

A virtual environment is managed locally via `uv` for lightning-fast package resolution.

### 1. Install Dependencies and Build KenLM

Ensure you have `cmake` installed on your machine (`brew install cmake` on macOS). Then, run the installer:

```bash
./colab-script-rips/install_requirements.sh
```

This script will:

- Activate your local virtual environment (`.venv`).
- Install pinning versions of PyTorch (`torchaudio`), Hugging Face `transformers` and `datasets`, evaluation utilities (`jiwer`, `evaluate`), and decoder toolsets.
- Clone and compile KenLM locally at `colab-script-rips/kenlm/build/bin/lmplz`.

---

## Local Dataset Preparation Pipeline

The main script is [local_csv_to_wav2vec2.py](file:///Users/charlesmcvicker/code/workshop-transcription/local_csv_to_wav2vec2.py). It bypasses Google Colab and Google Sheets, processing the dataset locally using the files in your workspace.

### Key Features

- **Zero Dependencies**: Relies solely on Python's standard library (no `pandas` or `numpy` installation needed).
- **Metadata Extraction**: Reads the WAV file headers directly to verify existence and extract durations and file sizes.
- **Normalization**: Standardizes transcripts by removing punctuation/extra formatting (including asterisks `*`) and downcasing.
- **Duration Filtering**: Automatically drops long audio clips that could exceed GPU memory limits during training.
- **Shuffled Splits**: Shuffles the dataset deterministically with a random seed and partitions it into Train, Validation, and Test subsets.
- **Relative Path Output**: Keeps the file paths relative (e.g. `sentence_audio/FILENAME.wav`) so the dataset splits can be easily loaded in downstream trainers.

### Usage

Run the script from the project root:

```bash
./local_csv_to_wav2vec2.py [options]
```

### Options

```
options:
  -h, --help            show this help message and exit
  --csv CSV             Path to the input CSV file containing metadata
                        (default: sentence_audio.csv)
  --audio-dir AUDIO_DIR
                        Path to the directory containing audio files (default:
                        sentence_audio)
  --text-col TEXT_COL   Column name in the CSV to use for transcription text
                        (default: phonetic)
  --output-prefix OUTPUT_PREFIX
                        Prefix for the generated train/valid/test split CSV
                        files (default: cim-wav2vec2)
  --max-duration MAX_DURATION
                        Maximum audio duration (seconds) to include in the
                        splits (default: 15.0)
  --split SPLIT SPLIT SPLIT
                        Train, Validation, and Test split percentages summing
                        to 100 (default: 80 10 10)
  --seed SEED           Random seed for reproducibility when shuffling dataset
                        (default: 42)
```

---

## Localized Model Training

The script [trainer_w2v2_local.py](file:///Users/charlesmcvicker/code/workshop-transcription/colab-script-rips/trainer_w2v2_local.py) allows offline, local training of the speech-to-text pipeline.

### Steps Undertaken in Training

1. **Pre-processing**: Reads the local train, validation, and test CSV files.
2. **Vocabulary Building**: Analyzes character frequency across the dataset and writes `vocab.json`.
3. **Language Modeling**: Feeds the dataset transcripts into `lmplz` to generate a 4-gram ARPA language model. It corrects ARPA formatting so it is fully compatible with `pyctcdecode`.
4. **HuggingFace Dataset mapping**: Resolves file path mappings to sound clip structures correctly.
5. **Fine-Tuning**: Trains the XLS-R model using the HuggingFace `Trainer`.
6. **Decoding Evaluation**: Runs evaluations on all saved checkpoints, comparing greedy CTC search vs. KenLM language-model-guided search, and promotes the best model checkpoint.

### Running the Trainer

To run the training script:

```bash
uv run python colab-script-rips/trainer_w2v2_local.py
```

_Note for Apple Silicon Users:_
If you are using the default stable PyTorch release, the MPS backend does not natively support the `ctc_loss` operator, requiring you to enable CPU fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python colab-script-rips/trainer_w2v2_local.py
```

However, if you have installed **PyTorch Nightly** (specifically version `2.14.0.dev20260630` or newer), native MPS acceleration is supported for CTC loss out-of-the-box, allowing you to run training fully on your GPU without the CPU fallback environment variable:

```bash
uv run python colab-script-rips/trainer_w2v2_local.py
```

## What to run on VM

```zsh
  python3 /workspace/trainer_w2v2_local.py \
      --train-csv /workspace/cim-wav2vec2-train.csv \
      --valid-csv /workspace/cim-wav2vec2-valid.csv \
      --test-csv /workspace/cim-wav2vec2-test.csv \
      --audio-dir /workspace/sentence_audio \
      --output-dir /workspace/output_w2v2 \
      --lmplz-path lmplz \
      --push-to-hub \
      --hub-model-id "charliemcvicker/asr-cherokee" \
      --hub-token "your_hf_write_token"
```

```zsh
   docker run -it wav2vec2-trainer:local \
      python3 /workspace/trainer_w2v2_local.py \
        --max-steps 3 \
        --push-to-hub \
        --hub-model-id "charliemcvicker/asr-cherokee" \
        --hub-token "your_hf_write_token"
```
