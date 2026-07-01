# Workshop Transcription & Wav2Vec2 Dataset Preparation

This repository contains tools for processing audio transcription files, segmenting raw recordings, labeling low-confidence predictions, and preparing/training speech recognition models like **Wav2Vec2** locally or in Docker containers.

## Repository Structure

```
workshop-transcription/
├── src/                          # All Python source code
│   └── transcription/            # Main package
│       ├── audio/                # Audio preprocessing and segmentation
│       │   ├── segment.py        # Hyperparameter sweep / segment evaluation
│       │   └── extract.py        # Extract audio segments and write manifest
│       ├── inference/            # Model inference and active labeling
│       │   ├── single.py         # Run inference on a single audio file
│       │   ├── batch.py          # Batch inference on a directory of WAVs
│       │   └── labeler.py        # Web-based interface for low-confidence labeling
│       ├── training/             # Training, evaluation, and data prep
│       │   ├── prepare_csv.py    # Prepare training splits from raw CSV
│       │   ├── train.py          # Offline/local/remote Wav2Vec2 training
│       │   ├── evaluate_checkpoint.py # Score checkpoint against test set
│       │   └── evaluate_revisions.py  # Score Git revisions from Hugging Face
│       └── utils/                # Utilities and checks
│           ├── tone_normalization.py  # Normalizes tones in transcripts
│           └── verify_mps.py          # Verifies PyTorch MPS backend support
│
├── data/                         # Dedicated data directories
│   ├── raw/                      # Raw unsegmented audio files and datasets
│   ├── processed/                # Segmented audio folders and dataset CSV splits
│   └── results/                  # Inference outputs and evaluation results
│
├── archive/                      # Installer files and ZIP backups
│
└── Dockerfile                    # Container configuration file
```

---

## Workspace Setup

A virtual environment is managed locally via `uv` or standard Python `venv`.

1. **Activate virtual environment** (e.g. `.venv`).
2. **Install requirements**: Make sure you have `cmake` and `ffmpeg` installed on your system. Run:
   ```bash
   uv pip install -r requirements.txt
   ```
3. Set your python path to include `src/`:
   ```bash
   export PYTHONPATH="src:${PYTHONPATH}"
   ```

---

## Packaged Entrypoints & Main Use Cases

All logic is package-based. Always run commands from the project root directory with `PYTHONPATH=src` (or after exporting it).

### 1. Audio Segmentation & Extraction
Analyze audio files to find speaking segments, or extract them into segmented WAV files.

- **Evaluate segmentation settings (Hyperparameter Sweep)**:
  ```bash
  python3 -m transcription.audio.segment data/raw/Bessie-Summerfield.wav --sweep
  ```
- **Extract segments to disk**:
  ```bash
  python3 -m transcription.audio.extract data/raw/Bessie-Summerfield.wav --out-dir data/processed/segments
  ```

### 2. Dataset Preparation
Split your local CSV dataset and WAV directory into Train, Validation, and Test partitions.
```bash
python3 -m transcription.training.prepare_csv \
  --csv data/processed/sentence_audio.csv \
  --audio-dir data/processed/sentence_audio \
  --output-prefix data/processed/cim-wav2vec2
```

### 3. Local Model Training
Train the Wav2Vec2 model offline.
```bash
python3 -m transcription.training.train \
  --train-csv data/processed/cim-wav2vec2-train.csv \
  --valid-csv data/processed/cim-wav2vec2-valid.csv \
  --test-csv data/processed/cim-wav2vec2-test.csv \
  --audio-dir data/processed/sentence_audio \
  --output-dir output_w2v2 \
  --epochs 50
```

### 4. Running Inference
- **Single file inference**:
  ```bash
  python3 -m transcription.inference.single data/raw/Bessie-Summerfield-2.wav \
    --checkpoint charliemcvicker/asr-cherokee
  ```
- **Batch inference on a directory**:
  ```bash
  python3 -m transcription.inference.batch data/processed/segments \
    --checkpoint charliemcvicker/asr-cherokee \
    --output data/results/batch_inference_results.csv
  ```

### 5. Web-based Active Labeler
Run a local labeling UI to manually review and label low-confidence audio segments:
```bash
python3 -m transcription.inference.labeler --port 8000
```
Then visit `http://localhost:8000/` in your browser. It automatically pulls data from `data/results/batch_inference_results.csv` and saves human-labeled transcripts to `data/processed/train_labeled.csv`.

### 6. Model Evaluation
- **Evaluate local checkpoint**:
  ```bash
  python3 -m transcription.training.evaluate_checkpoint \
    --test-csv data/processed/cim-wav2vec2-test.csv \
    --audio-dir data/processed/sentence_audio \
    --checkpoint remote_output_w2v2/checkpoint-800
  ```
- **Evaluate multiple Hugging Face commits/revisions**:
  ```bash
  python3 -m transcription.training.evaluate_revisions \
    --revisions-csv data/results/revisions_to_test.tsv \
    --test-csv data/processed/cim-wav2vec2-test.csv \
    --audio-dir data/processed/sentence_audio \
    --output-csv data/results/revision_scores.csv
  ```

---

## Docker Execution

The project contains a prebaked Docker image that compiles KenLM, installs all PyTorch/CUDA dependencies, and downloads the base model checkpoint.

To run a container and mount your workspace for training (with Hugging Face authentication to push results):

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  -e HF_TOKEN="your_hugging_face_token" \
  <image_name> \
  python3 -m transcription.training.train \
    --push-to-hub \
    --hub-model-id "your-username/asr-cherokee"
```
