#!/bin/bash
# install_requirements.sh
# Installation script for Wav2Vec2 training dependencies locally using uv.

# Determine if we are inside a virtual environment, otherwise activate the local .venv
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    echo "=== Activating local .venv virtual environment ==="
    source "$PROJECT_DIR/.venv/bin/activate"
fi

echo "=== Installing Python dependencies using uv ==="
uv pip install -U \
    "numpy==1.26.4" \
    "transformers==4.44.2" \
    "datasets==2.21.0" \
    "accelerate==0.34.2" \
    "torchaudio" \
    "jiwer==3.0.4" \
    "evaluate==0.4.2" \
    "soundfile" \
    "librosa==0.10.2.post1" \
    "pyctcdecode==0.5.0" \
    "https://github.com/kpu/kenlm/archive/master.zip"

echo "=== Checking / Installing KenLM Build Tools ==="
if ! command -v cmake &> /dev/null; then
    echo "Warning: cmake is not installed. You may need to install it via Homebrew: 'brew install cmake'"
fi

KENLM_DIR="$SCRIPT_DIR/kenlm"

if [ ! -f "$KENLM_DIR/build/bin/lmplz" ]; then
    echo "Cloning and building KenLM locally..."
    git clone https://github.com/kpu/kenlm.git "$KENLM_DIR"
    mkdir -p "$KENLM_DIR/build"
    cd "$KENLM_DIR/build"
    cmake ..
    make -j2
    echo "KenLM built successfully at: $KENLM_DIR/build/bin/lmplz"
else
    echo "KenLM is already built at: $KENLM_DIR/build/bin/lmplz"
fi
