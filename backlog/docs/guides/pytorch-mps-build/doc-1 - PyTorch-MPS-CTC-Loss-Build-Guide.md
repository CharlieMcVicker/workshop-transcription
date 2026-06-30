---
id: doc-1
title: PyTorch MPS CTC Loss Build Guide
type: guide
created_date: '2026-06-30 21:57'
---

# PyTorch Build from Source Guide (MPS CTC Loss Support)

> [!NOTE]
> **Recommended Alternative:** Instead of building PyTorch from source, you can install the pre-compiled **PyTorch Nightly** build which has the MPS CTC Loss support (merged in PR #176778) out-of-the-box. We have successfully installed the nightly build `torch-2.14.0.dev20260630` using:
> ```bash
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
> ```

This guide details the steps required to build PyTorch from source (`main` branch) on Apple Silicon (macOS) if you need a custom build, or if you want to understand the system dependencies.

---


## 📋 System Prerequisites

Ensure your macOS development environment meets the following conditions:

1. **Hardware:** Mac with Apple Silicon (M-series chip: M1, M2, M3, M4, etc.).
2. **OS Version:** macOS 12.3 or later (macOS 14.0+ is recommended for optimal MPS support).
3. **Xcode Command Line Tools:** Install the compiler and essential SDK headers by running:
   ```bash
   xcode-select --install
   ```
4. **Package Manager:** Homebrew is recommended to install build tools.
   ```bash
   brew install cmake ninja git
   ```

---

## 🐍 Step-by-Step Build & Installation

### 1. Set Up Python Environment

It is highly recommended to isolate the build environment using `conda` or virtual environments (`venv`/`uv`).

#### Option A: Using Conda (Recommended)
```bash
# Create and activate environment
conda create -n pytorch-build python=3.10 -y
conda activate pytorch-build

# Install cmake, ninja, and base build requirements
conda install cmake ninja pyyaml setuptools cffi typing_extensions -y
```

#### Option B: Using standard venv
```bash
python3 -m venv pytorch-build-env
source pytorch-build-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install cmake ninja pyyaml cffi typing_extensions
```

### 2. Clone the Repository

Clone PyTorch with all submodules. Since PyTorch has many dependencies vendorized as submodules, the `--recursive` flag is required.

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# If you have already cloned the repository without submodules, run:
git submodule update --init --recursive
```

### 3. Install Python Dependencies

Install the remaining libraries required to build PyTorch:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Define the build configuration before running the compilation.
* `USE_MPS=1` enables the Metal Performance Shaders backend (enabled by default on macOS but good to specify).
* `CMAKE_PREFIX_PATH` helps CMake locate conda/environment libraries.
* `MACOSX_DEPLOYMENT_TARGET` should match your macOS version (e.g., `14.0`).

```bash
# If using Conda:
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# Enable MPS and set deployment target
export USE_MPS=1
export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)

# Optional flags to speed up building or bypass non-essential modules
export BUILD_CAFFE2=0
export USE_ONNX=0
```

### 5. Build and Install PyTorch

Compile and install PyTorch in development mode (`develop` compiles the binaries and creates links to the source tree, making it easier to rebuild if needed).

```bash
# Build in development/editable mode
python setup.py develop
```

*Note: The initial compilation can take anywhere from 20 minutes to over an hour depending on your machine's processor and RAM.*

---

## 🔍 Verification

Once the build is complete, you can verify that MPS is working and that `ctc_loss` is supported on the MPS backend:

Create a script `verify_mps_ctc.py`:

```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"MPS Backend Available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    
    # 1. Test basic MPS tensor allocation
    x = torch.ones(2, 2, device=device)
    print(f"Basic allocation check: {x}")
    
    # 2. Test CTC Loss on MPS (requires PR #176778)
    log_probs = torch.randn(50, 16, 20, device=device).log_softmax(2).requires_grad_()
    targets = torch.randint(1, 20, (16, 30), dtype=torch.long, device=device)
    input_lengths = torch.full((16,), 50, dtype=torch.long, device=device)
    target_lengths = torch.randint(10, 30, (16,), dtype=torch.long, device=device)
    
    try:
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        print("✅ Successfully computed CTC loss and backward pass on MPS!")
        print(f"Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Failed to run CTC loss on MPS: {e}")
else:
    print("❌ MPS is not available on this device/installation.")
```

---

## 🛠️ Common Pitfalls & Troubleshooting

### Xcode Command Line Tools / SDK mismatch
If you get errors indicating missing compiler headers or SDK issues:
* Reset your active Xcode path:
  ```bash
  sudo xcode-select -r
  ```

### Out of Memory (OOM) during build
Compiling large C++ files in parallel can consume a lot of memory. If compilation crashes:
* Limit the number of parallel build jobs using `MAX_JOBS` environment variable:
  ```bash
  export MAX_JOBS=4  # Adjust based on your Mac's memory (e.g., use 4 for 8GB/16GB models)
  python setup.py develop
  ```

### Conda Linker Conflict
If Conda’s linker shadows the system linker and causes compilation errors:
* Temporarily rename or remove Conda's `ld` file from the path, or use a standard Python `venv` instead of Anaconda.
