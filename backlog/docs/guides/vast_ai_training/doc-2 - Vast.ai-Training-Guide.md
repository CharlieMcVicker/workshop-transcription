---
id: doc-2
title: Vast.ai Training Guide
type: guide
created_date: '2026-06-30 23:22'
updated_date: '2026-06-30 23:52'
---
# Vast.ai Training Guide for Wav2Vec2

This guide outlines the process for conducting a Wav2Vec2 training run on a remote GPU instance using **vast.ai**.

## 1. Prebaking Dependencies into a Custom Docker Image (Optional but Recommended) 🐳

Instead of compiling KenLM and installing Python packages from scratch on every run, you can build a prebaked Docker image using the provided [Dockerfile](file:///Users/charlesmcvicker/code/workshop-transcription/Dockerfile). This saves time, reduces setup costs, and ensures a completely consistent environment.

### Building & Pushing the Image:
Run these commands locally (requires Docker installed and a Docker Hub account):
```bash
# 1. Build the image
docker build -t your-docker-username/asr-w2v2-lm-prebaked:latest .

# 2. Push to Docker Hub
docker push your-docker-username/asr-w2v2-lm-prebaked:latest
```

---

## 2. Configuring Private Registry Credentials on Vast.ai 🔒

If you push your Docker image to a private repository on Docker Hub, GitHub Container Registry (GHCR), or another provider, you must configure Vast.ai with your registry credentials so it can pull the image:

1. Log in to the [Vast.ai Console](https://vast.ai/).
2. Navigate to the **[Account settings page](https://vast.ai/console/account/)**.
3. Locate the **Docker Registry Login** section.
4. Enter your registry information:
   - **Registry**: For Docker Hub, use `index.docker.io` (or leave empty if it defaults to Docker Hub). For GitHub, use `ghcr.io`.
   - **Username**: Your registry username.
   - **Password**: Your password or Personal Access Token (for security, it's best to use a read-only Access Token).
5. Click **Log In / Save**.

---

## 3. Instance Selection and Launch

1. **Sign in to Vast.ai**: Go to [Vast.ai Console](https://vast.ai/).
2. **Template Configuration**:
   - Choose a GPU instance with **at least 24 GB of VRAM** (e.g., RTX 3090, RTX 4090, or A6000) to ensure large batch sizes and stability during CTC loss calculation.
   - Set the Docker Image to: `your-docker-username/asr-w2v2-lm-prebaked:latest` (or `rolandocoto/asr-w2v2-lm-python310` if you prefer to install manually).
   - Ensure the disk allocation is **at least 50 GB** to accommodate the raw audio files, datasets, intermediate KenLM outputs, and multiple model checkpoints.
   - Select **Use SSH** (recommended) or **Jupyter Lab** for remote access.
3. **Launch the Instance** and note down the connection details:
   - **IP Address**: `[IP]`
   - **Port**: `[PORT]`

---

## 4. Preparing and Transferring Data

Before copying the dataset, ensure the relative paths in your local folder match what the script expects. You will transfer the local CSV files and the audio directory to `/workspace/` inside the vast.ai instance.

Run the following command from your local machine (within this project root):

```bash
# Transfer the CSV splits, audio files, and training module to the vast.ai workspace
scp -P [PORT] -r \
  cim-wav2vec2-train.csv \
  cim-wav2vec2-valid.csv \
  cim-wav2vec2-test.csv \
  sentence_audio/ \
  transcription/ \
  root@[IP]:/workspace/
```

---

## 5. Remote Setup & Verification

SSH into your rented instance:

```bash
ssh -p [PORT] root@[IP]
```

Once connected, run the following verification checks:

1. **Verify GPU Status**:
   ```bash
   nvidia-smi
   ```
2. **Verify CUDA availability in PyTorch**:
   ```bash
   python3 -c "import torch; print('CUDA active:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
   ```
3. **Verify Dependencies / Build Tools (Only needed if NOT using the prebaked image)**:
   If some Python dependencies or KenLM builds are missing:
   ```bash
   pip install transformers datasets accelerate evaluate jiwer pyctcdecode torchaudio pandas numpy
   
   apt-get update && apt-get install -y build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
   git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build && cmake .. && make -j
   ```

---

## 6. Cost-Saving Testing / Smoke Testing

Before renting a high-end GPU (like an RTX 4090 or A6000) for a full run:

1. **Rent a Cheap GPU**:
   Rent an entry-level GPU instance on vast.ai, such as an **RTX 3060, RTX 2080 Ti, or GTX 1080 Ti**. These usually cost **$0.05 to $0.15/hour**.
2. **Run a Smoke Test (5-10 Steps)**:
   Run the training script with the `--max-steps` argument set to `10`. This compiles the graphs, loads the dataset, performs forward/backward passes, calculates CTC loss, and completes a checkpoint write to verify the entire pipeline runs without memory/driver errors:

   ```bash
   python3 -m transcription.training.train \
     --train-csv /workspace/cim-wav2vec2-train.csv \
     --valid-csv /workspace/cim-wav2vec2-valid.csv \
     --test-csv /workspace/cim-wav2vec2-test.csv \
     --audio-dir /workspace/sentence_audio \
     --output-dir /workspace/output_w2v2 \
     --lmplz-path /workspace/kenlm/build/bin/lmplz \
     --max-steps 10
   ```
   *(Note: If using the prebaked image, `lmplz` is in the system PATH, so you can omit `--lmplz-path` or set it to `lmplz`)*

---

## 7. Run the Full Training with Hugging Face Hub Backups

To run the training on your high-performance GPU instance and automatically push saving checkpoints to Hugging Face Hub (as a free private repository backup):

```bash
python3 -m transcription.training.train \
  --train-csv /workspace/cim-wav2vec2-train.csv \
  --valid-csv /workspace/cim-wav2vec2-valid.csv \
  --test-csv /workspace/cim-wav2vec2-test.csv \
  --audio-dir /workspace/sentence_audio \
  --output-dir /workspace/output_w2v2 \
  --lmplz-path /workspace/kenlm/build/bin/lmplz \
  --push-to-hub \
  --hub-model-id "your-hf-username/your-model-name" \
  --hub-token "your_hugging_face_write_token"
```

---

## 8. Managing Overnight Runs & Fail-safes 🛡️

### Stopped vs. Destroyed Instances:
* **Stopped**: If your credits run out, or your bid is outpriced, or you manually stop the instance, the machine is paused. **Your data inside `/workspace` is preserved**. You will only pay a very small disk storage fee (approx. $0.005/GB/month) while it is stopped. You can click **Start** later, boot it back up, and download your files.
* **Destroyed**: If you click **Destroy** (trash bin icon) on the Vast.ai console, the VM is deleted entirely, and all data is lost.

---

## 9. Retrieving Checkpoints and Results 📥

Since Wav2Vec2 checkpoints are large (approx. 4.5 GB per checkpoint folder), you should use a download command that supports resuming in case of network drops. 

Run one of these commands **from your local terminal** (not inside the SSH session):

### Option A: Using `rsync` (Recommended - supports resuming)
```bash
rsync -avzP -e 'ssh -p [PORT]' root@[IP]:/workspace/output_w2v2/ ./output_w2v2/
```
* `-a`: Archive mode (preserves file attributes and recursion).
* `-v`: Verbose output.
* `-z`: Compress file data during transfer.
* `-P`: Shows progress bar and keeps partially transferred files (enabling resume).
* `-e 'ssh -p [PORT]'`: Directs `rsync` to connect using the custom port.

### Option B: Using standard `scp`
```bash
scp -P [PORT] -r root@[IP]:/workspace/output_w2v2 ./
```
