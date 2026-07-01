import os
import sys
import random
import subprocess
import wave
import contextlib
import pandas as pd
from datetime import date
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
import tempfile
import uuid

# Refresh PATH from Registry (Windows) so the server picks up new winget installations (like ffmpeg) without restarting
try:
    import winreg
    for hkey, subkey in [
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment")
    ]:
        try:
            with winreg.OpenKey(hkey, subkey, 0, winreg.KEY_READ) as key:
                path_val, _ = winreg.QueryValueEx(key, "Path")
                for path_dir in path_val.split(os.path.pathsep):
                    path_dir_expanded = os.path.expandvars(path_dir)
                    if path_dir_expanded and path_dir_expanded not in os.environ["PATH"]:
                        os.environ["PATH"] += os.path.pathsep + path_dir_expanded
        except Exception:
            pass
except Exception:
    pass


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppConfig:
    SANDBOX_DIR = os.path.abspath("sandbox-user")

    @classmethod
    def get_elan_dir(cls): return os.path.join(cls.SANDBOX_DIR, "processed-elan-files")

    @classmethod
    def get_wav_dir(cls): return os.path.join(cls.SANDBOX_DIR, "wav")

    @classmethod
    def get_metadata_csv(cls): return os.path.join(cls.SANDBOX_DIR, "wav-metadata.csv")

    @classmethod
    def get_inf_dir(cls): return os.path.join(cls.SANDBOX_DIR, "audiofiles-to-transcribe")

    @classmethod
    def get_model_dir(cls): return os.path.join(cls.SANDBOX_DIR, "wav2vec2-model")

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.get_elan_dir(), exist_ok=True)
        os.makedirs(cls.get_wav_dir(), exist_ok=True)
        os.makedirs(cls.get_inf_dir(), exist_ok=True)
        os.makedirs(cls.get_model_dir(), exist_ok=True)

AppConfig.ensure_dirs()

class ProcessElanRequest(BaseModel):
    txt_file: str
    wav_file: str
    gender: str

class GenerateSplitsRequest(BaseModel):
    train_pct: int
    valid_pct: int
    test_pct: int
    max_duration: int
    file_prefix: str
    use_code_switched: bool
    use_doubtful: bool

vad_model = None

def get_vad_model():
    global vad_model
    if vad_model is None:
        try:
            from speechbrain.inference.VAD import VAD
        except ImportError:
            from speechbrain.pretrained import VAD
        
        import torch
        device = "cpu" # Force CPU for VAD to avoid CUDA kernel arch mismatches
        
        vad_model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            run_opts={"device": device}
        )
    return vad_model

@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join(AppConfig.SANDBOX_DIR, "audiofiles-to-transcribe", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path)

class VADRequest(BaseModel):
    filename: str

@app.post("/api/vad_segments")
def get_vad_segments(req: VADRequest):
    try:
        path = os.path.join(AppConfig.SANDBOX_DIR, "audiofiles-to-transcribe", req.filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Audio file not found")
            
        vad = get_vad_model()
        
        temp_id = str(uuid.uuid4())
        temp_wav = f"temp_vad_{temp_id}.wav"
        
        cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", temp_wav]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {process.stderr}")
            
        boundaries = vad.get_speech_segments(temp_wav)
        
        segments = []
        for seg in boundaries:
            segments.append({"start": round(float(seg[0]), 3), "end": round(float(seg[1]), 3)})
            
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        return {"segments": segments}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class Region(BaseModel):
    start: float
    end: float
    text: str = ""

class SaveElanRequest(BaseModel):
    regions: list[Region]
    filename: str
    
@app.post("/api/save_elan")
def save_elan(req: SaveElanRequest):
    try:
        out_path = os.path.join(AppConfig.get_elan_dir(), req.filename)
        lines = []
        for r in req.regions:
            text = r.text if r.text else "TBD"
            lines.append(f"default\tspeaker\t{r.start:.3f}\t{r.end:.3f}\t{text}")
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
        return {"message": "Success", "filepath": out_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
def get_files():
    txt_files = [f for f in os.listdir(AppConfig.get_elan_dir()) if f.lower().endswith(".txt")]
    
    wav_files = []
    if os.path.exists(AppConfig.get_elan_dir()):
        wav_files.extend([f for f in os.listdir(AppConfig.get_elan_dir()) if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".mov", ".avi", ".webm", ".opus"))])
        
    inf_dir = AppConfig.get_inf_dir()
    if os.path.exists(inf_dir):
        wav_files.extend([f for f in os.listdir(inf_dir) if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".mov", ".avi", ".webm", ".opus"))])
        
    wav_files = list(set(wav_files))
    
    return {"txt_files": txt_files, "wav_files": wav_files}

@app.get("/api/inference_files")
def get_inference_files():
    inf_dir = AppConfig.get_inf_dir()
    if not os.path.exists(inf_dir):
        return {"wav_files": []}
    
    wav_files = [f for f in os.listdir(inf_dir) if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".mov", ".avi", ".webm", ".opus"))]
    return {"wav_files": wav_files}

def countDigits(num): return len(str(num))
def addZeros(num, max_num): return str(num).zfill(countDigits(max_num))
def reformatTranscription(text):
    punctuation = [ "[", "]", "\"", "(", ")", ".", "\u0f7b", "_", "|", "》", "?", "!", "/", ',', '-', '?', '<', '…', '>' ]
    for p in punctuation: text = text.replace(p, " ")
    for _ in range(5): text = text.replace("  "," ")
    return text.strip().lower()

@app.post("/api/process_elan")
def process_elan(req: ProcessElanRequest):
    try:
        tab_path = os.path.join(AppConfig.get_elan_dir(), req.txt_file)
        wav_path = os.path.join(AppConfig.get_elan_dir(), req.wav_file)
        
        if not os.path.exists(wav_path):
            inf_dir = AppConfig.get_inf_dir()
            wav_path = os.path.join(inf_dir, req.wav_file)
            
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=404, detail="wav file not found")
        
        if not os.path.exists(tab_path):
            raise HTTPException(status_code=404, detail="txt file not found")
            
        stem = os.path.splitext(req.txt_file)[0]
        speaker_code = stem.rsplit("-", 1)[1].upper() if "-" in stem else stem.upper()
        audio_prefix = os.path.splitext(req.wav_file)[0]

        try:
            with open(tab_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(tab_path, 'r', encoding='utf-16') as f:
                lines = f.readlines()
            
        timeStart, timeEnd, transcriptions = [], [], []
        
        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 4:
                try:
                    start = float(parts[2])
                    end = float(parts[3])
                except ValueError:
                    continue
                trans = parts[-1]
                dur = end - start
                if dur > 0 and reformatTranscription(trans):
                    timeStart.append(start)
                    timeEnd.append(end)
                    transcriptions.append(trans)
                    
        if not timeStart:
            raise HTTPException(status_code=400, detail="No valid annotations found.")
            
        points = []
        filenames = []
        
        for i, (start, end) in enumerate(zip(timeStart, timeEnd)):
            tempName1 = f"{speaker_code}-{audio_prefix}-{addZeros(i+1, len(timeStart))}.wav"
            tempName2 = f"{speaker_code}-{audio_prefix}-{addZeros(i+2, len(timeStart))}.wav"
            if points and points[-1] == start:
                points.append(end)
                filenames.append(tempName1)
            else:
                points.append(start)
                points.append(end)
                filenames.append(tempName2)
                
        pointSeq = ','.join(str(p) for p in points)
        zeros = countDigits(len(timeStart))
        
        outFileName = os.path.join(AppConfig.get_wav_dir(), f"{speaker_code}-{audio_prefix}-%0{zeros}d.wav")
        
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-f", "segment", "-ac", "1", "-ar", "16000", "-async", "1", "-segment_times", pointSeq, outFileName]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"FFmpeg error: {stderr}")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="FFmpeg is not installed or not in PATH.")
            
        todaysDate = date.today().strftime("%Y%m%d")
        
        try:
            if os.path.exists(AppConfig.get_metadata_csv()):
                df = pd.read_csv(AppConfig.get_metadata_csv())
            else:
                columns = ["wav_filename", "sandbox", "date", "speakerCode", "speakerGender", "wav_filesize", "duration_seconds", "codeSwitch", "needsFurtherCheck", "transcript_clean", "transcript"]
                df = pd.DataFrame(columns=columns)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading metadata CSV: {str(e)}")
            
        new_rows = []
        for i in range(len(timeStart)):
            fname = filenames[i]
            full_path = os.path.join(AppConfig.get_wav_dir(), fname)
            
            duration, size = 0, 0
            if os.path.exists(full_path):
                try:
                    with contextlib.closing(wave.open(full_path, 'r')) as f:
                        duration = f.getnframes() / float(f.getframerate())
                    size = os.path.getsize(full_path)
                except Exception:
                    pass
                
            new_rows.append({
                "wav_filename": fname,
                "sandbox": os.path.basename(AppConfig.SANDBOX_DIR).replace("sandbox-", ""),
                "date": todaysDate,
                "speakerCode": speaker_code,
                "speakerGender": req.gender,
                "wav_filesize": size,
                "duration_seconds": duration,
                "codeSwitch": "",
                "needsFurtherCheck": "",
                "transcript_clean": reformatTranscription(transcriptions[i]),
                "transcript": transcriptions[i]
            })
            
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(AppConfig.get_metadata_csv(), index=False)
        
        return {"message": "Success", "rows_added": len(new_rows)}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/generate_splits")
def generate_splits(req: GenerateSplitsRequest):
    if req.train_pct + req.valid_pct + req.test_pct != 100:
        raise HTTPException(status_code=400, detail="Percentages must sum to 100")
        
    if not os.path.exists(AppConfig.get_metadata_csv()):
        raise HTTPException(status_code=404, detail="Metadata CSV not found. Process ELAN first.")
        
    df = pd.read_csv(AppConfig.get_metadata_csv())
    
    if not req.use_code_switched and 'codeSwitch' in df.columns:
        df = df[df['codeSwitch'] != "1"]
        df = df[df['codeSwitch'] != 1]
    if not req.use_doubtful and 'needsFurtherCheck' in df.columns:
        df = df[df['needsFurtherCheck'] != "1"]
        df = df[df['needsFurtherCheck'] != 1]
        
    if 'transcript' in df.columns:
        df = df[df['transcript'].notna() & (df['transcript'] != "")]
    
    if 'duration_seconds' in df.columns:
        df['duration_seconds'] = df['duration_seconds'].astype(str).str.replace(',', '.').astype(float)
        df = df[df['duration_seconds'] <= req.max_duration]
        
    if 'wav_filename' not in df.columns or 'transcript' not in df.columns:
         raise HTTPException(status_code=400, detail="CSV missing required columns")

    paths = [os.path.join(AppConfig.get_wav_dir(), row['wav_filename']).replace("\\", "/") for _, row in df.iterrows()]
    sentences = df['transcript'].tolist()
    
    combined = list(zip(paths, sentences))
    random.shuffle(combined)
    
    n_total = len(combined)
    n_train = int(round(n_total * (req.train_pct / 100.0)))
    n_valid = int(round(n_total * (req.valid_pct / 100.0)))
    
    train_data = combined[:n_train]
    valid_data = combined[n_train:n_train+n_valid]
    test_data = combined[n_train+n_valid:]
    
    def save_split(data, filename):
        out_df = pd.DataFrame(data, columns=["path", "sentence"])
        out_path = os.path.join(AppConfig.SANDBOX_DIR, filename)
        out_df.to_csv(out_path, index=False)
        return len(data)
        
    t_cnt = save_split(train_data, f"{req.file_prefix}-train.csv")
    v_cnt = save_split(valid_data, f"{req.file_prefix}-valid.csv")
    te_cnt = save_split(test_data, f"{req.file_prefix}-test.csv")
    
    return {"message": "Splits generated successfully", "train": t_cnt, "valid": v_cnt, "test": te_cnt}

@app.get("/api/devices")
def get_devices():
    try:
        import torch
        devices = [{"id": "cpu", "name": "CPU"}]
        if torch.cuda.is_available():
            devices.append({"id": "all", "name": "All GPUs"})
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append({"id": f"cuda:{i}", "name": f"GPU {i}: {name}"})
        return {"devices": devices}
    except Exception:
        return {"devices": [{"id": "cpu", "name": "CPU (Default)"}]}

@app.get("/api/checkpoints")
def get_checkpoints():
    model_dir = AppConfig.get_model_dir()
    if not os.path.exists(model_dir):
        return {"checkpoints": []}
    
    checkpoints = []
    for d in os.listdir(model_dir):
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, d)):
            checkpoints.append(d)
    
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1, reverse=True)
    return {"checkpoints": checkpoints}

class TrainRequest(BaseModel):
    train_csv: str
    valid_csv: str
    test_csv: str
    epochs: int
    ngrams: int
    run_id: str
    lang_prefix: str
    lmplz_path: str = None
    device: str = "all"

@app.post("/api/train")
def train_model(req: TrainRequest):
    cmd = [
        sys.executable, "run_training.py",
        "--sandbox", AppConfig.SANDBOX_DIR,
        "--train_csv", req.train_csv,
        "--valid_csv", req.valid_csv,
        "--test_csv", req.test_csv,
        "--epochs", str(req.epochs),
        "--ngrams", str(req.ngrams),
        "--run_id", req.run_id,
        "--lang_prefix", req.lang_prefix
    ]
    if req.lmplz_path:
        cmd.extend(["--lmplz_path", req.lmplz_path])
        
    env = os.environ.copy()
    if req.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif req.device.startswith("cuda:"):
        idx = req.device.split(":")[1]
        env["CUDA_VISIBLE_DEVICES"] = idx

    try:
        # Run training in a separate process, non-blocking
        subprocess.Popen(cmd, env=env)
        return {"message": "Training started in the background. Check terminal for logs."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TranscribeLongRequest(BaseModel):
    audio_file: str
    checkpoint: str

@app.post("/api/transcribe_long")
def transcribe_long(req: TranscribeLongRequest):
    # Assuming audio files are uploaded or present in audiofiles-to-transcribe
    audio_path = os.path.join(AppConfig.SANDBOX_DIR, "audiofiles-to-transcribe", req.audio_file)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    model_dir = os.path.join(AppConfig.SANDBOX_DIR, "wav2vec2-model", req.checkpoint)
    out_tsv = os.path.join(AppConfig.SANDBOX_DIR, "audiofiles-to-transcribe", req.audio_file.rsplit('.', 1)[0] + ".tsv")
    
    cmd = [
        sys.executable, "run_inference.py",
        "--mode", "long",
        "--audio_file", audio_path,
        "--model_dir", model_dir,
        "--output_tsv", out_tsv
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {res.stderr}")
        return {"message": "Transcription complete", "tsv_file": out_tsv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File, Form

@app.post("/api/transcribe_mic")
def transcribe_mic(
    checkpoint: str = Form(...),
    audio: UploadFile = File(...)
):
    model_dir = os.path.join(AppConfig.SANDBOX_DIR, "wav2vec2-model", checkpoint)
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
        
    # Save uploaded audio to temp file
    temp_id = str(uuid.uuid4())
    temp_wav = os.path.join(tempfile.gettempdir(), f"mic_{temp_id}.wav")
    
    with open(temp_wav, "wb") as f:
        f.write(audio.file.read())
        
    cmd = [
        sys.executable, "run_inference.py",
        "--mode", "short",
        "--audio_file", temp_wav,
        "--model_dir", model_dir
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {res.stderr}")
            
        # Parse stdout for TRANSCRIPTION: ...
        transcription = ""
        for line in res.stdout.splitlines():
            if line.startswith("TRANSCRIPTION:"):
                transcription = line.split("TRANSCRIPTION:")[1].strip()
                break
                
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

@app.get("/api/config/folder")
def get_folder():
    return {"folder": AppConfig.SANDBOX_DIR}

@app.post("/api/config/select_folder")
def select_folder():
    cmd = [
        sys.executable, "-c",
        "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); folder = filedialog.askdirectory(title='Select Base Folder'); print(folder);"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    selected_folder = res.stdout.strip()
    if selected_folder:
        AppConfig.SANDBOX_DIR = os.path.abspath(selected_folder)
        AppConfig.ensure_dirs()
    return {"folder": AppConfig.SANDBOX_DIR}
