#=================================================
# Package installation.
# Run this once per session. If this cell restarts
# the runtime, just run it again and then continue.
#=================================================
# The current Colab runtime ships numpy 2.x, and torch / pandas / speechbrain
# are all built for it. We must keep numpy on the 2.x line, otherwise you get
# errors like "No module named 'numpy.strings'".

# Keep numpy where Colab put it (2.x).
# !pip install -q "numpy>=2.0"

# SpeechBrain (VAD) + transformers. These are compatible with numpy 2.x.
# !pip install -q "speechbrain>=1.0" "transformers>=4.40,<5"

# pyctcdecode 0.5.0 *declares* numpy<2 in its metadata, which conflicts with
# numpy 2.x. Its actual code runs fine on numpy 2.x, so we install it WITHOUT
# letting that pin downgrade numpy (--no-deps) and add its real dependency
# (pygtrie) ourselves.
# !pip install -q "pygtrie>=2.1,<3.0"
# !pip install -q --no-deps "pyctcdecode==0.5.0"

# KenLM backend used by the language-model decoder.
# !pip install -q https://github.com/kpu/kenlm/archive/master.zip

# Final guard: make sure nothing above dragged numpy back down to 1.x.
# !pip install -q "numpy>=2.0"

#=================================================
# If numpy was already loaded at the wrong version
# earlier in this session, restart the runtime once
# so the corrected version is picked up.
#=================================================
import numpy as _np
if int(_np.__version__.split(".")[0]) < 2:
    print(
        "numpy was downgraded earlier in this session. Restarting the "
        "runtime to load the correct version...\n"
        ">>> After it restarts, run THIS cell again, then continue. <<<"
    )
    import os, sys, time
    sys.stdout.flush(); time.sleep(1)
    os.kill(os.getpid(), 9)
else:
    print("numpy", _np.__version__, "OK \u2014 continue to the next cells.")


# ================================================================
# Mount the Google Drive onto the virtual computer
# ================================================================

# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

# ============================================================
#  LONG FILE INFERENCE SETUP
# ============================================================
#  Run this cell, choose the installation folder and sandbox, then
#  pick the audio file (from the sandbox's "audiofiles-to-transcribe"
#  folder) and the model checkpoint (from its "wav2vec2-model"
#  folder). Set the duration limits and GPU option, then click
#  "✅ Confirm selections".
#
#  You normally won't edit anything here. If your Drive lives
#  somewhere unusual, adjust MYDRIVE_ROOT below.
# ------------------------------------------------------------

# # from google.colab import drive; # drive.mount('/content/drive')   # <- uncomment if using Drive

import os
import ipywidgets as widgets
from IPython.display import display

# ----- Config -----------------------------------------------
MYDRIVE_ROOT = "/content/drive/MyDrive"           # where installation folders live
DEFAULT_INSTALLATION_FOLDER = "202606-cim-asr"    # default selection if present
AUDIO_SUBFOLDER = "audiofiles-to-transcribe"      # holds the files to transcribe
MODEL_SUBFOLDER = "wav2vec2-model"                # holds the checkpoint-* folders
AUDIO_EXTS = (".wav", ".mp4", ".mp3", ".m4a", ".flac",
              ".ogg", ".aac", ".mov", ".avi", ".webm", ".opus")
# ------------------------------------------------------------


# ---------- helpers: folders on disk ------------------------
def list_installations():
    if not os.path.isdir(MYDRIVE_ROOT):
        return []
    return sorted(d for d in os.listdir(MYDRIVE_ROOT)
                  if os.path.isdir(os.path.join(MYDRIVE_ROOT, d)))

def sandbox_root():
    inst = install_dd.value
    return os.path.join(MYDRIVE_ROOT, inst) if inst else ""

def list_sandboxes():
    root = sandbox_root()
    if not root or not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))

def sandbox_path(sandbox):
    root = sandbox_root()
    return os.path.join(root, sandbox) if (root and sandbox) else ""

def find_subfolder(sandbox, name):
    """Locate a named folder inside the sandbox (exact, then case-insensitive,
    then nested up to 3 levels). Returns its path or None."""
    base = sandbox_path(sandbox)
    if not base or not os.path.isdir(base):
        return None
    direct = os.path.join(base, name)
    if os.path.isdir(direct):
        return direct
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p) and d.lower() == name.lower():
            return p
    for dirpath, dirnames, files in os.walk(base):
        if dirpath[len(base):].count(os.sep) >= 3:
            dirnames[:] = []
        for d in list(dirnames):
            if d.lower() == name.lower():
                return os.path.join(dirpath, d)
    return None


# ---------- helpers: audio files & checkpoints --------------
def list_audio(sandbox):
    folder = find_subfolder(sandbox, AUDIO_SUBFOLDER)
    if not folder:
        return None, []
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(AUDIO_EXTS))
    return folder, files

def _checkpoint_step(name):
    tail = name.split("-")[-1]
    return int(tail) if tail.isdigit() else -1

def list_checkpoints(sandbox):
    folder = find_subfolder(sandbox, MODEL_SUBFOLDER)
    if not folder:
        return None, []
    subs = [d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))]
    cks = [d for d in subs if d.lower().startswith("checkpoint")]
    items = cks if cks else subs
    # latest checkpoint first (highest step number)
    items = sorted(items, key=_checkpoint_step, reverse=True)
    return folder, items


# ---------- build the widgets -------------------------------
label_style = {"description_width": "235px"}
row_layout  = widgets.Layout(width="660px")

_inst_options = list_installations()
install_dd = widgets.Dropdown(
    description="Installation folder:", options=_inst_options,
    value=(DEFAULT_INSTALLATION_FOLDER if DEFAULT_INSTALLATION_FOLDER in _inst_options
           else (_inst_options[0] if _inst_options else None)),
    style=label_style, layout=row_layout)

sandbox_dd = widgets.Dropdown(description="Sandbox (currentSandbox):",
                              options=[], style=label_style, layout=row_layout)

audio_dd = widgets.Dropdown(description="Audio file (audioFileName):",
                            style=label_style, layout=row_layout)
audio_lbl = widgets.HTML()

ckpt_dd = widgets.Dropdown(description="Checkpoint (modelCheckpointToUse):",
                           style=label_style, layout=row_layout)
ckpt_lbl = widgets.HTML()

mindur_tb = widgets.BoundedIntText(description="Min segment duration (ms):",
                                   value=100, min=0, max=100000,
                                   style=label_style, layout=row_layout)
maxdur_tb = widgets.BoundedIntText(description="Max WAV duration (s):",
                                   value=15, min=1, max=600,
                                   style=label_style, layout=row_layout)
gpu_cb = widgets.Checkbox(value=True, description="Use GPU (uncheck for slower CPU)",
                          indent=False)

confirm_btn = widgets.Button(description="✅ Confirm selections",
                             button_style="success",
                             layout=widgets.Layout(width="200px"))
status_out = widgets.Output()


# ---------- keep things in sync -----------------------------
def refresh_files(*_):
    sb = sandbox_dd.value

    audio_folder, audios = list_audio(sb)
    audio_dd.options = audios
    audio_dd.value = audios[0] if audios else None
    if not sb:
        audio_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No sandbox folders in "
            f"<code>{os.path.join(MYDRIVE_ROOT, str(install_dd.value))}</code>. "
            f"Pick a different installation folder.</span>")
    elif audio_folder and audios:
        audio_lbl.value = (
            f"<span style='color:#1a7f37'>🎧 {len(audios)} file(s) in "
            f"<code>{audio_folder}</code></span>")
    elif audio_folder:
        audio_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No audio files in "
            f"<code>{audio_folder}</code>.</span>")
    else:
        audio_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No <code>{AUDIO_SUBFOLDER}</code> "
            f"folder found in this sandbox.</span>")

    ckpt_folder, ckpts = list_checkpoints(sb)
    ckpt_dd.options = ckpts
    ckpt_dd.value = ckpts[0] if ckpts else None     # latest checkpoint
    if not sb:
        ckpt_lbl.value = ""
    elif ckpt_folder and ckpts:
        ckpt_lbl.value = (
            f"<span style='color:#1a7f37'>🧠 {len(ckpts)} checkpoint(s) in "
            f"<code>{ckpt_folder}</code> — latest selected</span>")
    elif ckpt_folder:
        ckpt_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No checkpoints in "
            f"<code>{ckpt_folder}</code>.</span>")
    else:
        ckpt_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No <code>{MODEL_SUBFOLDER}</code> "
            f"folder found in this sandbox.</span>")

def populate_sandboxes(*_):
    sbs = list_sandboxes()
    default_sb = ("sandbox-user" if "sandbox-user" in sbs
                  else (sbs[0] if sbs else None))
    sandbox_dd.options = sbs
    sandbox_dd.value = default_sb
    refresh_files()

install_dd.observe(populate_sandboxes, names="value")
sandbox_dd.observe(refresh_files, names="value")


# ---------- write the variables the notebook expects --------
def apply_selections():
    global currentSandbox, installationFolder, audioFileName
    global modelCheckpointToUse, minDurationOfFile, maxWavDuration, useGPU
    installationFolder   = install_dd.value
    currentSandbox       = sandbox_dd.value
    audioFileName        = audio_dd.value or ""
    modelCheckpointToUse = ckpt_dd.value or ""
    minDurationOfFile    = mindur_tb.value
    maxWavDuration       = maxdur_tb.value
    useGPU               = "yes" if gpu_cb.value else "no"

def on_confirm(_):
    apply_selections()
    with status_out:
        status_out.clear_output(wait=True)
        problems = []
        if not os.path.isdir(MYDRIVE_ROOT):
            problems.append(f"MyDrive not found at {MYDRIVE_ROOT} "
                            f"(is Google Drive mounted?).")
        elif not installationFolder:
            problems.append("No installation folder selected.")
        if not currentSandbox:
            problems.append("No sandbox selected.")
        if not audioFileName:
            problems.append("No audio file selected.")
        if not modelCheckpointToUse:
            problems.append("No model checkpoint selected.")

        if problems:
            print("⚠️  Please fix:")
            for p in problems:
                print("   • " + p)
            return

        print("✅  All set. The notebook will use:")
        print(f"   installationFolder   = {installationFolder}")
        print(f"   currentSandbox       = {currentSandbox}")
        print(f"   audioFileName        = {audioFileName}")
        print(f"   modelCheckpointToUse = {modelCheckpointToUse}")
        print(f"   minDurationOfFile    = {minDurationOfFile}")
        print(f"   maxWavDuration       = {maxWavDuration}")
        print(f"   useGPU               = {useGPU}")

confirm_btn.on_click(on_confirm)


# ---------- show it -----------------------------------------
if not _inst_options:
    apply_selections()                            # avoid NameErrors downstream
    display(widgets.HTML(
        f"<b style='color:#c0392b'>No folders found in {MYDRIVE_ROOT}.</b> "
        f"Mount Google Drive (uncomment the mount line at the top) and/or fix "
        f"MYDRIVE_ROOT, then re-run this cell."))
else:
    populate_sandboxes()                          # fill sandbox + audio + checkpoints
    apply_selections()                            # so later cells never NameError
    display(widgets.VBox([
        widgets.HTML("<b>1 · What to transcribe</b>"),
        install_dd, sandbox_dd,
        audio_dd, audio_lbl,
        ckpt_dd, ckpt_lbl,
        widgets.HTML("<b>2 · Processing options</b>"),
        mindur_tb, maxdur_tb, gpu_cb,
        widgets.HTML("&nbsp;"),
        confirm_btn, status_out,
    ]))


#=============================================================
# Load the SpeechBrain Voice Activity Detection (VAD) model.
# The pretrained model is downloaded the first time only.
#=============================================================
try:
    from speechbrain.inference.VAD import VAD      # SpeechBrain >= 1.0
except ImportError:
    from speechbrain.pretrained import VAD         # older SpeechBrain

vad = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty",
    savedir="/content/pretrained_models/vad-crdnn-libriparty",
)


#=============================================================
# Determine type of processing
#=============================================================

typeProcessor = "cuda"
if (useGPU == "no"): typeProcessor = "cpu"

#=============================================================
# Downloads ASR model for CIM
#=============================================================

!mkdir /content/wav2vec2-model
!cp /content/drive/MyDrive/{installationFolder}/{currentSandbox}/wav2vec2-model/*.* /content/wav2vec2-model
!mkdir /content/wav2vec2-model/checkpoint
!cp /content/drive/MyDrive/{installationFolder}/{currentSandbox}/wav2vec2-model/{modelCheckpointToUse}/*.* /content/wav2vec2-model/checkpoint

pathCheckpoint = "/content/wav2vec2-model/checkpoint"
modelPath = "/content/wav2vec2-model"

#model = Wav2Vec2ForCTC.from_pretrained(pathCheckpoint).to(typeProcessor)
#processor = Wav2Vec2Processor.from_pretrained("wav2vec2-model")

import os

def findKenLMModel(folder_path):
    for entry in os.listdir(folder_path):
        if entry.endswith('-correct.arpa') and os.path.isfile(os.path.join(folder_path, entry)):
            return entry
    return "-1"  # if no such file is found

filenameCorrectKenlmModel = findKenLMModel(modelPath + "/")
filenameCorrectKenlmModel = modelPath + "/" + filenameCorrectKenlmModel

def extractStartEndTimes(filepath):
    startTimes = []
    endTimes = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip newline and split by tabs
            parts = line.strip().split('\t')

            if len(parts) < 3:
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
                startTimes.append(start)
                endTimes.append(end)
            except ValueError:
                # If conversion to float fails, skip that line
                continue

    return startTimes, endTimes

def countDigits(number):
  count=0
  while(number>0):
    count=count+1
    number=number//10
  return(count)

def addZerosInFrontOfNumber(number, total):

  lenNum = len(str(number))
  lenTotal = len(str(total))

  zerosToAdd = lenTotal-lenNum

  stringZeros = ""
  for i in range(0,zerosToAdd): stringZeros = stringZeros + "0"
  retNum = stringZeros + str(number)
  return retNum

def findSegmentIndex(timepoints, starttime, endtime):
    # Convert timepoints string to a sorted list of floats
    points = sorted([float(tp) for tp in timepoints.split(',')])

    # Find the segment index where starttime and endtime correspond exactly
    # Segment 0: [0, points[0])
    # Segment i: [points[i-1], points[i])

    # Check if starttime is 0 (start of segment 0) or matches points
    # Since segments are defined between these points, the input start and end should match
    # one of the segment boundaries

    # We look for the segment where:
    # segment i means interval [points[i-1], points[i])

    # For segment 0: interval [0, points[0])
    for i, point in enumerate(points):
        if i == 0:
            seg_start = 0.0
        else:
            seg_start = points[i-1]
        seg_end = point

        # Check if the starttime and endtime match this segment
        if starttime == seg_start and endtime == seg_end:
            return i

    # If no match found, optionally return None or error
    return -1

from decimal import Decimal
import os

def fixIntervals(audio_path, file_path, maxDuration):
    # This list will hold all the intervals after processing
    intervals = []

    totalLines = 0

    # Open the file and read lines
    with open(file_path, 'r') as f:
        for line in f:
            totalLines = totalLines + 1
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            start, end = float(parts[0]), float(parts[1])

            # Check if the duration exceeds 15 seconds
            while end - start > maxDuration:
                # Add an interval with a maximum of 15 seconds
                intervals.append((start, start + maxDuration))
                start += maxDuration + 0.001  # Increment start for the next interval

            # Add the final interval (which is <= 15 seconds)
            intervals.append((start, end))

    # Sort intervals based on the start time
    intervals.sort()

    output = ""

    lineCounter = 0
    filenames = []
    # Print the processed and sorted intervals
    for interval in intervals:
        lineCounter = lineCounter + 1
        nameAudio = audio_path.replace(".wav", "-" + str(addZerosInFrontOfNumber(lineCounter,totalLines)) + ".wav")
        filenames.append(nameAudio)
        #output = output + str(round(Decimal(interval[0]),3)) + "\t" + interval[1] + "\n"
        output = output + str(round(Decimal(interval[0]),3)) + "\t" + str(round(Decimal(interval[1]),3)) + "\t" + nameAudio + "\n"
        #print(f"{interval[0]:.3f} {interval[1]:.3f}")

    file_path = file_path.replace(".csv", ".tsv")
    f = open(file_path, "w")
    f.write(output)
    f.close()

    sampleCSV = "path,sentence\n"
    for f in filenames:
      sampleCSV = sampleCSV + f + ", \n"
    sampleCSV = sampleCSV[:-1]

    folderPath = os.path.dirname(file_path)
    samplePath = folderPath + "/sample.csv"
    #print(samplePath)
    #print(sampleCSV)
    f = open(samplePath, "w")
    f.write(sampleCSV)
    f.close()


def leaveOnlyLastThreeColsOfTSV(inPath, outPath):
    try:
        with open(inPath, 'r', encoding='utf-8') as infile:  # Open the input file
            lines = infile.readlines()  # Read all lines

        processed_lines = []
        for line in lines:
            columns = line.strip().split('\t')  # Split the line into columns using tab as the delimiter
            if len(columns) >= 3:  # Ensure there are at least 3 columns to keep
                # Keep only the last three columns (columns[2], columns[3], columns[4])
                processed_lines.append(','.join(columns[2:]) + '\n')  # Rejoin and append to the list

        with open(outPath, 'w', encoding='utf-8') as outfile:  # Open the output file for writing
            outfile.writelines(processed_lines)  # Write the processed lines to the output file

        print("File processed successfully from {} to {}.".format(inPath, outPath))

    except IOError as e:
        print("An IOError occurred: {}".format(e))
    except Exception as e:
        print("An unexpected error occurred: {}".format(e))


import glob

#=============================================================
# Erase previous files and get new audio file
#=============================================================

# Erase previous files
%cd /content/
%rm *.wav >/dev/null 2>&1
%rm *.WAV >/dev/null 2>&1
%rm *.mp3 >/dev/null 2>&1
%rm *.MP3 >/dev/null 2>&1
%rm *.mp4 >/dev/null 2>&1
%rm *.MP4 >/dev/null 2>&1
%rm *.csv >/dev/null 2>&1
%rm *.CSV >/dev/null 2>&1
%rm *.tsv >/dev/null 2>&1
%rm *.TSV >/dev/null 2>&1
%rm *.txt >/dev/null 2>&1
%rm *.TXT >/dev/null 2>&1

folderWithAudioFiles = "/content/drive/MyDrive/"+installationFolder+"/"+currentSandbox+"/audiofiles-to-transcribe/"
!cp {folderWithAudioFiles}{audioFileName} .

#=============================================================
# Get proper path to audio file
#=============================================================

# get filenames of wave files in the remote server
path1 = r'/content/*.wav'
path2 = r'/content/*.WAV'
path3 = r'/content/*.mp3'
path4 = r'/content/*.MP3'
path5 = r'/content/*.mp4'
path6 = r'/content/*.MP4'
path7 = r'/content/*.mov'
path8 = r'/content/*.MOV'

files = []
files = glob.glob(path1) + glob.glob(path2) + glob.glob(path3) + glob.glob(path4) + glob.glob(path5) + glob.glob(path6) + glob.glob(path7) + glob.glob(path8)

# get name of the first file
if (len(files) == 0):
  print("=== ERROR: THERE ARE NO AUDIO FILES IN THE REMOTE SERVER ===")
else:
  wavfile = ""
  annotfile = ""
  for i in range(0,1): wavfile = files[i].replace("/content/","")
  print("File to split: " + wavfile)
  print("Annotation file: " + annotfile)

fileExtensionOrig = wavfile[-3:]
fileExtension = fileExtensionOrig.lower()
origFilename = wavfile


#=============================================================
# Convert file from mp3/mp4 to wav
#=============================================================

if (fileExtension == "mp3" or fileExtension == "mp4"):
  wavfile = origFilename.replace(origFilename[-3:],"") + "wav"
  !ffmpeg -i $origFilename -acodec pcm_u8 $wavfile
  print(wavfile)


#=============================================================
# Downgrade WAV file to the right ASR format (e.g. 16K)
#=============================================================

!ffmpeg -y -i $wavfile -ac 1 -ar 16000 temp-$wavfile
!rm $wavfile
!mv temp-$wavfile $wavfile

#=============================================================
# Find voice regions with SpeechBrain VAD
#=============================================================
# get_speech_segments() returns a tensor of [start, end] times
# in seconds. The audio is already 16 kHz mono at this point.

boundaries = vad.get_speech_segments(wavfile)

speech_timestamps = [
    {"start": round(float(seg[0]), 3), "end": round(float(seg[1]), 3)}
    for seg in boundaries
]

print("Found " + str(len(speech_timestamps)) + " speech regions.")

#=============================================================
# Write voice regions into a file
#=============================================================

output = ""

linePrefix = "tiername\tspeakername\t"

for t in speech_timestamps:
  output += linePrefix + str(t['start']) + "\t" + str(t['end']) + "\t \n"
output = output[:-1]

with open("voice-regions.txt", "w") as file: file.write(output)

#=============================================================
# Make sure there aren't any regions that are longer
# than the memory limit
#=============================================================

leaveOnlyLastThreeColsOfTSV("voice-regions.txt","temp-regions.txt")
fixIntervals(wavfile, "temp-regions.txt", maxWavDuration)


timeStart, timeEnd = extractStartEndTimes('temp-regions.txt')

#============================================================
# Separate the big file into smaller wav files
#============================================================

%mkdir wavs

filenames = []
print(len(timeStart))


points = []
for start, end in zip(timeStart, timeEnd):
  i = len(points)-1
  #tempName = "out" + addZerosToNumber(i+1,len(timeStart)) + ".wav"
  #tempName2 = "out" + addZerosToNumber(i+2,len(timeStart)) + ".wav"
  if points and points[-1] == start:
    points.append(end)
    #filenames.append(tempName)
  else:
    points.append(start)
    points.append(end)
    #filenames.append(tempName2)

pointSeq = ','.join(str(t) for t in points)
inFileName = wavfile

zeros = countDigits(len(timeStart))
outFileName = "wavs/out" + f'-%0{zeros}d.wav'
!ffmpeg -y -i "$inFileName" -f segment -ac 1 -ar 16000 -async 1 -segment_times $pointSeq $outFileName

outwavNames = []

for i in range(0,len(timeStart)):
  #print(timeStart[i])
  #print(timeEnd[i])
  segmentIndex = findSegmentIndex(pointSeq, timeStart[i], timeEnd[i])
  outwavNames.append("/content/wavs/out-" + addZerosInFrontOfNumber(str(segmentIndex),len(timeStart)) + ".wav")

print(outwavNames)

#outputName = "tempWav\tstart\tend\n"
outputName = ""
transcribeFile = "path,sentence\n"

for i in range(0,len(outwavNames)):
  outputName += str(timeStart[i]) + "\t" + str(timeEnd[i]) + "\t" + outwavNames[i] + "\n"
  transcribeFile += outwavNames[i] + ", \n"
outputName = outputName[:-1]
transcribeFile = transcribeFile[:-1]

with open("sample.csv", 'w', encoding='utf-8') as f: f.write(transcribeFile)
with open("files-and-times.txt", 'w', encoding='utf-8') as f: f.write(outputName)

from datetime import datetime
currentDateAndTime = datetime.now()
startTime = str(currentDateAndTime)

import os

import numpy as np
import pandas as pd

import torch
import librosa

import transformers
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ProcessorWithLM

import pyctcdecode
from pyctcdecode import build_ctcdecoder

import kenlm


#============================================================================
# Make sure we have a usable processor (GPU vs CPU)
#============================================================================
if typeProcessor == "cuda" and not torch.cuda.is_available():
    print("No GPU detected -> falling back to CPU (this will be slower).")
    typeProcessor = "cpu"

#============================================================================
# Load the list of audio chunks to transcribe
#============================================================================
dataTest = pd.read_csv("sample.csv")
dataTest.columns = [c.strip() for c in dataTest.columns]
wavPaths = [str(p).strip() for p in dataTest["path"].tolist()]

#============================================================================
# Load model + processor
#============================================================================
model = Wav2Vec2ForCTC.from_pretrained(pathCheckpoint).to(typeProcessor)
model.eval()
processor = Wav2Vec2Processor.from_pretrained(modelPath)

#============================================================================
# Build the KenLM decoder and wrap it in a processor-with-LM
#============================================================================
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

tempDecoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path=filenameCorrectKenlmModel,
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=tempDecoder,
)

#===============================================================================
# Transcribe each chunk
#===============================================================================
predictionLM = []
paths = []

for wavPath in wavPaths:
    # Chunks are already 16 kHz mono; librosa.load guarantees that format.
    speech, _ = librosa.load(wavPath, sr=16000)

    input_dict = processor(
        speech, sampling_rate=16000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        logits = model(input_dict.input_values.to(typeProcessor)).logits

    # Single-sample LM decode (no multiprocessing pool -> safe with CUDA)
    predictionlm = processor_with_lm.decode(logits[0].cpu().numpy()).text
    predictionLM.append(predictionlm)
    print(predictionlm)

    paths.append(os.path.basename(wavPath))

#===============================================================================
# Save transcriptions, aligned with their start/end times
#===============================================================================
tsvTimes = "files-and-times.txt"

with open(tsvTimes, 'r') as file:
    linesTSV = [line.strip() for line in file.readlines()]

outputTranscriptions = "start\tend\ttranscription\n"

for l in linesTSV:
    l = l.split("\t")
    tsvWaveChunk = os.path.basename(l[2])
    for i in range(0, len(predictionLM)):
        if (tsvWaveChunk == paths[i]):
            outputTranscriptions = outputTranscriptions + l[0] + "\t" + l[1] + "\t" + predictionLM[i] + "\n"

outputTranscriptions = outputTranscriptions[:-1]

tsvOutputputFilename = os.path.basename(wavfile.replace(".wav",".tsv"))

with open(tsvOutputputFilename, 'w') as file:
    file.write(outputTranscriptions)

print("\nThe results are stored in: \n" + tsvOutputputFilename)
