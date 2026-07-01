# Mount the Google Drive onto the virtual computer

# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

# ============================================================
#  RECORD & TRANSCRIBE
# ============================================================
#  Run this cell, choose the folder where your sandboxes live and
#  the model checkpoint you want to use, then click
#  "✅ Confirm selection". Choosing a checkpoint automatically sets
#  which sandbox it belongs to.
#
#  You normally won't edit anything here. If your Drive lives
#  somewhere unusual, adjust MYDRIVE_ROOT below.
# ------------------------------------------------------------

import os
import ipywidgets as widgets
from IPython.display import display

# ----- Config -----------------------------------------------
MYDRIVE_ROOT = "/content/drive/MyDrive"           # where installation folders live
DEFAULT_INSTALLATION_FOLDER = "202606-cim-asr"    # default selection if present
MODEL_SUBFOLDER = "wav2vec2-model"                # holds the checkpoint-* folders
# ------------------------------------------------------------


# ---------- helpers: folders on disk ------------------------
def list_installations():
    if not os.path.isdir(MYDRIVE_ROOT):
        return []
    return sorted(d for d in os.listdir(MYDRIVE_ROOT)
                  if os.path.isdir(os.path.join(MYDRIVE_ROOT, d)))

def installation_root():
    inst = install_dd.value
    return os.path.join(MYDRIVE_ROOT, inst) if inst else ""

def list_sandboxes():
    root = installation_root()
    if not root or not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))

def find_model_folder(sandbox):
    """Locate the wav2vec2-model folder inside a sandbox (exact, then
    case-insensitive, then nested up to 3 levels). Returns path or None."""
    base = os.path.join(installation_root(), sandbox)
    if not os.path.isdir(base):
        return None
    direct = os.path.join(base, MODEL_SUBFOLDER)
    if os.path.isdir(direct):
        return direct
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p) and d.lower() == MODEL_SUBFOLDER.lower():
            return p
    for dirpath, dirnames, files in os.walk(base):
        if dirpath[len(base):].count(os.sep) >= 3:
            dirnames[:] = []
        for d in list(dirnames):
            if d.lower() == MODEL_SUBFOLDER.lower():
                return os.path.join(dirpath, d)
    return None

def _step(name):
    tail = name.split("-")[-1]
    return int(tail) if tail.isdigit() else -1

def list_all_checkpoints():
    """(sandbox, checkpoint, step) for every checkpoint in every sandbox."""
    out = []
    for sb in list_sandboxes():
        folder = find_model_folder(sb)
        if not folder:
            continue
        subs = [d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))]
        cks = [d for d in subs if d.lower().startswith("checkpoint")] or subs
        for c in cks:
            out.append((sb, c, _step(c)))
    out.sort(key=lambda e: e[2], reverse=True)        # latest first
    return out


# ---------- build the widgets -------------------------------
label_style = {"description_width": "170px"}
row_layout  = widgets.Layout(width="560px")

_inst_options = list_installations()
install_dd = widgets.Dropdown(
    description="Sandboxes folder:", options=_inst_options,
    value=(DEFAULT_INSTALLATION_FOLDER if DEFAULT_INSTALLATION_FOLDER in _inst_options
           else (_inst_options[0] if _inst_options else None)),
    style=label_style, layout=row_layout)

ckpt_dd = widgets.Dropdown(description="Model checkpoint:", options=[],
                           style=label_style, layout=row_layout)
ckpt_lbl = widgets.HTML()
sel_lbl  = widgets.HTML()

confirm_btn = widgets.Button(description="✅ Confirm selection",
                             button_style="success",
                             layout=widgets.Layout(width="200px"))
status_out = widgets.Output()


# ---------- keep things in sync -----------------------------
def refresh_checkpoints(*_):
    entries = list_all_checkpoints()
    ckpt_dd.options = [(f"{sb}  /  {ck}", (sb, ck)) for (sb, ck, _s) in entries]

    # default to the latest checkpoint in 'sandbox-user', else the latest overall
    default_val = None
    user_entries = [e for e in entries if e[0] == "sandbox-user"]
    if user_entries:
        default_val = (user_entries[0][0], user_entries[0][1])
    elif entries:
        default_val = (entries[0][0], entries[0][1])
    ckpt_dd.value = default_val

    if entries:
        sandboxes = sorted({e[0] for e in entries})
        ckpt_lbl.value = (
            f"<span style='color:#1a7f37'>🧠 {len(entries)} checkpoint(s) "
            f"across: {', '.join(sandboxes)}</span>")
    elif not list_sandboxes():
        ckpt_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No sandbox folders in "
            f"<code>{installation_root()}</code>. Pick a different folder.</span>")
    else:
        ckpt_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No checkpoints found in any "
            f"<code>{MODEL_SUBFOLDER}</code> folder here.</span>")
    update_sel_label()

def update_sel_label(*_):
    sel = ckpt_dd.value
    if sel:
        sel_lbl.value = (f"<span style='color:#555'>→ will use checkpoint "
                         f"<b>{sel[1]}</b> from sandbox <b>{sel[0]}</b></span>")
    else:
        sel_lbl.value = ""

install_dd.observe(refresh_checkpoints, names="value")
ckpt_dd.observe(update_sel_label, names="value")


# ---------- write the variables the notebook expects --------
def apply_selections():
    global currentSandbox, installationFolder, modelCheckpointToUse, useGPU
    installationFolder = install_dd.value
    sel = ckpt_dd.value
    currentSandbox, modelCheckpointToUse = sel if sel else ("", "")
    useGPU = "yes"        # GPU by default; set to "no" here to force CPU

def on_confirm(_):
    apply_selections()
    with status_out:
        status_out.clear_output(wait=True)
        problems = []
        if not os.path.isdir(MYDRIVE_ROOT):
            problems.append(f"MyDrive not found at {MYDRIVE_ROOT} "
                            f"(is Google Drive mounted?).")
        elif not installationFolder:
            problems.append("No sandboxes folder selected.")
        if not modelCheckpointToUse:
            problems.append("No checkpoint selected.")

        if problems:
            print("⚠️  Please fix:")
            for p in problems:
                print("   • " + p)
            return

        print("✅  All set. The notebook will use:")
        print(f"   installationFolder   = {installationFolder}")
        print(f"   currentSandbox       = {currentSandbox}")
        print(f"   modelCheckpointToUse = {modelCheckpointToUse}")
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
    refresh_checkpoints()                          # fill checkpoint dropdown
    apply_selections()                             # so later cells never NameError
    display(widgets.VBox([
        widgets.HTML("<b>Pick your folder and checkpoint, then Confirm.</b>"),
        install_dd, ckpt_dd, ckpt_lbl, sel_lbl,
        widgets.HTML("&nbsp;"),
        confirm_btn, status_out,
    ]))


#=============================================================
# Installs software for the ASR model
# Takes about 1-2 minutes. Run this once per session.
#=============================================================

# IMPORTANT: We use the PyTorch that is ALREADY installed in
# Colab (it is GPU-enabled). The old "torch==1.10.0+cu113"
# pin no longer exists for current Colab Python and was the
# main reason this notebook stopped working. We only install
# the Hugging Face libraries.
# !pip install -q -U transformers datasets accelerate soundfile

from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor

import glob
import torch
import numpy as np
import soundfile as sf
import pandas as pd
from datasets import Dataset
from transformers import Wav2Vec2FeatureExtractor

#=============================================================
# Determine type of processing
#=============================================================

typeProcessor = "cuda" if (useGPU == "yes" and torch.cuda.is_available()) else "cpu"
print("Using device:", typeProcessor)

#=============================================================
# Downloads ASR model
#=============================================================

!mkdir -p /content/wav2vec2-model
!cp /content/drive/MyDrive/{installationFolder}/{currentSandbox}/wav2vec2-model/*.* /content/wav2vec2-model
!mkdir -p /content/wav2vec2-model/checkpoint
!cp /content/drive/MyDrive/{installationFolder}/{currentSandbox}/wav2vec2-model/{modelCheckpointToUse}/*.* /content/wav2vec2-model/checkpoint

pathCheckpoint = "wav2vec2-model/checkpoint"
model = Wav2Vec2ForCTC.from_pretrained(pathCheckpoint).to(typeProcessor)
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-model")

# Minimum duration of segments that the computer should transcribe
minDurationOfFile = 100 #ms


#=============================================================
# Create dummy CSV file
#=============================================================

output = "path,sentence\nsample-recording.wav,kia orana"
f = open("sample-recording.csv", "w")
f.write(output)
f.close()


#=============================================================
# Convert model orthography to human orthography
#=============================================================

def convertToHumanOrthography(inputString):

  outputString = inputString

  # Replace the output transcription with orthographic output
  orthOrigin = ['ax', 'ex', 'ix', 'ox', 'ux', 'q']
  orthTarget = ['ā', 'ē', 'ī', 'ō', 'ū', 'ꞌ']
  for i in range(0,len(orthOrigin)): outputString = outputString.replace(orthOrigin[i], orthTarget[i])

  return outputString


#=============================================================
# Function to run inference of new files
#=============================================================

def runInference ():

  # Convert audio file to array (using soundfile instead of the
  # now-deprecated torchaudio loading backend)
  def speech_file_to_array_fn(batch):
      speech_array, sampling_rate = sf.read(batch["path"])
      if getattr(speech_array, "ndim", 1) > 1:
          speech_array = speech_array[:, 0]   # keep first channel if stereo
      batch["speech"] = np.asarray(speech_array, dtype=np.float32)
      batch["sampling_rate"] = sampling_rate
      batch["target_text"] = batch["sentence"]
      return batch

  # Prepare batch processing of files.
  # NOTE: the old label-encoding block used processor.as_target_processor(),
  # which is deprecated/removed. Labels are not needed for inference, so it
  # has been removed.
  def prepare_dataset(batch):
      # check that all files have the correct sampling rate
      assert (
          len(set(batch["sampling_rate"])) == 1
      ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

      batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
      return batch

  # Load CSV file with audio files to be transcribed
  dataTest = pd.read_csv("sample-recording.csv")
  common_voice_test = Dataset.from_pandas(dataTest)

  # Extract features from audio files
  common_voice_test = common_voice_test.map(speech_file_to_array_fn, remove_columns=common_voice_test.column_names)
  common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names, batch_size=8, num_proc=1, batched=True)

  # Process audio files
  input_dict = processor(common_voice_test[0]["input_values"], return_tensors="pt", padding=True)
  logits = model(input_dict.input_values.to(typeProcessor)).logits
  pred_ids = torch.argmax(logits, dim=-1)[0]

  # Decode audio files
  predictedText = convertToHumanOrthography(processor.decode(pred_ids))

  return predictedText

#=============================================================
# Code to record audio from the Colab Notebook
# Takes a few seconds
#=============================================================

# Only ffmpeg-python is needed (to convert the browser recording
# to wav). The old sox / WavAugment installs were unused and tend
# to fail to build on current Colab, so they have been removed.
# !pip install -q ffmpeg-python

import soundfile as sf
import numpy as np

# # code from https://ricardodeazambuja.com/deep_learning/2019/03/09/audio_and_video_google_colab/
from IPython.display import HTML, Audio
# from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
import io
import ffmpeg
import tempfile
import pathlib


AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
recordButton.onclick = ()=>{
  toggleRecording();
  // Poll until the browser has finished encoding the recording
  // (base64data stays 0 until the FileReader is done). This replaces
  // the old fixed 2-second wait, which often fired too early.
  var waitForData = setInterval(function(){
    if (base64data != 0) {
      clearInterval(waitForData);
      resolve(base64data.toString());
    }
  }, 200);
  // Safety net: give up after 15s and return whatever we have,
  // so the Python side can show a helpful message instead of hanging.
  setTimeout(function(){
    clearInterval(waitForData);
    resolve(base64data.toString());
  }, 15000);
};
});

</script>
"""

def get_audio():
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  if (not data) or (',' not in data):
    raise RuntimeError(
        "No audio was captured. Please check that:\n"
        "  1. You ALLOWED microphone access when the browser asked.\n"
        "  2. You actually spoke after the cell started listening.\n"
        "  3. You clicked the button to STOP recording and waited a moment.\n"
        "Then run this cell again."
    )
  binary = b64decode(data.split(',')[1])

  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)

  riff_chunk_size = len(output) - 8
  # Break up the chunk size into four bytes, held in b.
  q = riff_chunk_size
  b = []
  for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)

  # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
  riff = output[:4] + bytes(b) + output[8:]

  with tempfile.TemporaryDirectory() as tmpdirname:
    path = pathlib.Path(tmpdirname) / 'tmp.wav'
    with open(path, 'wb') as f:
       f.write(riff)
    with open('temp.wav', 'wb') as f:
       f.write(riff)

    x, sr = sf.read(path)

  return x, sr

# GET READY!
# When you run the next box, the computer will say that it's listening to you.
# I recommend that you say something simple. For example:
#
# Kia orana kotou kātoatoa!
#
# It means "Hello, everyone" or "How are you all doing?"
# After you said it, click on "Press to stop". Once the "play" button (on the left) stops rotating, that's
# when your recording will be ready and you can continue to the next step. (The button will keep saying
# "saving the recording... pls wait!". Don't pay attention to that. Pay attention to the play button. If
# the rotating line has stopped, then you're good).
#
# The system will make many mistakes because it's only trained on 16 minutes on audio.
# As the system learns from more data, the transcriptions will get better.

#==========================================================================
# This will open a button to record audio.
# It will save the file as temp.wav
# Once you see a play button, this means that the file is saved correctly.
#==========================================================================

%rm /content/temp.wav
%cd /content/
x, sr = get_audio()

#==========================================================================
# Convert the audio file to mono (one channel)
# and downgrade its quality to 16KHz
#==========================================================================

!ffmpeg -y -i temp.wav -ac 1 -ar 16000 temp-sample-recording.wav
!rm temp.wav
!mv temp-sample-recording.wav sample-recording.wav

#==========================================================================
# Transcribe recording
#==========================================================================

predictedText = runInference()

#==========================================================================
# Print the transcription of the recording
#==========================================================================

print("Prediction:")
print(predictedText)

#==========================================================================
# Play recording
#==========================================================================

import IPython
IPython.display.Audio('sample-recording.wav') # This is required on Google Colab due to compatibility issues