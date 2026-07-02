# !pip install -q -U \
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

# IMPORTANT: NumPy was already imported by Colab at startup, so the new
# version won't take effect until the Python process restarts. We trigger
# an automatic restart here.
import os
print("Install finished. Restarting runtime so NumPy 1.26.4 is loaded...")
os.kill(os.getpid(), 9)

# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

# ============================================================
#  SELECT TRAINING FILES AND SET UP TRAINING
# ============================================================
#  Run this cell, choose the installation folder and sandbox, and
#  the train / validation / test CSVs found in that sandbox are
#  offered in dropdowns (auto-matched by name, override if needed).
#  Set the language prefix, run id and training options, then click
#  "✅ Confirm selections".
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


# ---------- helpers: find the CSVs --------------------------
def resolve_csv_folder(sandbox):
    """Find the folder inside the sandbox that holds the partition CSVs.
    Prefers the sandbox folder itself; otherwise the first nested folder
    (down to 3 levels) that contains a train/valid/test CSV."""
    root = sandbox_root()
    if not sandbox or not root:
        return None
    base = os.path.join(root, sandbox)
    if not os.path.isdir(base):
        return None
    if any(f.lower().endswith(".csv") for f in os.listdir(base)):
        return base
    fallback = None
    for dirpath, dirnames, files in os.walk(base):
        if dirpath[len(base):].count(os.sep) >= 3:
            dirnames[:] = []
        csvs = [f for f in files if f.lower().endswith(".csv")]
        if csvs:
            joined = " ".join(c.lower() for c in csvs)
            if any(k in joined for k in ("train", "valid", "test")):
                return dirpath
            if fallback is None:
                fallback = dirpath
    return fallback

def list_csvs(sandbox):
    folder = resolve_csv_folder(sandbox)
    if not folder or not os.path.isdir(folder):
        return folder, []
    csvs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
    return folder, csvs

def pick_default(csvs, keyword):
    for c in csvs:
        if keyword in c.lower():
            return c
    return None


# ---------- build the widgets -------------------------------
label_style = {"description_width": "200px"}
row_layout  = widgets.Layout(width="620px")

_inst_options = list_installations()
install_dd = widgets.Dropdown(
    description="Installation folder:", options=_inst_options,
    value=(DEFAULT_INSTALLATION_FOLDER if DEFAULT_INSTALLATION_FOLDER in _inst_options
           else (_inst_options[0] if _inst_options else None)),
    style=label_style, layout=row_layout)

sandbox_dd = widgets.Dropdown(description="Sandbox (currentSandbox):",
                              options=[], style=label_style, layout=row_layout)
csv_lbl = widgets.HTML()

train_dd = widgets.Dropdown(description="Train CSV (trainFile):",
                            style=label_style, layout=row_layout)
valid_dd = widgets.Dropdown(description="Validation CSV (validFile):",
                            style=label_style, layout=row_layout)
test_dd  = widgets.Dropdown(description="Test CSV (testFile):",
                            style=label_style, layout=row_layout)

lang_tb = widgets.Text(description="Language prefix (asrLang):", value="cim",
                       style=label_style, layout=row_layout)
runid_tb = widgets.Text(description="Run ID (runId):", value="01",
                        style=label_style, layout=row_layout)
epochs_tb = widgets.BoundedIntText(description="Train epochs:", value=34,
                                   min=1, max=100000,
                                   style=label_style, layout=row_layout)
ngrams_tb = widgets.BoundedIntText(description="KenLM n-grams:", value=4,
                                   min=1, max=10,
                                   style=label_style, layout=row_layout)

confirm_btn = widgets.Button(description="✅ Confirm selections",
                             button_style="success",
                             layout=widgets.Layout(width="200px"))
status_out = widgets.Output()


# ---------- keep things in sync -----------------------------
def refresh_csvs(*_):
    sb = sandbox_dd.value
    folder, csvs = list_csvs(sb)
    for dd in (train_dd, valid_dd, test_dd):
        dd.options = csvs
    train_dd.value = pick_default(csvs, "train")
    valid_dd.value = pick_default(csvs, "valid")
    test_dd.value = pick_default(csvs, "test")

    if not sb:
        csv_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No sandbox folders in "
            f"<code>{os.path.join(MYDRIVE_ROOT, str(install_dd.value))}</code>. "
            f"Pick a different installation folder.</span>")
    elif folder and csvs:
        csv_lbl.value = (
            f"<span style='color:#1a7f37'>📄 Found {len(csvs)} CSV(s) in "
            f"<code>{folder}</code></span>")
    else:
        csv_lbl.value = (
            f"<span style='color:#c0392b'>⚠️ No .csv files found in "
            f"<code>{os.path.join(sandbox_root(), str(sb))}</code> "
            f"(have the partitions been created yet?).</span>")

def populate_sandboxes(*_):
    sbs = list_sandboxes()
    default_sb = ("raw" if "raw" in sbs
                  else (sbs[0] if sbs else None))
    sandbox_dd.options = sbs
    sandbox_dd.value = default_sb
    refresh_csvs()

install_dd.observe(populate_sandboxes, names="value")
sandbox_dd.observe(refresh_csvs, names="value")


# ---------- write the variables the notebook expects --------
def apply_selections():
    global currentSandbox, installationFolder
    global trainFile, validFile, testFile
    global runId, desiredTrainEpochs, asrLang, ngrams
    installationFolder = install_dd.value
    currentSandbox     = sandbox_dd.value
    trainFile          = train_dd.value or ""
    validFile          = valid_dd.value or ""
    testFile           = test_dd.value or ""
    runId              = runid_tb.value.strip()
    desiredTrainEpochs = epochs_tb.value
    asrLang            = lang_tb.value.strip()
    ngrams             = ngrams_tb.value

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
        if not trainFile:
            problems.append("No train CSV selected.")
        if not validFile:
            problems.append("No validation CSV selected.")
        if not testFile:
            problems.append("No test CSV selected.")
        if len({trainFile, validFile, testFile}) < 3 and all(
                (trainFile, validFile, testFile)):
            problems.append("Train/validation/test CSVs are not all different.")
        if not asrLang:
            problems.append("Language prefix is empty.")
        if not runId:
            problems.append("Run ID is empty.")

        if problems:
            print("⚠️  Please fix:")
            for p in problems:
                print("   • " + p)
            return

        print("✅  All set. The notebook will use:")
        print(f"   installationFolder = {installationFolder}")
        print(f"   currentSandbox     = {currentSandbox}")
        print(f"   trainFile          = {trainFile}")
        print(f"   validFile          = {validFile}")
        print(f"   testFile           = {testFile}")
        print(f"   asrLang            = {asrLang}")
        print(f"   runId              = {runId}")
        print(f"   desiredTrainEpochs = {desiredTrainEpochs}")
        print(f"   ngrams             = {ngrams}")

confirm_btn.on_click(on_confirm)


# ---------- show it -----------------------------------------
if not _inst_options:
    apply_selections()                            # avoid NameErrors downstream
    display(widgets.HTML(
        f"<b style='color:#c0392b'>No folders found in {MYDRIVE_ROOT}.</b> "
        f"Mount Google Drive (uncomment the mount line at the top) and/or fix "
        f"MYDRIVE_ROOT, then re-run this cell."))
else:
    populate_sandboxes()                          # fill sandbox + CSV dropdowns
    apply_selections()                            # so later cells never NameError
    display(widgets.VBox([
        widgets.HTML("<b>1 · Which files to train on</b>"),
        install_dd, sandbox_dd, csv_lbl,
        train_dd, valid_dd, test_dd,
        widgets.HTML("<b>2 · Training options</b>"),
        lang_tb, runid_tb, epochs_tb, ngrams_tb,
        widgets.HTML("&nbsp;"),
        confirm_btn, status_out,
    ]))


datasetPath = "/content/drive/MyDrive/"+installationFolder+"/" + currentSandbox + "/"

csvTrain = datasetPath + trainFile
csvValid = datasetPath + validFile
csvTest = datasetPath + testFile
corpusFile = datasetPath + trainFile.replace("-train.csv","-corpus.txt")

filenameKenlmModel = "lm-" + asrLang + "-" + str(ngrams) + ".arpa"
filenameCorrectKenlmModel = filenameKenlmModel.replace(".arpa", "-correct.arpa")

folderLogFiles = datasetPath + "logs-wav2vec2-res/"
folderModelFiles = "/content/wav2vec2-large-xlsr/"

condition = "wav2vec2"

outputPrefix = asrLang + "-" + condition
transferModelPath = ""

import os

if not os.path.exists("/content/kenlm/build/bin/lmplz"):
    !apt-get install -y -q build-essential cmake libboost-system-dev \
        libboost-thread-dev libboost-program-options-dev \
        libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev > /dev/null 2>&1
    !git clone -q https://github.com/kpu/kenlm.git /content/kenlm
    !mkdir -p /content/kenlm/build
    !cd /content/kenlm/build && cmake .. > /dev/null 2>&1 && make -j2 > /dev/null 2>&1

KENLM_BIN = "/content/kenlm/build/bin"
assert os.path.exists(f"{KENLM_BIN}/lmplz"), "lmplz failed to build!"
print("KenLM tools ready at:", KENLM_BIN)

import numpy as np
assert np.__version__.startswith("1."), \
    f"NumPy is {np.__version__}; expected 1.x. Re-run Cell 4 and restart."
print("NumPy OK:", np.__version__)

import os
import re
import json
import unicodedata
import pandas as pd
import torch
import torchaudio

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# ---- COLUMN NAMES: edit these if auto-detection picks the wrong columns ----
# Set to None to auto-detect, or set to the exact column name string.
AUDIO_COLUMN = None        # e.g. "path"
TEXT_COLUMN  = None        # e.g. "sentence"
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000

os.makedirs(folderLogFiles, exist_ok=True)
print("Log folder:", folderLogFiles)

def _detect_columns(df):
    audio_candidates = ["path", "audio", "wav", "file", "filename", "filepath", "audio_path"]
    text_candidates  = ["sentence", "text", "transcription", "transcript", "label", "target"]
    cols_lower = {c.lower(): c for c in df.columns}

    audio_col = AUDIO_COLUMN
    text_col  = TEXT_COLUMN
    if audio_col is None:
        for cand in audio_candidates:
            if cand in cols_lower:
                audio_col = cols_lower[cand]; break
    if text_col is None:
        for cand in text_candidates:
            if cand in cols_lower:
                text_col = cols_lower[cand]; break

    if audio_col is None or text_col is None:
        raise ValueError(
            f"Could not auto-detect columns. Found columns: {list(df.columns)}. "
            f"Please set AUDIO_COLUMN and TEXT_COLUMN in Cell 6."
        )
    return audio_col, text_col

def _try_read_csv(path):
    # Be tolerant of separators and quoting.
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return pd.read_csv(path)  # last resort, let it raise

df_train = _try_read_csv(csvTrain)
df_valid = _try_read_csv(csvValid)
df_test  = _try_read_csv(csvTest)

audio_col, text_col = _detect_columns(df_train)
print(f"Using audio column: '{audio_col}' | text column: '{text_col}'")
print("Train/Valid/Test sizes:", len(df_train), len(df_valid), len(df_test))

def _resolve_audio_path(p):
    p = str(p).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(datasetPath, p)
    if os.path.exists(cand):
        return cand
    # try a 'clips' or 'wavs' subfolder fallback
    for sub in ["wavs", "clips", "audio", "data"]:
        cand2 = os.path.join(datasetPath, sub, os.path.basename(p))
        if os.path.exists(cand2):
            return cand2
    return p  # return as-is; will error later if truly missing

# Text normalisation: lowercase, strip punctuation we won't model,
# but KEEP apostrophes (they are meaningful in this language).

# Normalise all typographic/curly apostrophe variants to a single
# straight apostrophe (') BEFORE we remove any other punctuation,
# so apostrophes are consistent across the whole corpus.
apostrophe_variants = r"[’‘ʼʻ`´‛]"   # curly, modifier letter, grave/acute, etc.

# Note: the apostrophe is deliberately NOT in this removal class.
chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\”\�\(\)\[\]\{\}«»…]'

def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(apostrophe_variants, "'", text)   # unify apostrophes -> '
    text = re.sub(chars_to_remove_regex, "", text)  # remove other punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

for df in (df_train, df_valid, df_test):
    df[audio_col] = df[audio_col].apply(_resolve_audio_path)
    df[text_col]  = df[text_col].apply(normalize_text)
    df.dropna(subset=[audio_col, text_col], inplace=True)

# Quick existence check
missing = [p for p in df_train[audio_col].tolist()[:50] if not os.path.exists(p)]
if missing:
    print("WARNING: some audio files not found, e.g.:", missing[:5])
else:
    print("Audio path check OK (sampled).")

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

def build_vocab(*dfs):
    all_text = " ".join(pd.concat([d[text_col] for d in dfs]).tolist())
    vocab = sorted(set(all_text))
    return vocab

vocab_list = build_vocab(df_train, df_valid, df_test)
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# Replace space with a visible delimiter and add special tokens.
if " " in vocab_dict:
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

vocab_path = "/content/vocab.json"
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False)
print("Vocab size:", len(vocab_dict))

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=TARGET_SAMPLE_RATE,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

os.makedirs(folderModelFiles, exist_ok=True)
processor.save_pretrained(folderModelFiles)


all_sentences = df_train[text_col].tolist()

with open(corpusFile, "w", encoding="utf-8") as f:
    for s in all_sentences:
        s = s.strip()
        if s:
            f.write(s + "\n")
print("Wrote corpus:", corpusFile, "| lines:", len(all_sentences))

print("Skipping KenLM language model building as requested.")

from datasets import Dataset, Audio

def df_to_ds(df):
    ds = Dataset.from_pandas(
        df[[audio_col, text_col]].rename(
            columns={audio_col: "audio", text_col: "sentence"}
        ).reset_index(drop=True)
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    return ds

train_ds = df_to_ds(df_train)
valid_ds = df_to_ds(df_valid)
test_ds  = df_to_ds(df_test)

def prepare_batch(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names, num_proc=1)
valid_ds = valid_ds.map(prepare_batch, remove_columns=valid_ds.column_names, num_proc=1)
# Keep test sentences for evaluation, so map but retain 'sentence' separately later.
test_ds_prepared = test_ds.map(prepare_batch, remove_columns=[c for c in test_ds.column_names if c != "sentence"], num_proc=1)

# Optional: filter out overly long clips to avoid OOM (e.g. > 20 s).
MAX_INPUT_LENGTH = TARGET_SAMPLE_RATE * 20
train_ds = train_ds.filter(lambda x: x < MAX_INPUT_LENGTH, input_columns=["input_length"])
print("Prepared. Train size after length filter:", len(train_ds))

import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

from transformers import Wav2Vec2ForCTC

base_checkpoint = transferModelPath if transferModelPath else "facebook/wav2vec2-large-xlsr-53"

model = Wav2Vec2ForCTC.from_pretrained(
    base_checkpoint,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
# Freeze the CNN feature encoder (standard for XLSR fine-tuning).
model.freeze_feature_encoder()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=folderModelFiles,
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=400,
    num_train_epochs=desiredTrainEpochs,
    fp16=torch.cuda.is_available(),
    learning_rate=3e-4,
    warmup_ratio=0.1,
    save_total_limit=20,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

# Save the best model + processor.
trainer.save_model(folderModelFiles)
processor.save_pretrained(folderModelFiles)
print("Training complete. Model saved to:", folderModelFiles)

from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import os, shutil

# --- Reconstruct ARPA paths from Cell 1-3 variables (restart-safe) ---
arpa_path = os.path.join("/content", filenameKenlmModel)
correct_arpa_path = os.path.join("/content", filenameCorrectKenlmModel)

# If the local copies are gone (e.g. after a restart), restore them from Drive.
drive_arpa         = os.path.join(datasetPath, filenameKenlmModel)
drive_correct_arpa = os.path.join(datasetPath, filenameCorrectKenlmModel)

if not os.path.exists(correct_arpa_path):
    if os.path.exists(drive_correct_arpa):
        shutil.copy(drive_correct_arpa, correct_arpa_path)
        print("Restored corrected ARPA from Drive.")
    elif os.path.exists(drive_arpa) or os.path.exists(arpa_path):
        # We have the raw ARPA but not the corrected one: rebuild the patch.
        src = arpa_path if os.path.exists(arpa_path) else drive_arpa
        if not os.path.exists(arpa_path):
            shutil.copy(drive_arpa, arpa_path)
        with open(arpa_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(correct_arpa_path, "w", encoding="utf-8") as out:
            has_added_eos = False
            for line in lines:
                if not has_added_eos and "ngram 1=" in line:
                    count = int(line.strip().split("=")[-1])
                    out.write(line.replace(f"ngram 1={count}", f"ngram 1={count+1}"))
                elif not has_added_eos and "<s>" in line:
                    out.write(line)
                    out.write(line.replace("<s>", "</s>"))
                    has_added_eos = True
                else:
                    out.write(line)
        print("Rebuilt corrected ARPA from raw ARPA.")
    else:
        raise FileNotFoundError(
            "No ARPA file found locally or on Drive. Please re-run Cell 9 "
            "to build the KenLM model first."
        )

assert os.path.exists(correct_arpa_path), "Corrected ARPA still missing."
print("Using KenLM ARPA:", correct_arpa_path)

# --- Reload the trained model for inference ---
model = Wav2Vec2ForCTC.from_pretrained(folderModelFiles)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Build label list in tokenizer index order for pyctcdecode ---
vocab_dict_sorted = processor.tokenizer.get_vocab()
sorted_vocab = sorted(vocab_dict_sorted.items(), key=lambda kv: kv[1])
labels = [t for t, _ in sorted_vocab]
labels = ["" if t == "[PAD]" else (" " if t == "|" else t) for t in labels]

decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path=correct_arpa_path,
    alpha=0.5,   # LM weight
    beta=1.0,    # word insertion bonus
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder,
)
print("KenLM decoder ready.")

import os, re, glob, shutil
import numpy as np
import pandas as pd
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Discover ONLY numbered checkpoints (exclude the existing 'final') ----
def find_numbered_checkpoints(model_dir):
    found = []
    ckpt_dirs = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    def _step(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpt_dirs = sorted(ckpt_dirs, key=_step)
    for d in ckpt_dirs:
        if os.path.exists(os.path.join(d, "config.json")):
            found.append((os.path.basename(d), d))
    return found

checkpoints = find_numbered_checkpoints(folderModelFiles)
if not checkpoints:
    raise FileNotFoundError(
        f"No numbered checkpoints found under {folderModelFiles}. "
        f"Was training run with save_strategy='epoch'?"
    )
print("Found numbered checkpoints:")
for label, path in checkpoints:
    print(f"  - {label}: {path}")

# ---- Build KenLM decoder once (tokenizer/vocab shared across checkpoints) ----
vocab_dict_sorted = processor.tokenizer.get_vocab()
sorted_vocab = sorted(vocab_dict_sorted.items(), key=lambda kv: kv[1])
labels = [t for t, _ in sorted_vocab]
labels = ["" if t == "[PAD]" else (" " if t == "|" else t) for t in labels]

decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path=correct_arpa_path,
    alpha=0.5,
    beta=1.0,
)
processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder,
)

# ---- Inference helpers ----
def get_logits(model, input_values):
    iv = torch.tensor(input_values).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(iv).logits
    return logits.cpu().numpy()[0]

def safe(s):
    return s if s.strip() else " "

# ---- Evaluate every numbered checkpoint ----
# Keep per-checkpoint rows so we can (a) reuse them for 'final' selection and
# (b) rank checkpoints by LM performance.
rows_by_ckpt = {}      # ckpt_label -> list[dict]
first_sentence_preview = {}   # ckpt_label -> dict(gold, greedy, kenlm)
for ckpt_label, ckpt_path in checkpoints:
    print(f"\n=== Evaluating checkpoint: {ckpt_label} ===")
    ckpt_model = Wav2Vec2ForCTC.from_pretrained(ckpt_path)
    ckpt_model.eval()
    ckpt_model.to(device)

    ckpt_rows = []
    for i in range(len(test_ds_prepared)):
        ex = test_ds_prepared[i]
        reference = ex["sentence"]
        logits = get_logits(ckpt_model, ex["input_values"])

        pred_ids = np.argmax(logits, axis=-1)
        hyp_greedy = processor.decode(pred_ids).strip()
        hyp_lm = processor_with_lm.decode(logits).text.strip()

        # Capture the first sentence of the test set for a quick preview.
        if i == 0:
            first_sentence_preview[ckpt_label] = {
                "gold": reference,
                "hyp_greedy": hyp_greedy,
                "hyp_kenlm": hyp_lm,
            }

        ref_m = safe(reference)
        ckpt_rows.append({
            "checkpoint": ckpt_label,
            "index": i,
            "gold": reference,
            "hyp_greedy": hyp_greedy,
            "hyp_kenlm": hyp_lm,
            "wer_greedy": jiwer_wer(ref_m, safe(hyp_greedy)),
            "cer_greedy": jiwer_cer(ref_m, safe(hyp_greedy)),
            "wer_kenlm":  jiwer_wer(ref_m, safe(hyp_lm)),
            "cer_kenlm":  jiwer_cer(ref_m, safe(hyp_lm)),
        })

        if (i + 1) % 25 == 0:
            print(f"  [{ckpt_label}] decoded {i+1}/{len(test_ds_prepared)}")

    rows_by_ckpt[ckpt_label] = ckpt_rows

    # ---- Print first-sentence preview for this checkpoint ----
    prev = first_sentence_preview[ckpt_label]
    print(f"\n  --- First test sentence preview [{ckpt_label}] ---")
    print(f"    GOLD       : {prev['gold']}")
    print(f"    NO-LM      : {prev['hyp_greedy']}")
    print(f"    KenLM      : {prev['hyp_kenlm']}")

    del ckpt_model
    if device == "cuda":
        torch.cuda.empty_cache()

# ---- Rank checkpoints by LM (KenLM) performance ----
# Primary: median WER+KenLM (coarse); tie-breakers: median CER+KenLM,
# then the more sensitive aggregate (corpus-level) WER+KenLM and CER+KenLM.
def ckpt_lm_scores(rows):
    df = pd.DataFrame(rows)
    return {
        "median_wer_kenlm": float(np.median(df["wer_kenlm"])),
        "median_cer_kenlm": float(np.median(df["cer_kenlm"])),
        "agg_wer_kenlm":    jiwer_wer(list(df["gold"]), list(df["hyp_kenlm"])),
        "agg_cer_kenlm":    jiwer_cer(list(df["gold"]), list(df["hyp_kenlm"])),
    }

ranking = []
for ckpt_label, rows in rows_by_ckpt.items():
    s = ckpt_lm_scores(rows)
    s["checkpoint"] = ckpt_label
    ranking.append(s)

ranking_df = pd.DataFrame(ranking).sort_values(
    by=["median_wer_kenlm", "median_cer_kenlm", "agg_wer_kenlm", "agg_cer_kenlm"],
    ascending=True,
).reset_index(drop=True)

best_ckpt_label = ranking_df.iloc[0]["checkpoint"]
best_ckpt_path  = dict(checkpoints)[best_ckpt_label]
print("\nLM-performance ranking (best -> worst):")
print(ranking_df.to_string(index=False))
print(f"\n>>> Best LM checkpoint: {best_ckpt_label}")

# ---- PROMOTE the best checkpoint to be the 'final' model on disk ----
# Copy the winner's weights/config into folderModelFiles itself, overwriting
# whatever the Trainer left there, so the saved 'final' model = best-LM model.
def promote_to_final(src_dir, dst_dir):
    # Copy model + config + processor files from the checkpoint into dst_dir.
    # We do NOT delete the numbered checkpoint folders.
    for fname in os.listdir(src_dir):
        # Skip optimizer/scheduler/trainer state -- not needed for inference.
        if fname in ("optimizer.pt", "scheduler.pt", "rng_state.pth",
                     "trainer_state.json", "training_args.bin", "scaler.pt"):
            continue
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))
    # Make sure the processor (tokenizer/vocab/feature extractor) is present.
    processor.save_pretrained(dst_dir)

promote_to_final(best_ckpt_path, folderModelFiles)
print(f"Promoted {best_ckpt_label} -> 'final' at {folderModelFiles}")

# ---- Build the combined results_df (checkpoints keep their real names) ----
# NOTE: We do NOT add a separate 'final' entry. The best checkpoint has been
# copied to disk as the 'final' model, but in the results we keep every
# checkpoint under its own name (e.g. checkpoint-1224) to avoid duplicate rows.
all_rows = []
for ckpt_label, rows in rows_by_ckpt.items():
    all_rows.extend(rows)

results_df = pd.DataFrame(all_rows)

# Record which numbered checkpoint was promoted to 'final' (used by Cell 17).
final_source_checkpoint = best_ckpt_label

print("\nTotal rows (checkpoints x sentences):", len(results_df))
print("Best checkpoint (promoted to 'final' on disk):", final_source_checkpoint)
print(results_df.head())

per_sentence_csv = os.path.join(
    folderLogFiles, f"{outputPrefix}-run{runId}-test-results.csv"
)
results_df.to_csv(per_sentence_csv, index=False, encoding="utf-8")
print("Wrote per-sentence results:", per_sentence_csv)

import os, re
import numpy as np
import pandas as pd
from jiwer import wer as jiwer_wer, cer as jiwer_cer

# 'final_source_checkpoint' is set in Cell 15. Fall back gracefully if missing.
final_source_checkpoint = globals().get("final_source_checkpoint", "unknown")

def _natural_ckpt_key(label):
    m = re.search(r"(\d+)", str(label))
    return int(m.group(1)) if m else 0

summary_rows = []
for ckpt_label, grp in results_df.groupby("checkpoint"):
    golds      = list(grp["gold"])
    hyp_greedy = list(grp["hyp_greedy"])
    hyp_kenlm  = list(grp["hyp_kenlm"])
    summary_rows.append({
        "checkpoint": ckpt_label,
        "n_sentences": len(grp),
        "median_wer_greedy": float(np.median(grp["wer_greedy"])),
        "median_cer_greedy": float(np.median(grp["cer_greedy"])),
        "median_wer_kenlm":  float(np.median(grp["wer_kenlm"])),
        "median_cer_kenlm":  float(np.median(grp["cer_kenlm"])),
        "agg_wer_greedy": jiwer_wer(golds, hyp_greedy),
        "agg_cer_greedy": jiwer_cer(golds, hyp_greedy),
        "agg_wer_kenlm":  jiwer_wer(golds, hyp_kenlm),
        "agg_cer_kenlm":  jiwer_cer(golds, hyp_kenlm),
    })

summary_df = pd.DataFrame(summary_rows)

# Rank by LM performance with a cascade of keys (coarse median -> sensitive aggregate).
summary_df = summary_df.sort_values(
    by=["median_wer_kenlm", "median_cer_kenlm", "agg_wer_kenlm", "agg_cer_kenlm"],
    ascending=True,
).reset_index(drop=True)
summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

# The rank-1 checkpoint is the one promoted to 'final'.
best = summary_df.iloc[0]

# ---- Write per-checkpoint summary CSV (chronological by checkpoint number) ----
summary_csv = os.path.join(
    folderLogFiles, f"{outputPrefix}-run{runId}-checkpoint-summary.csv"
)
summary_df_ordered = summary_df.copy()
summary_df_ordered["_sort"] = summary_df_ordered["checkpoint"].map(_natural_ckpt_key)
summary_df_ordered = summary_df_ordered.sort_values("_sort").drop(columns="_sort")
summary_df_ordered.to_csv(summary_csv, index=False, encoding="utf-8")
print("Wrote per-checkpoint summary CSV:", summary_csv)

# ---- Write the human-readable .txt summary ----
summary_txt = os.path.join(
    folderLogFiles, f"{outputPrefix}-run{runId}-summary.txt"
)
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(f"ASR language: {asrLang}\n")
    f.write(f"Condition: {condition} | Run: {runId} | Epochs: {desiredTrainEpochs}\n")
    f.write(f"Base model: {base_checkpoint}\n")
    f.write(f"KenLM: {ngrams}-gram ({filenameCorrectKenlmModel})\n")
    f.write(f"Checkpoints evaluated: {len(summary_df)}\n")
    f.write(f"Test sentences per checkpoint: {int(summary_df['n_sentences'].iloc[0])}\n")

    f.write("\n================ FINAL / BEST MODEL ================\n")
    f.write("(The 'final' saved model has been set to the checkpoint with the\n")
    f.write(" best KenLM performance.)\n")
    f.write(f"'final' is a copy of: {final_source_checkpoint}\n")
    f.write(f"Median WER (no LM)  : {best['median_wer_greedy']:.4f}\n")
    f.write(f"Median CER (no LM)  : {best['median_cer_greedy']:.4f}\n")
    f.write(f"Median WER (KenLM)  : {best['median_wer_kenlm']:.4f}\n")
    f.write(f"Median CER (KenLM)  : {best['median_cer_kenlm']:.4f}\n")
    f.write(f"Agg    WER (KenLM)  : {best['agg_wer_kenlm']:.4f}\n")
    f.write(f"Agg    CER (KenLM)  : {best['agg_cer_kenlm']:.4f}\n")

    f.write("\n================ ALL CHECKPOINTS ================\n")
    f.write("(sorted best -> worst by KenLM performance)\n\n")
    for _, row in summary_df.iterrows():
        ckpt_label = row["checkpoint"]
        tag = "  <-- promoted to 'final'" if ckpt_label == final_source_checkpoint else ""
        f.write(f"--- [rank {int(row['rank'])}] {ckpt_label}{tag} ---\n")
        f.write("  MEDIAN (per-sentence):\n")
        f.write(f"    WER no-LM: {row['median_wer_greedy']:.4f}  "
                f"CER no-LM: {row['median_cer_greedy']:.4f}\n")
        f.write(f"    WER KenLM: {row['median_wer_kenlm']:.4f}  "
                f"CER KenLM: {row['median_cer_kenlm']:.4f}\n")
        f.write("  AGGREGATE (corpus-level):\n")
        f.write(f"    WER no-LM: {row['agg_wer_greedy']:.4f}  "
                f"CER no-LM: {row['agg_cer_greedy']:.4f}\n")
        f.write(f"    WER KenLM: {row['agg_wer_kenlm']:.4f}  "
                f"CER KenLM: {row['agg_cer_kenlm']:.4f}\n")

        # First test sentence: gold, non-KenLM, KenLM.
        # Pull from results_df so this works even if Cell 15's preview dict
        # is not in memory (e.g. Cell 17 re-run on its own).
        first = results_df[
            (results_df["checkpoint"] == ckpt_label) & (results_df["index"] == 0)
        ]
        if len(first):
            fr = first.iloc[0]
            f.write("  FIRST TEST SENTENCE:\n")
            f.write(f"    GOLD : {fr['gold']}\n")
            f.write(f"    NO-LM: {fr['hyp_greedy']}\n")
            f.write(f"    KenLM: {fr['hyp_kenlm']}\n")
        f.write("\n")

print("Wrote summary:", summary_txt)
print("\n" + "="*50)
print(f"'final' model promoted from: {final_source_checkpoint}  "
      f"(median WER+KenLM = {best['median_wer_kenlm']:.4f}, "
      f"agg CER+KenLM = {best['agg_cer_kenlm']:.4f})")
print("="*50 + "\n")
with open(summary_txt) as f:
    print(f.read())

import os

# This will save the best model. This is an array, so
# you can add as many checkpoints to save as you wish.
saveCheckpoints = [str(final_source_checkpoint)]

# Erase previous models
modelFolder = datasetPath + "wav2vec2-model"
!rm -r {modelFolder}
!mkdir {modelFolder}

# Save new model

for s in saveCheckpoints:

  print("===== Saving " + s + " =====")

  originFolder = folderModelFiles + s
  destinationFolder = datasetPath + "wav2vec2-model/checkpoint-" + s
  !cp -r $originFolder $destinationFolder

!cp {folderModelFiles}preprocessor_config.json {datasetPath}wav2vec2-model/
!cp {folderModelFiles}special_tokens_map.json {datasetPath}wav2vec2-model/
!cp {folderModelFiles}tokenizer_config.json {datasetPath}wav2vec2-model/
!cp {folderModelFiles}vocab.json {datasetPath}wav2vec2-model/

!cp /content/{filenameCorrectKenlmModel} {datasetPath}wav2vec2-model/
!cp /content/{filenameKenlmModel} {datasetPath}wav2vec2-model/
!cp {corpusFile} {datasetPath}wav2vec2-model/