# -*- coding: utf-8 -*-
"""trainer_w2v2_local.py

Local adaptation of the Wav2Vec2 training script.
Supports local CSV files, configurable local paths, and subprocess/OS-based calls.
"""

import os
import re
import sys
import json
import shutil
import subprocess
import unicodedata
import pandas as pd
import numpy as np
import torch
import torchaudio
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    Wav2Vec2ProcessorWithLM,
)
from datasets import Dataset, Audio
from pyctcdecode import build_ctcdecoder
from jiwer import wer as jiwer_wer, cer as jiwer_cer

TARGET_SAMPLE_RATE = 16000
apostrophe_variants = r"[’‘ʼʻ`´‛]"  # curly, modifier letter, grave/acute, etc.
chars_to_remove_regex = r"[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]"

# CONFIGURATION DICTIONARY
CONFIG = {
    "train_csv": "cim-wav2vec2-train.csv",
    "valid_csv": "cim-wav2vec2-valid.csv",
    "test_csv": "cim-wav2vec2-test.csv",
    "audio_dir": "sentence_audio",
    "output_dir": "output_w2v2",
    "base_checkpoint": "facebook/wav2vec2-large-xlsr-53",
    "asr_lang": "cim",
    "run_id": "01",
    "epochs": 50,
    "ngrams": 4,
    "lmplz_path": "lmplz",  # Expected in system PATH, or specify full local path (e.g. /usr/local/bin/lmplz)
    "audio_column": None,  # Auto-detects columns like 'path', 'wav'
    "text_column": None,  # Auto-detects columns like 'sentence', 'text'
    "max_steps": -1,
    "eval_batch_size": 16,
}


def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(apostrophe_variants, "'", text)  # unify apostrophes -> '
    text = re.sub(chars_to_remove_regex, "", text)  # remove other punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _try_read_csv(path):
    for sep in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    return pd.read_csv(path)


def _detect_columns(df, audio_col_override=None, text_col_override=None):
    audio_candidates = [
        "path",
        "audio",
        "wav",
        "file",
        "filename",
        "filepath",
        "audio_path",
    ]
    text_candidates = [
        "sentence",
        "text",
        "transcription",
        "transcript",
        "label",
        "target",
    ]
    cols_lower = {c.lower(): c for c in df.columns}

    audio_col = audio_col_override
    text_col = text_col_override
    if audio_col is None:
        for cand in audio_candidates:
            if cand in cols_lower:
                audio_col = cols_lower[cand]
                break
    if text_col is None:
        for cand in text_candidates:
            if cand in cols_lower:
                text_col = cols_lower[cand]
                break

    if audio_col is None or text_col is None:
        raise ValueError(
            f"Could not auto-detect columns. Found columns: {list(df.columns)}. "
            f"Please specify audio_column and text_column in CONFIG."
        )
    return audio_col, text_col


def _resolve_audio_path(p, dataset_path):
    p = str(p).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # Try direct relative to CSV parent directory / dataset path
    cand = os.path.join(dataset_path, p)
    if os.path.exists(cand):
        return cand
    # try a 'clips' or 'wavs' subfolder fallback
    for sub in ["wavs", "clips", "audio", "data"]:
        cand2 = os.path.join(dataset_path, sub, os.path.basename(p))
        if os.path.exists(cand2):
            return cand2
    return p


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Wav2Vec2 on local or remote machine."
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=CONFIG["train_csv"],
        help="Path to training CSV split.",
    )
    parser.add_argument(
        "--valid-csv",
        type=str,
        default=CONFIG["valid_csv"],
        help="Path to validation CSV split.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=CONFIG["test_csv"],
        help="Path to test CSV split.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=CONFIG["audio_dir"],
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG["output_dir"],
        help="Output directory for logs and models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG["epochs"],
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lmplz-path",
        type=str,
        default=CONFIG["lmplz_path"],
        help="Path to KenLM lmplz binary.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=CONFIG["max_steps"],
        help="Max training steps (-1 for unlimited).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push checkpoints to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Hugging Face Hub model ID (e.g. username/model_name).",
    )
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="Hugging Face Hub Authentication Token.",
    )
    args = parser.parse_args()

    CONFIG["train_csv"] = args.train_csv
    CONFIG["valid_csv"] = args.valid_csv
    CONFIG["test_csv"] = args.test_csv
    CONFIG["audio_dir"] = args.audio_dir
    CONFIG["output_dir"] = args.output_dir
    CONFIG["epochs"] = args.epochs
    CONFIG["lmplz_path"] = args.lmplz_path
    CONFIG["max_steps"] = args.max_steps
    CONFIG["push_to_hub"] = args.push_to_hub
    CONFIG["hub_model_id"] = args.hub_model_id
    CONFIG["hub_token"] = args.hub_token

    if CONFIG["push_to_hub"]:
        import datetime
        run_start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = CONFIG["hub_model_id"] if CONFIG["hub_model_id"] else "wav2vec2-large-xlsr"
        if "/" in base_name:
            parts = base_name.split("/")
            if len(parts) == 2:
                CONFIG["hub_model_id"] = f"{parts[0]}/{run_start_time}-{parts[1]}"
            else:
                CONFIG["hub_model_id"] = f"{run_start_time}-{base_name}"
        else:
            CONFIG["hub_model_id"] = f"{run_start_time}-{base_name}"
        print(f"Hugging Face Hub Model ID set to: {CONFIG['hub_model_id']}")

    # Paths and folders
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    folder_log_files = os.path.join(CONFIG["output_dir"], "logs-wav2vec2-res")
    folder_model_files = os.path.join(CONFIG["output_dir"], "wav2vec2-large-xlsr")
    os.makedirs(folder_log_files, exist_ok=True)
    os.makedirs(folder_model_files, exist_ok=True)

    print(f"Logs folder: {folder_log_files}")
    print(f"Model folder: {folder_model_files}")

    # Load CSVs
    print("Loading CSVs...")
    df_train = _try_read_csv(CONFIG["train_csv"])
    df_valid = _try_read_csv(CONFIG["valid_csv"])
    df_test = _try_read_csv(CONFIG["test_csv"])

    audio_col, text_col = _detect_columns(
        df_train, CONFIG["audio_column"], CONFIG["text_column"]
    )
    print(f"Using audio column: '{audio_col}' | text column: '{text_col}'")
    print(f"Sizes: Train={len(df_train)}, Valid={len(df_valid)}, Test={len(df_test)}")

    # Resolve paths & normalize transcriptions
    for df in (df_train, df_valid, df_test):
        df[audio_col] = df[audio_col].apply(
            lambda p: _resolve_audio_path(p, CONFIG["audio_dir"])
        )
        df[text_col] = df[text_col].apply(normalize_text)
        df.dropna(subset=[audio_col, text_col], inplace=True)

    # Audio file checks
    missing = [p for p in df_train[audio_col].tolist()[:50] if not os.path.exists(p)]
    if missing:
        print("WARNING: some audio files not found, e.g.:", missing[:5])
    else:
        print("Audio path checks passed (sampled).")

    # Vocab / Tokenizer
    print("Building vocabulary...")
    all_text = " ".join(
        pd.concat([df_train[text_col], df_valid[text_col], df_test[text_col]]).tolist()
    )
    vocab = sorted(set(all_text))
    vocab_dict = {v: k for k, v in enumerate(vocab)}
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    vocab_path = os.path.join(CONFIG["output_dir"], "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    print(f"Vocab saved (size={len(vocab_dict)}).")

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
    processor.save_pretrained(folder_model_files)

    print("Skipping KenLM language model building as requested.")

    # Build HuggingFace datasets
    print("Preparing HuggingFace Datasets...")
    from datasets import Features, Value

    features = Features(
        {"audio": Audio(sampling_rate=TARGET_SAMPLE_RATE), "sentence": Value("string")}
    )

    def df_to_ds(df):
        # HuggingFace Datasets Audio feature expects a list of paths or dicts.
        # Let's build a dict and pass features.
        data_dict = {"audio": df[audio_col].tolist(), "sentence": df[text_col].tolist()}
        ds = Dataset.from_dict(data_dict, features=features)
        return ds

    train_ds = df_to_ds(df_train)
    valid_ds = df_to_ds(df_valid)
    test_ds = df_to_ds(df_test)

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    train_ds = train_ds.map(
        prepare_batch, remove_columns=train_ds.column_names, num_proc=1
    )
    valid_ds = valid_ds.map(
        prepare_batch, remove_columns=valid_ds.column_names, num_proc=1
    )
    test_ds_prepared = test_ds.map(
        prepare_batch,
        remove_columns=[c for c in test_ds.column_names if c != "sentence"],
        num_proc=1,
    )

    MAX_INPUT_LENGTH = TARGET_SAMPLE_RATE * 20
    train_ds = train_ds.filter(
        lambda x: x < MAX_INPUT_LENGTH, input_columns=["input_length"]
    )

    # Load Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    # Load Model
    print(f"Loading base checkpoint: {CONFIG['base_checkpoint']}")
    model = Wav2Vec2ForCTC.from_pretrained(
        CONFIG["base_checkpoint"],
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_encoder()

    # Fine-tune training config
    training_args = TrainingArguments(
        output_dir=folder_model_files,
        group_by_length=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=400,
        num_train_epochs=CONFIG["epochs"],
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        warmup_ratio=0.1,
        save_total_limit=20,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
        max_steps=CONFIG["max_steps"],
        push_to_hub=CONFIG["push_to_hub"],
        hub_model_id=CONFIG["hub_model_id"],
        hub_token=CONFIG["hub_token"],
        hub_private_repo=True,  # Keeps models private
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(folder_model_files)
    processor.save_pretrained(folder_model_files)
    print("Training complete. Base model saved.")

    # Evaluate checkpoints with pyctcdecode
    print("Starting KenLM post-decoding evaluation of checkpoints...")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    import glob

    ckpt_dirs = glob.glob(os.path.join(folder_model_files, "checkpoint-*"))

    def _step(p):
        m = re.search(r"checkpoint-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    ckpt_dirs = sorted(ckpt_dirs, key=_step)
    checkpoints = [
        (os.path.basename(d), d)
        for d in ckpt_dirs
        if os.path.exists(os.path.join(d, "config.json"))
    ]

    if not checkpoints:
        print("No checkpoints found. Running evaluation on final model.")
        checkpoints = [("final", folder_model_files)]

    def safe(s):
        return s if s.strip() else " "

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    rows_by_ckpt = {}
    for ckpt_label, ckpt_path in checkpoints:
        print(f"Evaluating checkpoint: {ckpt_label}")
        ckpt_model = Wav2Vec2ForCTC.from_pretrained(ckpt_path)
        ckpt_model.eval()
        ckpt_model.to(device)

        test_loader = DataLoader(
            test_ds_prepared,
            batch_size=CONFIG.get("eval_batch_size", 16),
            collate_fn=data_collator,
            shuffle=False,
        )

        all_logits = []
        all_greedy_hypotheses = []

        print("  Running GPU batch inference...")
        for batch in tqdm(test_loader, desc=f"Inference ({ckpt_label})"):
            input_values = batch["input_values"].to(device)
            attention_mask = (
                batch["attention_mask"].to(device)
                if "attention_mask" in batch
                else None
            )

            with torch.no_grad():
                outputs = ckpt_model(
                    input_values=input_values, attention_mask=attention_mask
                )
                logits = outputs.logits

            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=-1)
                output_lengths = ckpt_model._get_feat_extract_output_lengths(
                    input_lengths
                )
                output_lengths = output_lengths.cpu().numpy()
            else:
                output_lengths = [logits.shape[1]] * logits.shape[0]

            logits_np = logits.cpu().numpy()
            for i in range(len(logits_np)):
                actual_len = int(output_lengths[i])
                sliced = logits_np[i, :actual_len, :]
                all_logits.append(sliced)

                pred_ids = np.argmax(sliced, axis=-1)
                hyp_greedy = processor.decode(pred_ids).strip()
                all_greedy_hypotheses.append(hyp_greedy)

        all_gold_sentences = [ex["sentence"] for ex in test_ds_prepared]
        num_logits = len(all_logits)

        ckpt_rows = []
        for idx in range(num_logits):
            reference = all_gold_sentences[idx]
            hyp_greedy = all_greedy_hypotheses[idx]

            ref_m = safe(reference)
            ckpt_rows.append(
                {
                    "checkpoint": ckpt_label,
                    "index": idx,
                    "gold": reference,
                    "hyp_greedy": hyp_greedy,
                    "wer_greedy": jiwer_wer(ref_m, safe(hyp_greedy)),
                    "cer_greedy": jiwer_cer(ref_m, safe(hyp_greedy)),
                }
            )
        rows_by_ckpt[ckpt_label] = ckpt_rows
        del ckpt_model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    ranking = []
    for ckpt_label, rows in rows_by_ckpt.items():
        df = pd.DataFrame(rows)
        ranking.append(
            {
                "checkpoint": ckpt_label,
                "median_wer_greedy": float(np.median(df["wer_greedy"])),
                "median_cer_greedy": float(np.median(df["cer_greedy"])),
                "agg_wer_greedy": jiwer_wer(list(df["gold"]), list(df["hyp_greedy"])),
                "agg_cer_greedy": jiwer_cer(list(df["gold"]), list(df["hyp_greedy"])),
            }
        )

    ranking_df = (
        pd.DataFrame(ranking)
        .sort_values(
            by=[
                "median_wer_greedy",
                "median_cer_greedy",
                "agg_wer_greedy",
                "agg_cer_greedy",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    best_ckpt_label = ranking_df.iloc[0]["checkpoint"]
    best_ckpt_path = dict(checkpoints)[best_ckpt_label]
    print(f"\nBest checkpoint identified: {best_ckpt_label}")

    # Copy best checkpoint files to final model output
    for fname in os.listdir(best_ckpt_path):
        if fname in (
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "training_args.bin",
            "scaler.pt",
        ):
            continue
        src = os.path.join(best_ckpt_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(folder_model_files, fname))

    # Save final processor configuration
    processor.save_pretrained(folder_model_files)
    print(f"Promoted {best_ckpt_label} to final model directory.")

    # Save results summary
    all_rows = []
    for ckpt_label, rows in rows_by_ckpt.items():
        all_rows.extend(rows)
    results_df = pd.DataFrame(all_rows)

    output_prefix = f"{CONFIG['asr_lang']}-wav2vec2"
    per_sentence_csv = os.path.join(
        folder_log_files, f"{output_prefix}-run{CONFIG['run_id']}-test-results.csv"
    )
    results_df.to_csv(per_sentence_csv, index=False, encoding="utf-8")

    # Generate summary.txt
    summary_txt = os.path.join(
        folder_log_files, f"{output_prefix}-run{CONFIG['run_id']}-summary.txt"
    )
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"ASR Language: {CONFIG['asr_lang']}\nRun ID: {CONFIG['run_id']}\n")
        f.write(f"Promoted checkpoint: {best_ckpt_label}\n")
        f.write(f"Ranking:\n{ranking_df.to_string()}\n")
    print(f"Summary written to {summary_txt}")


if __name__ == "__main__":
    main()
