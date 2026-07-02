import argparse
import os
import re
import json
import unicodedata
import pandas as pd
import numpy as np
import torch
import subprocess
import shutil

from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset, Audio
import evaluate
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 Model")
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--valid_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--epochs", type=int, default=34)
    parser.add_argument("--ngrams", type=int, default=4)
    parser.add_argument("--run_id", type=str, default="01")
    parser.add_argument("--lang_prefix", type=str, default="cim")
    parser.add_argument("--lmplz_path", type=str, default=None, help="Path to lmplz executable")
    return parser.parse_args()

def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[’‘ʼʻ`´‛]", "'", text)
    text = re.sub(r'[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]', "", text)
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

def build_vocab(*dfs):
    all_text = " ".join(pd.concat([d["sentence"] for d in dfs]).tolist())
    vocab = sorted(set(all_text))
    return vocab

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

def main():
    args = parse_args()
    
    sandbox_dir = args.sandbox
    model_dir = os.path.join(sandbox_dir, "wav2vec2-model")
    os.makedirs(model_dir, exist_ok=True)
    
    csv_train = os.path.join(sandbox_dir, args.train_csv)
    csv_valid = os.path.join(sandbox_dir, args.valid_csv)
    csv_test = os.path.join(sandbox_dir, args.test_csv)
    corpus_file = os.path.join(model_dir, f"{args.lang_prefix}-corpus.txt")
    
    df_train = _try_read_csv(csv_train)
    df_valid = _try_read_csv(csv_valid)
    df_test = _try_read_csv(csv_test)
    
    # Text normalization
    for df in (df_train, df_valid, df_test):
        df["sentence"] = df["sentence"].apply(normalize_text)
        df.dropna(subset=["path", "sentence"], inplace=True)
        # Fix paths to be absolute
        def resolve_path(p):
            if os.path.isabs(p):
                return p
            p = p.replace("\\", "/")
            base_sandbox = os.path.basename(os.path.normpath(sandbox_dir))
            if p.startswith(base_sandbox + "/"):
                p = p.split("/", 1)[1]
            return os.path.abspath(os.path.join(sandbox_dir, p))
        df["path"] = df["path"].apply(resolve_path)
    
    # Vocab
    vocab_list = build_vocab(df_train, df_valid, df_test)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    vocab_path = os.path.join(model_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
        
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(model_dir)
    
    print("Skipping KenLM language model building as requested.")

    def df_to_ds(df):
        ds = Dataset.from_pandas(df[["path", "sentence"]].reset_index(drop=True).rename(columns={"path": "audio"}))
        return ds

    train_ds = df_to_ds(df_train)
    valid_ds = df_to_ds(df_valid)
    
    def prepare_batch(batch):
        import soundfile as sf
        import librosa
        try:
            speech, sr = sf.read(batch["audio"])
            if len(speech.shape) > 1:
                speech = speech.mean(axis=-1)
            if sr != 16000:
                speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        except Exception as e:
            # Fallback if soundfile fails
            speech, sr = librosa.load(batch["audio"], sr=16000)
            
        batch["input_values"] = processor(speech, sampling_rate=16000).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names, num_proc=4)
    valid_ds = valid_ds.map(prepare_batch, remove_columns=valid_ds.column_names, num_proc=4)
    
    MAX_INPUT_LENGTH = 16000 * 20
    train_ds = train_ds.filter(lambda x: x < MAX_INPUT_LENGTH, input_columns=["input_length"])

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
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

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
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

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=400,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        warmup_steps=100,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()
    
    # Save best model to root of model_dir
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)
    print(f"Training complete. Model saved to {model_dir}")

if __name__ == "__main__":
    main()
