import os
import re
import unicodedata
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from datasets import Dataset, Audio
from pyctcdecode import build_ctcdecoder
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from tqdm import tqdm

# Constants
TARGET_SAMPLE_RATE = 16000
apostrophe_variants = r"[’‘ʼʻ`´‛]"
chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\“\%\”\\(\)\[\]\{\}«»…]'

def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(apostrophe_variants, "'", text)
    text = re.sub(chars_to_remove_regex, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_tones(text):
    # Remove all digits (0-9) representing tones
    text = re.sub(r"\d", "", text)
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

def _detect_columns(df):
    audio_candidates = ["path", "audio", "wav", "file", "filename", "filepath", "audio_path"]
    text_candidates  = ["sentence", "text", "transcription", "transcript", "label", "target"]
    cols_lower = {c.lower(): c for c in df.columns}
    
    audio_col = None
    text_col = None
    for cand in audio_candidates:
        if cand in cols_lower:
            audio_col = cols_lower[cand]
            break
    for cand in text_candidates:
        if cand in cols_lower:
            text_col = cols_lower[cand]
            break
    if audio_col is None or text_col is None:
        raise ValueError(f"Could not auto-detect columns from: {list(df.columns)}")
    return audio_col, text_col

def _resolve_audio_path(p, dataset_path):
    p = str(p).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(dataset_path, p)
    if os.path.exists(cand):
        return cand
    for sub in ["wavs", "clips", "audio", "data"]:
        cand2 = os.path.join(dataset_path, sub, os.path.basename(p))
        if os.path.exists(cand2):
            return cand2
    return p

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Wav2Vec2 checkpoint on a test dataset.")
    parser.add_argument("--test-csv", type=str, default="cim-wav2vec2-test.csv", help="Path to the test CSV file.")
    parser.add_argument("--audio-dir", type=str, default="sentence_audio", help="Directory containing audio files.")
    parser.add_argument("--checkpoint", type=str, default="remote_output_w2v2/checkpoint-800", help="Path or HF repo ID to the model checkpoint.")
    parser.add_argument("--processor", type=str, default=None, help="Path or HF repo ID to the processor (defaults to checkpoint).")
    parser.add_argument("--arpa", type=str, default="output_w2v2/lm-cim-4-correct.arpa", help="Path to KenLM ARPA model.")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Hub authentication token.")
    parser.add_argument("--revision", type=str, default=None, help="Specific HF commit hash/branch/tag.")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    processor_path = args.processor or args.checkpoint
    checkpoint_path = args.checkpoint
    
    print(f"Loading test CSV: {args.test_csv}")
    df_test = _try_read_csv(args.test_csv)
    audio_col, text_col = _detect_columns(df_test)
    print(f"Using audio column: '{audio_col}' | text column: '{text_col}'")
    
    df_test[audio_col] = df_test[audio_col].apply(lambda p: _resolve_audio_path(p, args.audio_dir))
    df_test[text_col]  = df_test[text_col].apply(normalize_text)
    df_test.dropna(subset=[audio_col, text_col], inplace=True)
    
    print(f"Number of test items: {len(df_test)}")
    
    # Load processor and model
    if os.path.exists(processor_path):
        print(f"Loading processor from local path: {processor_path}...")
    else:
        print(f"Loading processor from Hugging Face Hub: {processor_path} (revision: {args.revision})...")
    processor = Wav2Vec2Processor.from_pretrained(processor_path, token=token, revision=args.revision)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from local path: {checkpoint_path}...")
    else:
        print(f"Loading model from Hugging Face Hub: {checkpoint_path} (revision: {args.revision})...")
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path, token=token, revision=args.revision)
    model.eval()
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Set up KenLM decoder
    use_lm = False
    if os.path.exists(args.arpa):
        print(f"Building CTC decoder using KenLM ARPA model from {args.arpa}...")
        vocab_dict_sorted = processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab_dict_sorted.items(), key=lambda kv: kv[1])
        labels = [t for t, _ in sorted_vocab]
        labels = ["" if t == "[PAD]" else (" " if t == "|" else t) for t in labels]
        
        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=args.arpa,
            alpha=0.5,
            beta=1.0,
        )
        processor_with_lm = Wav2Vec2ProcessorWithLM(
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            decoder=decoder,
        )
        use_lm = True
    else:
        print(f"Warning: ARPA model '{args.arpa}' not found. Evaluation will run without KenLM decoding.")
    
    # Prepare Dataset
    print("Preparing HuggingFace dataset...")
    data_dict = {
        "audio": df_test[audio_col].tolist(),
        "sentence": df_test[text_col].tolist()
    }
    from datasets import Features, Value
    features = Features({
        "audio": Audio(sampling_rate=TARGET_SAMPLE_RATE),
        "sentence": Value("string")
    })
    test_ds = Dataset.from_dict(data_dict, features=features)
    
    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        return batch
        
    test_ds_prepared = test_ds.map(prepare_batch, remove_columns=[c for c in test_ds.column_names if c != "sentence"], num_proc=1)
    
    # Run Inference
    print("Running batch inference...")
    results = []
    
    for idx, ex in enumerate(tqdm(test_ds_prepared)):
        input_values = torch.tensor([ex["input_values"]]).to(device)
        with torch.no_grad():
            logits = model(input_values=input_values).logits
        
        logits_np = logits.squeeze(0).cpu().numpy()
        pred_ids = np.argmax(logits_np, axis=-1)
        hyp_greedy = processor.decode(pred_ids).strip()
        
        gold = ex["sentence"]
        
        def safe(s):
            return s if s.strip() else " "
            
        wer_g = jiwer_wer(safe(gold), safe(hyp_greedy))
        cer_g = jiwer_cer(safe(gold), safe(hyp_greedy))
        
        row_res = {
            "idx": idx,
            "gold": gold,
            "greedy": hyp_greedy,
            "wer_greedy": wer_g,
            "cer_greedy": cer_g,
        }
        
        # Decode with LM if available
        if use_lm:
            hyp_lm = processor_with_lm.decoder.decode(logits_np).strip()
            wer_lm = jiwer_wer(safe(gold), safe(hyp_lm))
            cer_lm = jiwer_cer(safe(gold), safe(hyp_lm))
            row_res.update({
                "kenlm": hyp_lm,
                "wer_kenlm": wer_lm,
                "cer_kenlm": cer_lm
            })
            
        results.append(row_res)
        
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    golds = results_df["gold"].tolist()
    greedies = results_df["greedy"].tolist()
    
    overall_wer_greedy = jiwer_wer(golds, greedies)
    overall_cer_greedy = jiwer_cer(golds, greedies)
    
    print("\n" + "="*50)
    print("OVERALL RAW METRICS ON TEST SET (WITH TONES & ORIGINAL TRANSC):")
    print(f"Greedy: WER = {overall_wer_greedy:.4f} | CER = {overall_cer_greedy:.4f}")
    if use_lm:
        kenlms = results_df["kenlm"].tolist()
        overall_wer_kenlm = jiwer_wer(golds, kenlms)
        overall_cer_kenlm = jiwer_cer(golds, kenlms)
        print(f"KenLM:  WER = {overall_wer_kenlm:.4f} | CER = {overall_cer_kenlm:.4f}")
    print("="*50 + "\n")
    
    # Calculate tone-masked metrics
    golds_masked = [strip_tones(g) for g in golds]
    greedies_masked = [strip_tones(g) for g in greedies]
    
    overall_wer_greedy_masked = jiwer_wer(golds_masked, greedies_masked)
    overall_cer_greedy_masked = jiwer_cer(golds_masked, greedies_masked)
    
    print("="*50)
    print("OVERALL METRICS ON TEST SET WITH TONES MASKED (REMOVED):")
    print(f"Greedy (Masked): WER = {overall_wer_greedy_masked:.4f} | CER = {overall_cer_greedy_masked:.4f}")
    if use_lm:
        kenlms_masked = [strip_tones(k) for k in kenlms]
        overall_wer_kenlm_masked = jiwer_wer(golds_masked, kenlms_masked)
        overall_cer_kenlm_masked = jiwer_cer(golds_masked, kenlms_masked)
        print(f"KenLM (Masked):  WER = {overall_wer_kenlm_masked:.4f} | CER = {overall_cer_kenlm_masked:.4f}")
    print("="*50 + "\n")
    
    # Save output to a file
    results_df.to_csv("test_inference_results.csv", index=False)
    print("Saved test results to test_inference_results.csv")
    
    # Display first few comparisons
    print("\nSample Comparisons:")
    for i in range(min(15, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n[{i}] Gold:   {row['gold']}")
        print(f"    Greedy: {row['greedy']} (WER: {row['wer_greedy']:.2f})")
        if use_lm:
            print(f"    KenLM:  {row['kenlm']} (WER: {row['wer_kenlm']:.2f})")

if __name__ == "__main__":
    main()
