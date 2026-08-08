# ============================================================
# MODEL 1: ICD CODE PREDICTION USING BIOCLINICALBERT
# Folder-based split loading
#
# This model performs ICD code prediction using BioClinicalBERT
# on clinical text data.
#
# Each patient record is converted into a single text sequence
# and directly used as input to the model.
#
# The model does not explicitly treat the data as multi-document.
# All available information is merged into a flat text representation.
#
# The text is then passed to BioClinicalBERT for multi-label
# ICD classification.
#
# This is a Single-Document BioClinicalBERT model where each
# sample is treated as one input sequence.
#
# ------------------------------------------------------------
# Expected folder structure:
# data/
#   train/
#       train1.jsonl
#       train2.jsonl
#       train3.jsonl
#       train4.jsonl
#   val/
#       val.jsonl
#   test/
#       test.jsonl
#
# Outputs saved to:
#   ./model1_icd_output
# ============================================================

import os
import re
import json
import copy
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

from safetensors.torch import save_file

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "output_dir": "./model1_icd_output",
    "pretrained_model": "emilyalsentzer/Bio_ClinicalBERT",
    "max_length": 512,
    "batch_size": 8,
    "epochs": 5,
    "encoder_lr": 2e-5,
    "classifier_lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "grad_clip": 1.0,
    "threshold": 0.3,
    "min_label_freq": 2,
    "top_k_labels": 50,          # set None to keep all labels
    "num_workers": 0,
    "fp16": True,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("All outputs will be saved to:", CONFIG["output_dir"])


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])


# ============================================================
# HELPERS
# ============================================================
def natural_sort_key(path_obj):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path_obj.name)
    ]


def safe_str(x):
    if x is None:
        return ""
    return str(x)


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def make_state_dict_safe_for_saving(state_dict):
    safe_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            safe_state_dict[k] = v.detach().cpu().contiguous()
        else:
            safe_state_dict[k] = v
    return safe_state_dict


# ============================================================
# FILE LOADING
# ============================================================
def get_jsonl_files(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(folder.glob("*.jsonl"), key=natural_sort_key)
    if len(files) == 0:
        raise FileNotFoundError(f"No .jsonl files found in: {folder_path}")

    return files


def load_jsonl_file(file_path: Path) -> pd.DataFrame:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {file_path} at line {line_num}: {e}") from e

    return pd.DataFrame(rows)


def load_split_from_folder(folder_path: str, split_name: str) -> pd.DataFrame:
    files = get_jsonl_files(folder_path)

    print(f"\nLoading {split_name} files from: {folder_path}")
    for f in files:
        print(" -", f.name)

    dfs = []
    for file_path in files:
        df = load_jsonl_file(file_path)
        print(f"Loaded {file_path.name} -> shape: {df.shape}")
        dfs.append(df)

    split_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {split_name} shape: {split_df.shape}")
    return split_df


# ============================================================
# EVENT SERIALIZATION
# ============================================================
def format_lab_event(event: dict) -> str:
    value = event.get("value", {}) or {}
    label = safe_str(value.get("label", "unknown_lab")).strip()
    valuenum = value.get("valuenum", None)
    valueuom = safe_str(value.get("valueuom", "")).strip()
    abnormal = value.get("is_abnormal", None)
    rel_time = safe_float(event.get("relative_time_hrs", None))

    abnormal_text = ""
    if abnormal is True:
        abnormal_text = " abnormal"
    elif abnormal is False:
        abnormal_text = " normal"

    time_text = f" at {rel_time:.2f} hours" if rel_time is not None else ""

    if valuenum is None:
        return f"Lab {label}{time_text}{abnormal_text}."

    return f"Lab {label} {valuenum} {valueuom}{time_text}{abnormal_text}."


def format_text_event(event: dict, event_type: str) -> str:
    text = safe_str(event.get("text", "")).strip()
    text = " ".join(text.split())
    rel_time = safe_float(event.get("relative_time_hrs", None))
    time_text = f" at {rel_time:.2f} hours" if rel_time is not None else ""
    return f"{event_type.capitalize()} note{time_text}: {text}"


def format_generic_event(event: dict) -> str:
    event_type = safe_str(event.get("event_type", "unknown")).strip().lower()
    rel_time = safe_float(event.get("relative_time_hrs", None))
    time_text = f" at {rel_time:.2f} hours" if rel_time is not None else ""

    if "text" in event and event["text"] is not None:
        return format_text_event(event, event_type)

    value = event.get("value", None)
    if isinstance(value, dict):
        pairs = []
        for k, v in value.items():
            if v is not None:
                pairs.append(f"{k}={v}")
        joined = ", ".join(pairs)
        if joined:
            return f"{event_type.capitalize()} event{time_text}: {joined}"

    return f"{event_type.capitalize()} event{time_text}."


def serialize_events_to_text(events) -> str:
    if not isinstance(events, list):
        return ""

    def sort_key(x):
        rel = safe_float(x.get("relative_time_hrs", None))
        return float("inf") if rel is None else rel

    events_sorted = sorted(events, key=sort_key)
    lines = []

    for event in events_sorted:
        event_type = safe_str(event.get("event_type", "")).lower().strip()

        if event_type == "lab":
            lines.append(format_lab_event(event))
        elif event_type in {"radiology", "nursing", "discharge", "physician"}:
            lines.append(format_text_event(event, event_type))
        else:
            lines.append(format_generic_event(event))

    return " ".join([x for x in lines if x]).strip()


# ============================================================
# LABEL HANDLING
# ============================================================
def normalize_label_list(x):
    if x is None:
        return []

    if isinstance(x, list):
        out = []
        for item in x:
            if item is None:
                continue
            if isinstance(item, dict):
                for key in ["code", "label", "value", "name"]:
                    if key in item and item[key] is not None:
                        val = str(item[key]).strip()
                        if val:
                            out.append(val)
                        break
            else:
                val = str(item).strip()
                if val:
                    out.append(val)
        return out

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        for sep in [",", ";", "|"]:
            if sep in x:
                return [part.strip() for part in x.split(sep) if part.strip()]
        return [x]

    val = str(x).strip()
    return [val] if val else []


def extract_icd_labels(label_obj):
    if not isinstance(label_obj, dict):
        return []
    return normalize_label_list(label_obj.get("icd10", []))


def get_kept_labels_from_train(df_train: pd.DataFrame, label_col: str, min_freq: int, top_k):
    counter = Counter()
    for labels in df_train[label_col]:
        counter.update(labels)

    kept = [lab for lab, freq in counter.items() if freq >= min_freq]
    kept = sorted(kept, key=lambda x: counter[x], reverse=True)

    if top_k is not None:
        kept = kept[:top_k]

    return kept


def apply_label_filter(df: pd.DataFrame, label_col: str, kept_labels):
    kept_set = set(kept_labels)
    df = df.copy()
    df[label_col] = df[label_col].apply(lambda labs: [x for x in labs if x in kept_set])
    df = df[df[label_col].map(len) > 0].copy()
    return df


# ============================================================
# DATA PREP
# ============================================================
def preprocess_split(df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "events" not in df_raw.columns:
        raise ValueError(f"{split_name} dataset must contain 'events' column.")
    if "labels" not in df_raw.columns:
        raise ValueError(f"{split_name} dataset must contain 'labels' column.")

    df = df_raw.copy()
    df["text"] = df["events"].apply(serialize_events_to_text)
    df["target_labels"] = df["labels"].apply(extract_icd_labels)

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()
    df = df[df["target_labels"].map(len) > 0].copy()

    if len(df) == 0:
        raise ValueError(f"No usable rows left in {split_name} after preprocessing.")

    return df.reset_index(drop=True)


def build_datasets(train_raw: pd.DataFrame, val_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train_df = preprocess_split(train_raw, "train")
    val_df = preprocess_split(val_raw, "val")
    test_df = preprocess_split(test_raw, "test")

    kept_labels = get_kept_labels_from_train(
        train_df,
        label_col="target_labels",
        min_freq=CONFIG["min_label_freq"],
        top_k=CONFIG["top_k_labels"]
    )

    if len(kept_labels) == 0:
        raise ValueError(
            "No ICD labels left after filtering train data. "
            "Try reducing min_label_freq or setting top_k_labels=None."
        )

    train_df = apply_label_filter(train_df, "target_labels", kept_labels)
    val_df = apply_label_filter(val_df, "target_labels", kept_labels)
    test_df = apply_label_filter(test_df, "target_labels", kept_labels)

    if len(train_df) == 0:
        raise ValueError("No rows left in train split after label filtering.")
    if len(val_df) == 0:
        raise ValueError("No rows left in val split after label filtering.")
    if len(test_df) == 0:
        raise ValueError("No rows left in test split after label filtering.")

    mlb = MultiLabelBinarizer(classes=sorted(kept_labels))
    train_y = mlb.fit_transform(train_df["target_labels"])
    val_y = mlb.transform(val_df["target_labels"])
    test_y = mlb.transform(test_df["target_labels"])

    train_df["label_vector"] = list(train_y)
    val_df["label_vector"] = list(val_y)
    test_df["label_vector"] = list(test_y)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        mlb
    )


# ============================================================
# DATASET CLASS
# ============================================================
class ICDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = df["label_vector"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }


# ============================================================
# MODEL
# ============================================================
class ICDClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        return logits


# ============================================================
# METRICS
# ============================================================
def precision_at_k(y_true, y_probs, k=5):
    topk = np.argsort(-y_probs, axis=1)[:, :k]
    scores = []

    for i in range(y_true.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0].tolist())
        pred_set = set(topk[i].tolist())
        scores.append(len(true_set & pred_set) / k)

    return float(np.mean(scores))


def compute_metrics(y_true, y_probs, threshold=0.3):
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_at_5": precision_at_k(y_true, y_probs, 5),
        "precision_at_8": precision_at_k(y_true, y_probs, 8),
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.3):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    metrics = compute_metrics(y_true, y_probs, threshold)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics, y_true, y_probs


# ============================================================
# TRAIN
# ============================================================
def train_model(model, train_loader, val_loader):
    device = CONFIG["device"]
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": CONFIG["encoder_lr"]},
            {"params": model.classifier.parameters(), "lr": CONFIG["classifier_lr"]},
        ],
        weight_decay=CONFIG["weight_decay"]
    )

    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    use_amp = CONFIG["fp16"] and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_micro_f1 = -1.0
    best_state_dict = None
    history = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            avg_loss = total_train_loss / step
            progress.set_postfix({"train_loss": f"{avg_loss:.4f}"})

        train_loss = total_train_loss / max(1, len(train_loader))
        val_metrics, _, _ = evaluate(model, val_loader, device, CONFIG["threshold"])

        print("\n" + "=" * 70)
        print(f"Epoch {epoch+1}")
        print(f"Train Loss   : {train_loss:.4f}")
        print(f"Val Loss     : {val_metrics['loss']:.4f}")
        print(f"Val Micro F1 : {val_metrics['micro_f1']:.4f}")
        print(f"Val Macro F1 : {val_metrics['macro_f1']:.4f}")
        print(f"Val P@5      : {val_metrics['precision_at_5']:.4f}")
        print(f"Val P@8      : {val_metrics['precision_at_8']:.4f}")
        print("=" * 70)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            best_state_dict = copy.deepcopy(model.state_dict())
            print("Saved new best ICD model in memory.")

    if best_state_dict is None:
        raise RuntimeError("Training finished but no best model state was captured.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(CONFIG["output_dir"], "training_history.csv"), index=False)

    return best_state_dict, history_df


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Using device: {CONFIG['device']}")

    train_raw = load_split_from_folder(CONFIG["train_dir"], "train")
    val_raw = load_split_from_folder(CONFIG["val_dir"], "val")
    test_raw = load_split_from_folder(CONFIG["test_dir"], "test")

    train_df, val_df, test_df, mlb = build_datasets(train_raw, val_raw, test_raw)

    print("\nAfter preprocessing and label filtering:")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape  : {val_df.shape}")
    print(f"Test shape : {test_df.shape}")
    print(f"Number of ICD labels: {len(mlb.classes_)}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["pretrained_model"])

    train_dataset = ICDDataset(train_df, tokenizer, CONFIG["max_length"])
    val_dataset = ICDDataset(val_df, tokenizer, CONFIG["max_length"])
    test_dataset = ICDDataset(test_df, tokenizer, CONFIG["max_length"])

    pin_memory = CONFIG["device"] == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory
    )

    model = ICDClassifier(
        pretrained_model_name=CONFIG["pretrained_model"],
        num_labels=len(mlb.classes_),
        dropout=CONFIG["dropout"]
    )

    print("\nTraining ICD model...")
    best_state_dict, history_df = train_model(model, train_loader, val_loader)

    print("\nLoading best ICD model from memory...")
    model.load_state_dict(best_state_dict)
    model.to(CONFIG["device"])

    print("Evaluating ICD model on test set...")
    test_metrics, y_true, y_probs = evaluate(
        model,
        test_loader,
        CONFIG["device"],
        CONFIG["threshold"]
    )

    print("\n" + "=" * 70)
    print("ICD TEST RESULTS")
    print("=" * 70)
    for k, v in test_metrics.items():
        print(f"{k:20s}: {v:.4f}")
    print("=" * 70)

    output_dir = CONFIG["output_dir"]

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    with open(os.path.join(output_dir, "label_classes.json"), "w", encoding="utf-8") as f:
        json.dump(mlb.classes_.tolist(), f, indent=2)

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    np.save(os.path.join(output_dir, "test_y_true.npy"), y_true)
    np.save(os.path.join(output_dir, "test_y_probs.npy"), y_probs)

    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    safe_state_dict = make_state_dict_safe_for_saving(best_state_dict)
    save_file(safe_state_dict, os.path.join(output_dir, "best_model.safetensors"))

    print(f"\nAll outputs saved successfully to: {output_dir}")
    print(f"Training history rows: {len(history_df)}")


if __name__ == "__main__":
    main()