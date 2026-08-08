# ============================================================
# MODEL 2: CPT CODE PREDICTION USING LONGFORMER
# FOLDER-BASED VERSION
#
# Uses:
#   - Training data from:   data/train/train1.jsonl, train2.jsonl, ...
#   - Validation data from: data/val/val.jsonl
#   - Test data from:       data/test/test.jsonl
#
# Expected JSONL row format:
#   {
#       "discharge_text": ...,
#       "discharge_narrative": ...,
#       "events": [...],
#       "labels": {"cpt": [...], ...}
#   }
#
# Model 2 setup:
#   - discharge summary + radiology notes
#   - radiology sorted by relative_time_hrs
#   - merged into one long input
#   - Longformer for multi-label CPT prediction
#
# Output:
#   ./model2_cpt_output
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

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from safetensors.torch import save_file

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "output_dir": "./model2_cpt_output",
    "pretrained_model": "allenai/longformer-base-4096",
    "max_length": 1024,          # reduce to 512 if memory is tight
    "batch_size": 2,             # reduce to 1 if CUDA OOM
    "epochs": 3,
    "encoder_lr": 2e-5,
    "classifier_lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "grad_clip": 1.0,
    "threshold": 0.3,
    "min_label_freq": 1,         # use 1 for small/sample runs
    "top_k_labels": 50,          # set None to keep all CPT labels
    "num_workers": 0,            # safer on Windows
    "fp16": True,
    "seed": 42,
    "max_radiology_notes": 3,    # reduce to 2 for speed
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("All outputs will be saved to:", CONFIG["output_dir"])
print("Using device:", CONFIG["device"])


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


def safe_str(x) -> str:
    return "" if x is None else str(x)


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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

    if len(rows) == 0:
        raise ValueError(f"No rows found in file: {file_path}")

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
# TEXT BUILDING
# ============================================================
def build_model2_text(row) -> str:
    """
    Build Model 2 input:
    - discharge summary
    - radiology notes sorted by relative_time_hrs
    """
    sections = []

    discharge = safe_str(
        row.get("discharge_text") or row.get("discharge_narrative", "")
    ).strip()

    if discharge:
        sections.append("DISCHARGE SUMMARY:\n" + " ".join(discharge.split()))

    events = row.get("events") or []
    if isinstance(events, list):
        radiology_notes = []

        for e in events:
            event_type = safe_str(e.get("event_type", "")).lower().strip()
            if event_type != "radiology":
                continue

            text = safe_str(e.get("text", "")).strip()
            if not text:
                continue

            t = safe_float(e.get("relative_time_hrs"))
            if t is None:
                t = float("inf")

            radiology_notes.append({
                "t": t,
                "text": " ".join(text.split())
            })

        radiology_notes.sort(key=lambda x: x["t"])
        radiology_notes = radiology_notes[:CONFIG["max_radiology_notes"]]

        if radiology_notes:
            parts = []
            for i, note in enumerate(radiology_notes, start=1):
                if np.isfinite(note["t"]):
                    header = f"RADIOLOGY NOTE {i} AT {note['t']:.2f} HOURS:"
                else:
                    header = f"RADIOLOGY NOTE {i}:"
                parts.append(f"{header}\n{note['text']}")
            sections.append("\n\n".join(parts))

    return "\n\n".join([s for s in sections if s]).strip()


# ============================================================
# LABEL HANDLING
# ============================================================
def _item_to_str(item) -> str:
    if isinstance(item, dict):
        for key in ("code", "label", "value", "name"):
            val = safe_str(item.get(key, "")).strip()
            if val:
                return val
        return ""
    return safe_str(item).strip()


def normalize_label_list(x) -> list:
    if x is None:
        return []

    if isinstance(x, list):
        return [s for item in x for s in [_item_to_str(item)] if s]

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        for sep in (",", ";", "|"):
            if sep in x:
                return [p.strip() for p in x.split(sep) if p.strip()]
        return [x]

    val = safe_str(x).strip()
    return [val] if val else []


def extract_cpt_labels(label_obj) -> list:
    if not isinstance(label_obj, dict):
        return []
    return normalize_label_list(label_obj.get("cpt", []))


def get_kept_labels(df_train: pd.DataFrame, label_col: str, min_freq: int, top_k):
    counter = Counter()
    for labels in df_train[label_col]:
        counter.update(labels)

    kept = sorted(
        (lab for lab, freq in counter.items() if freq >= min_freq),
        key=lambda x: counter[x],
        reverse=True,
    )
    return kept[:top_k] if top_k is not None else kept


def filter_labels(df: pd.DataFrame, label_col: str, kept_set: set) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = df[label_col].apply(lambda labs: [x for x in labs if x in kept_set])
    return df[df[label_col].map(len) > 0].copy()


# ============================================================
# DATA PREP
# ============================================================
def preprocess_split(df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "labels" not in df_raw.columns:
        raise ValueError(f"{split_name} dataset must contain a 'labels' column.")

    df = df_raw.copy()
    df["text"] = df.apply(build_model2_text, axis=1).fillna("").astype(str)
    df["target_labels"] = df["labels"].apply(extract_cpt_labels)

    df = df[df["text"].str.len() > 0]
    df = df[df["target_labels"].map(len) > 0]

    if df.empty:
        raise ValueError(f"No usable rows left after preprocessing for {split_name}.")

    return df.reset_index(drop=True)


def build_datasets(train_raw: pd.DataFrame, val_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train_df = preprocess_split(train_raw, "train")
    val_df = preprocess_split(val_raw, "val")
    test_df = preprocess_split(test_raw, "test")

    kept_labels = get_kept_labels(
        train_df,
        label_col="target_labels",
        min_freq=CONFIG["min_label_freq"],
        top_k=CONFIG["top_k_labels"],
    )

    if not kept_labels:
        raise ValueError("No CPT labels left after filtering.")

    kept_set = set(kept_labels)

    train_df = filter_labels(train_df, "target_labels", kept_set)
    val_df = filter_labels(val_df, "target_labels", kept_set)
    test_df = filter_labels(test_df, "target_labels", kept_set)

    for split_name, split in (("train", train_df), ("val", val_df), ("test", test_df)):
        if split.empty:
            raise ValueError(f"No rows left in {split_name} split after label filtering.")

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
        mlb,
    )


# ============================================================
# TOKENIZATION
# ============================================================
def batch_tokenize(texts: list, tokenizer, max_length: int):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


# ============================================================
# DATASET
# ============================================================
class Model2CPTDataset(Dataset):
    def __init__(self, encodings, labels: np.ndarray):
        self.encodings = encodings
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ============================================================
# MODEL
# ============================================================
class LongformerCPTClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        return self.head(cls)


# ============================================================
# METRICS
# ============================================================
def precision_at_k(y_true: np.ndarray, y_probs: np.ndarray, k: int = 5) -> float:
    y_true = y_true.astype(int)
    topk = np.argpartition(-y_probs, k, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, topk, axis=1).sum(axis=1)
    return float(hits.mean()) / k


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.3) -> dict:
    y_true = y_true.astype(int)
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_at_5": precision_at_k(y_true, y_probs, 5),
        "precision_at_8": precision_at_k(y_true, y_probs, 8),
    }


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    thresholds = np.arange(0.1, 0.9, 0.05)
    y_true_bool = y_true.astype(bool)
    preds = (y_probs[None] >= thresholds[:, None, None])

    tp = (preds & y_true_bool[None]).sum(axis=(1, 2))
    fp = (preds & ~y_true_bool[None]).sum(axis=(1, 2))
    fn = (~preds & y_true_bool[None]).sum(axis=(1, 2))

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    best_idx = int(f1.argmax())
    return float(thresholds[best_idx]), float(f1[best_idx])


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.3):
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

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    metrics = compute_metrics(y_true, y_probs, threshold)
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics, y_true, y_probs


# ============================================================
# TRAINING
# ============================================================
def train_model(model, train_loader, val_loader):
    device = CONFIG["device"]
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": CONFIG["encoder_lr"]},
            {"params": model.head.parameters(), "lr": CONFIG["classifier_lr"]},
        ],
        weight_decay=CONFIG["weight_decay"],
    )

    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(CONFIG["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    use_amp = CONFIG["fp16"] and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_state_dict = None
    best_threshold = CONFIG["threshold"]
    history = []

    print("Starting training loop...")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        for step, batch in enumerate(progress, start=1):
            if step == 1:
                print("First batch loaded")

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
            progress.set_postfix({"train_loss": f"{total_train_loss / step:.4f}"})

        train_loss = total_train_loss / max(1, len(train_loader))

        _, val_y_true, val_y_probs = evaluate(model, val_loader, device, CONFIG["threshold"])
        tuned_threshold, _ = find_best_threshold(val_y_true, val_y_probs)
        val_metrics = compute_metrics(val_y_true, val_y_probs, tuned_threshold)

        print("\n" + "=" * 80)
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss        : {train_loss:.4f}")
        print(f"Val Best Threshold: {tuned_threshold:.2f}")
        print(f"Val Micro F1      : {val_metrics['micro_f1']:.4f}")
        print(f"Val Macro F1      : {val_metrics['macro_f1']:.4f}")
        print(f"Val P@5           : {val_metrics['precision_at_5']:.4f}")
        print(f"Val P@8           : {val_metrics['precision_at_8']:.4f}")
        print("=" * 80)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_threshold": tuned_threshold,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_threshold = tuned_threshold
            best_state_dict = copy.deepcopy(model.state_dict())
            print("Saved new best model.")

    if best_state_dict is None:
        raise RuntimeError("Training finished with no best model captured.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(CONFIG["output_dir"], "training_history.csv"), index=False)

    return best_state_dict, best_threshold, history_df


# ============================================================
# SAVE HELPERS
# ============================================================
def save_model(state_dict: dict, path: str):
    safe_sd = {
        k: v.detach().cpu().contiguous()
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
    }
    save_file(safe_sd, path)


def save_outputs(output_dir, config, mlb, test_metrics, y_true, y_probs,
                 train_df, val_df, test_df, history_df, best_state_dict):
    save_json(config, os.path.join(output_dir, "config.json"))
    save_json(mlb.classes_.tolist(), os.path.join(output_dir, "label_classes.json"))
    save_json({k: float(v) for k, v in test_metrics.items()},
              os.path.join(output_dir, "test_metrics.json"))

    np.save(os.path.join(output_dir, "test_y_true.npy"), y_true)
    np.save(os.path.join(output_dir, "test_y_probs.npy"), y_probs)

    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    save_model(best_state_dict, os.path.join(output_dir, "best_model.safetensors"))


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading datasets...")
    train_raw = load_split_from_folder(CONFIG["train_dir"], "train")
    val_raw = load_split_from_folder(CONFIG["val_dir"], "val")
    test_raw = load_split_from_folder(CONFIG["test_dir"], "test")

    train_df, val_df, test_df, mlb = build_datasets(train_raw, val_raw, test_raw)

    print(
        f"\nAfter preprocessing:\n"
        f"  Train: {train_df.shape}  Val: {val_df.shape}  Test: {test_df.shape}\n"
        f"  CPT labels: {len(mlb.classes_)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["pretrained_model"])

    print("\nTokenizing splits...")
    train_enc = batch_tokenize(train_df["text"].tolist(), tokenizer, CONFIG["max_length"])
    val_enc = batch_tokenize(val_df["text"].tolist(), tokenizer, CONFIG["max_length"])
    test_enc = batch_tokenize(test_df["text"].tolist(), tokenizer, CONFIG["max_length"])

    def make_loader(df, enc, shuffle):
        ds = Model2CPTDataset(enc, np.stack(df["label_vector"].values))
        return DataLoader(
            ds,
            batch_size=CONFIG["batch_size"],
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"],
            pin_memory=(CONFIG["device"] == "cuda")
        )

    train_loader = make_loader(train_df, train_enc, shuffle=True)
    val_loader = make_loader(val_df, val_enc, shuffle=False)
    test_loader = make_loader(test_df, test_enc, shuffle=False)

    model = LongformerCPTClassifier(
        CONFIG["pretrained_model"],
        len(mlb.classes_),
        CONFIG["dropout"]
    )

    print("\nTraining...")
    best_state_dict, best_threshold, history_df = train_model(model, train_loader, val_loader)

    model.load_state_dict(best_state_dict)
    model.to(CONFIG["device"])

    print(f"Best threshold: {best_threshold:.2f} — evaluating on test set...")
    test_metrics, y_true, y_probs = evaluate(
        model,
        test_loader,
        CONFIG["device"],
        best_threshold
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("=" * 80)

    full_config = copy.deepcopy(CONFIG)
    full_config["best_threshold"] = best_threshold

    save_outputs(
        CONFIG["output_dir"],
        full_config,
        mlb,
        test_metrics,
        y_true,
        y_probs,
        train_df,
        val_df,
        test_df,
        history_df,
        best_state_dict,
    )

    print(f"\nAll outputs saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()