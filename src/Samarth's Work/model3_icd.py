# ============================================================
# MODEL 3: ICD CODE PREDICTION USING TEMPORAL CODING
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
#       "labels": {
#           "icd10": [...],
#           "cpt": [...]
#       }
#   }
#
# Model 3 setup:
#   - clinical events sorted by relative_time_hrs
#   - discharge summary appended as final event
#   - BioClinicalBERT encodes each event
#   - BiLSTM models event sequence
#   - temporal attention pools sequence outputs
#   - multi-label ICD prediction
#
# Output:
#   ./model3_icd_output
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
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
# 2. CONFIGURATION
# ============================================================
CONFIG = {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "output_dir": "./model3_icd_output",
    "pretrained_model": "emilyalsentzer/Bio_ClinicalBERT",

    # Temporal settings
    "max_event_length": 128,
    "max_events": 32,

    # Training settings
    "batch_size": 2,
    "epochs": 8,
    "patience": 3,
    "encoder_lr": 2e-5,
    "lstm_lr": 8e-4,
    "head_lr": 8e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.35,
    "grad_clip": 1.0,

    # Architecture
    "hidden_dim": 256,
    "num_lstm_layers": 1,
    "task_hidden_dim": 256,
    "bidirectional": True,

    # Label filtering
    "min_label_freq": 1,
    "top_k_labels": 40,

    # Threshold search
    "threshold_grid": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],

    # Top-k search
    "topk_grid": [8, 10, 12, 15],

    # Hard cap
    "max_predictions": 12,

    # Positive class weighting
    "min_pos_weight": 1.0,
    "max_pos_weight": 10.0,

    # Misc
    "num_workers": 0,
    "fp16": True,
    "seed": 42,
    "append_discharge_as_last_event": True,
    "freeze_encoder_first_epoch": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print("All outputs will be saved to:", CONFIG["output_dir"])
print("Using device:", CONFIG["device"])


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])


# ============================================================
# 4. GENERAL HELPER FUNCTIONS
# ============================================================
def natural_sort_key(path_obj: Path):
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


def make_state_dict_safe_for_saving(state_dict):
    safe_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            safe_state_dict[k] = v.detach().cpu().contiguous()
        else:
            safe_state_dict[k] = v
    return safe_state_dict


# ============================================================
# 5. FILE LOADING
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
# 6. TEMPORAL EVENT SERIALIZATION
# ============================================================
def format_lab_event(event: dict) -> str:
    value = event.get("value", {}) or {}
    label = safe_str(value.get("label", "unknown_lab")).strip()
    valuenum = value.get("valuenum", None)
    valueuom = safe_str(value.get("valueuom", "")).strip()
    abnormal = value.get("is_abnormal", None)
    rel_time = safe_float(event.get("relative_time_hrs", None))

    if abnormal is True:
        abnormal_text = "abnormal"
    elif abnormal is False:
        abnormal_text = "normal"
    else:
        abnormal_text = "unknown_status"

    time_text = f"{rel_time:.2f} hours" if rel_time is not None else "unknown_time"

    if valuenum is None:
        return f"[LAB] [TIME={time_text}] {label} [{abnormal_text}]"
    return f"[LAB] [TIME={time_text}] {label} = {valuenum} {valueuom} [{abnormal_text}]"


def format_text_event(event: dict, event_type: str) -> str:
    text = safe_str(event.get("text", "")).strip()
    text = " ".join(text.split())
    rel_time = safe_float(event.get("relative_time_hrs", None))
    time_text = f"{rel_time:.2f} hours" if rel_time is not None else "unknown_time"
    return f"[{event_type.upper()}] [TIME={time_text}] {text}"


def format_generic_event(event: dict) -> str:
    event_type = safe_str(event.get("event_type", "unknown")).strip().lower()
    rel_time = safe_float(event.get("relative_time_hrs", None))
    time_text = f"{rel_time:.2f} hours" if rel_time is not None else "unknown_time"

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
            return f"[{event_type.upper()}] [TIME={time_text}] {joined}"

    return f"[{event_type.upper()}] [TIME={time_text}]"


def serialize_events_to_list(events):
    if not isinstance(events, list):
        return []

    def sort_key(x):
        rel = safe_float(x.get("relative_time_hrs", None))
        return float("inf") if rel is None else rel

    events_sorted = sorted(events, key=sort_key)
    event_texts = []

    for event in events_sorted:
        event_type = safe_str(event.get("event_type", "")).lower().strip()

        if event_type == "lab":
            event_texts.append(format_lab_event(event))
        elif event_type in {"radiology", "nursing", "discharge", "physician"}:
            event_texts.append(format_text_event(event, event_type))
        else:
            event_texts.append(format_generic_event(event))

    return [x for x in event_texts if x]


# ============================================================
# 7. LABEL PROCESSING
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


# ============================================================
# 8. PREPROCESS DATA
# ============================================================
def preprocess_split(df_raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "events" not in df_raw.columns:
        raise ValueError(f"{split_name} dataset must contain 'events' column.")
    if "labels" not in df_raw.columns:
        raise ValueError(f"{split_name} dataset must contain 'labels' column.")

    df = df_raw.copy()
    df["event_texts"] = df["events"].apply(serialize_events_to_list)

    def get_discharge_text(row):
        discharge_text = safe_str(row.get("discharge_narrative", "")).strip()
        if not discharge_text:
            discharge_text = safe_str(row.get("discharge_text", "")).strip()
        discharge_text = " ".join(discharge_text.split())
        return discharge_text

    df["discharge_note"] = df.apply(get_discharge_text, axis=1)
    df["icd_labels"] = df["labels"].apply(extract_icd_labels)
    df["num_events"] = df["event_texts"].apply(len)

    if CONFIG["append_discharge_as_last_event"]:
        df = df[(df["num_events"] > 0) | (df["discharge_note"].str.len() > 0)].copy()
    else:
        df = df[df["num_events"] > 0].copy()

    df = df[df["icd_labels"].map(len) > 0].copy()

    if len(df) == 0:
        raise ValueError(f"No usable rows left in {split_name} after preprocessing.")

    return df.reset_index(drop=True)


# ============================================================
# 9. TEMPORAL DATASET CLASS
# ============================================================
class TemporalICDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label_binarizer,
        max_event_length: int,
        max_events: int,
        append_discharge_as_last_event: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_binarizer = label_binarizer
        self.max_event_length = max_event_length
        self.max_events = max_events
        self.append_discharge_as_last_event = append_discharge_as_last_event

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        event_texts = list(row["event_texts"]) if isinstance(row["event_texts"], list) else []
        discharge_note = safe_str(row["discharge_note"]).strip()

        if self.append_discharge_as_last_event and discharge_note:
            event_texts = event_texts[: self.max_events - 1] + [f"[DISCHARGE_SUMMARY] {discharge_note}"]
        else:
            event_texts = event_texts[: self.max_events]

        if len(event_texts) == 0:
            event_texts = ["[EMPTY_EVENT] no clinical events available"]

        seq_len = min(len(event_texts), self.max_events)

        enc = self.tokenizer(
            event_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_event_length,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        if seq_len < self.max_events:
            pad_events = self.max_events - seq_len
            input_pad = torch.zeros((pad_events, self.max_event_length), dtype=torch.long)
            mask_pad = torch.zeros((pad_events, self.max_event_length), dtype=torch.long)

            input_ids = torch.cat([input_ids, input_pad], dim=0)
            attention_mask = torch.cat([attention_mask, mask_pad], dim=0)

        event_mask = torch.zeros(self.max_events, dtype=torch.long)
        event_mask[:seq_len] = 1

        labels = self.label_binarizer.transform([row["icd_labels"]])[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "event_mask": event_mask,
            "labels": torch.tensor(labels, dtype=torch.float),
        }


# ============================================================
# 10. MODEL DEFINITION
# ============================================================
class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence_output, mask):
        scores = self.score(sequence_output).squeeze(-1)
        scores = scores.float()
        scores = scores.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(scores, dim=1).to(sequence_output.dtype)
        pooled = torch.bmm(attn.unsqueeze(1), sequence_output).squeeze(1)
        return pooled, attn


class TemporalICDClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        hidden_dim: int = 256,
        num_lstm_layers: int = 1,
        task_hidden_dim: int = 256,
        dropout: float = 0.35,
        bidirectional: bool = True
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        encoder_hidden_size = self.encoder.config.hidden_size
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.lstm = nn.LSTM(
            input_size=encoder_hidden_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        self.temporal_attention = TemporalAttentionPooling(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, task_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, event_mask):
        batch_size, max_events, max_event_length = input_ids.shape

        flat_input_ids = input_ids.view(batch_size * max_events, max_event_length)
        flat_attention_mask = attention_mask.view(batch_size * max_events, max_event_length)

        encoder_outputs = self.encoder(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )

        cls_embeddings = encoder_outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(batch_size, max_events, -1)

        lengths = event_mask.sum(dim=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            cls_embeddings,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_events
        )

        pooled_output, _ = self.temporal_attention(lstm_output, event_mask)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return logits


# ============================================================
# 11. POSITIVE CLASS WEIGHTS
# ============================================================
def compute_pos_weights(df: pd.DataFrame, label_col: str, mlb, min_w=1.0, max_w=10.0):
    y = mlb.transform(df[label_col])
    positives = y.sum(axis=0).astype(np.float32)
    negatives = len(y) - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, min_w, max_w)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ============================================================
# 12. DECODING UTILITIES
# ============================================================
def apply_labelwise_thresholds(y_probs, thresholds):
    return (y_probs >= thresholds.reshape(1, -1)).astype(int)


def apply_topk_cap(y_pred, y_probs, top_k=None):
    if top_k is None:
        return y_pred

    capped = np.zeros_like(y_pred)
    top_indices = np.argsort(-y_probs, axis=1)[:, :top_k]

    for i in range(y_probs.shape[0]):
        capped[i, top_indices[i]] = 1

    return y_pred * capped


def apply_hard_max_predictions(y_pred, y_probs, hard_cap=None):
    if hard_cap is None:
        return y_pred

    capped = np.zeros_like(y_pred)
    top_indices = np.argsort(-y_probs, axis=1)[:, :hard_cap]

    for i in range(y_probs.shape[0]):
        active = np.where(y_pred[i] == 1)[0]
        active_set = set(active.tolist())
        keep = [idx for idx in top_indices[i] if idx in active_set]
        capped[i, keep] = 1

    return capped


def precision_at_k(y_true, y_probs, k=5):
    k = min(k, y_probs.shape[1])
    topk = np.argsort(-y_probs, axis=1)[:, :k]
    scores = []

    for i in range(y_true.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0].tolist())
        pred_set = set(topk[i].tolist())
        scores.append(len(true_set & pred_set) / k)

    return float(np.mean(scores))


def compute_metrics_from_preds(y_true, y_pred, y_probs):
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_at_5": precision_at_k(y_true, y_probs, 5),
        "precision_at_8": precision_at_k(y_true, y_probs, 8),
    }


def tune_labelwise_thresholds(y_true, y_probs, threshold_grid, beta=0.75, default_threshold=0.50, absent_threshold=0.65):
    num_labels = y_true.shape[1]
    best_thresholds = np.full(num_labels, default_threshold, dtype=np.float32)

    for j in range(num_labels):
        y_true_j = y_true[:, j]
        y_prob_j = y_probs[:, j]

        if y_true_j.sum() == 0:
            best_thresholds[j] = absent_threshold
            continue

        best_score = -1.0
        best_th = default_threshold

        for th in threshold_grid:
            y_pred_j = (y_prob_j >= th).astype(int)
            p = precision_score(y_true_j, y_pred_j, zero_division=0)
            r = recall_score(y_true_j, y_pred_j, zero_division=0)

            if p + r == 0:
                score = 0.0
            else:
                score = (1 + beta * beta) * p * r / ((beta * beta * p) + r)

            if score > best_score:
                best_score = score
                best_th = th

        best_thresholds[j] = best_th

    return best_thresholds


def tune_topk_cap(y_true, y_probs, thresholds, topk_grid, hard_cap=None, beta=0.75):
    best_topk = topk_grid[0]
    best_score = -1.0

    base_pred = apply_labelwise_thresholds(y_probs, thresholds)

    for topk in topk_grid:
        pred = apply_topk_cap(base_pred, y_probs, topk)
        pred = apply_hard_max_predictions(pred, y_probs, hard_cap)

        p = precision_score(y_true, pred, average="micro", zero_division=0)
        r = recall_score(y_true, pred, average="micro", zero_division=0)

        if p + r == 0:
            score = 0.0
        else:
            score = (1 + beta * beta) * p * r / ((beta * beta * p) + r)

        if score > best_score:
            best_score = score
            best_topk = topk

    return best_topk, best_score


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    thresholds=None,
    topk=None,
    hard_cap=None
):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        event_mask = batch["event_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(input_ids, attention_mask, event_mask)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.50, dtype=np.float32)

    y_pred = apply_labelwise_thresholds(y_probs, thresholds)
    y_pred = apply_topk_cap(y_pred, y_probs, topk)
    y_pred = apply_hard_max_predictions(y_pred, y_probs, hard_cap)

    metrics = compute_metrics_from_preds(y_true, y_pred, y_probs)
    metrics["loss"] = total_loss / max(1, len(loader))

    return metrics, y_true, y_probs


# ============================================================
# 14. FREEZE / UNFREEZE ENCODER
# ============================================================
def set_encoder_trainable(model, is_trainable: bool):
    for param in model.encoder.parameters():
        param.requires_grad = is_trainable


# ============================================================
# 15. TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, pos_weight):
    device = CONFIG["device"]
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": CONFIG["encoder_lr"]},
            {"params": model.lstm.parameters(), "lr": CONFIG["lstm_lr"]},
            {"params": model.temporal_attention.parameters(), "lr": CONFIG["head_lr"]},
            {"params": model.classifier.parameters(), "lr": CONFIG["head_lr"]},
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

    use_amp = CONFIG["fp16"] and CONFIG["device"] == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_selection_score = -1.0
    best_thresholds = None
    best_topk = None
    best_state_dict = None
    history = []
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        if CONFIG["freeze_encoder_first_epoch"] and epoch == 0:
            set_encoder_trainable(model, False)
            print("\nEncoder frozen for epoch 1.")
        else:
            set_encoder_trainable(model, True)

        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(CONFIG["device"], non_blocking=True)
            attention_mask = batch["attention_mask"].to(CONFIG["device"], non_blocking=True)
            event_mask = batch["event_mask"].to(CONFIG["device"], non_blocking=True)
            labels = batch["labels"].to(CONFIG["device"], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask, event_mask)
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

        _, val_y_true, val_y_probs = evaluate(
            model,
            val_loader,
            CONFIG["device"],
            None,
            None,
            CONFIG["max_predictions"]
        )

        tuned_thresholds = tune_labelwise_thresholds(
            val_y_true,
            val_y_probs,
            CONFIG["threshold_grid"],
            beta=0.75,
            default_threshold=0.50,
            absent_threshold=0.65
        )

        tuned_topk, _ = tune_topk_cap(
            val_y_true,
            val_y_probs,
            tuned_thresholds,
            CONFIG["topk_grid"],
            CONFIG["max_predictions"],
            beta=0.75
        )

        val_pred = apply_labelwise_thresholds(val_y_probs, tuned_thresholds)
        val_pred = apply_topk_cap(val_pred, val_y_probs, tuned_topk)
        val_pred = apply_hard_max_predictions(val_pred, val_y_probs, CONFIG["max_predictions"])

        val_metrics = compute_metrics_from_preds(val_y_true, val_pred, val_y_probs)

        print("\n" + "=" * 100)
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss            : {train_loss:.4f}")
        print(f"Val Avg Threshold     : {float(np.mean(tuned_thresholds)):.4f}")
        print(f"Val Top-K Cap         : {tuned_topk}")
        print(f"Val Micro F1          : {val_metrics['micro_f1']:.4f}")
        print(f"Val Micro Precision   : {val_metrics['micro_precision']:.4f}")
        print(f"Val Micro Recall      : {val_metrics['micro_recall']:.4f}")
        print(f"Val Precision@5       : {val_metrics['precision_at_5']:.4f}")
        print("=" * 100)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_avg_threshold": float(np.mean(tuned_thresholds)),
            "val_topk": tuned_topk,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        selection_score = (
            0.50 * val_metrics["micro_f1"] +
            0.25 * val_metrics["micro_precision"] +
            0.25 * val_metrics["micro_recall"]
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_thresholds = tuned_thresholds.copy()
            best_topk = tuned_topk
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("Saved new best ICD model in memory.")
        else:
            patience_counter += 1
            print(f"No improvement. Early stop patience: {patience_counter}/{CONFIG['patience']}")

        if patience_counter >= CONFIG["patience"]:
            print("\nEarly stopping triggered.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished but no best model state was captured.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(CONFIG["output_dir"], "training_history.csv"), index=False)

    return best_state_dict, best_thresholds, best_topk, history_df


# ============================================================
# 16. MAIN PIPELINE
# ============================================================
def main():
    print("Loading datasets...")
    train_raw = load_split_from_folder(CONFIG["train_dir"], "train")
    val_raw = load_split_from_folder(CONFIG["val_dir"], "val")
    test_raw = load_split_from_folder(CONFIG["test_dir"], "test")

    train_df = preprocess_split(train_raw, "train")
    val_df = preprocess_split(val_raw, "val")
    test_df = preprocess_split(test_raw, "test")

    kept_labels = get_kept_labels_from_train(
        train_df,
        label_col="icd_labels",
        min_freq=CONFIG["min_label_freq"],
        top_k=CONFIG["top_k_labels"]
    )

    if len(kept_labels) == 0:
        raise ValueError("No ICD labels left after filtering.")

    kept_set = set(kept_labels)

    train_df["icd_labels"] = train_df["icd_labels"].apply(lambda labs: [x for x in labs if x in kept_set])
    val_df["icd_labels"] = val_df["icd_labels"].apply(lambda labs: [x for x in labs if x in kept_set])
    test_df["icd_labels"] = test_df["icd_labels"].apply(lambda labs: [x for x in labs if x in kept_set])

    train_df = train_df[train_df["icd_labels"].map(len) > 0].copy()
    val_df = val_df[val_df["icd_labels"].map(len) > 0].copy()
    test_df = test_df[test_df["icd_labels"].map(len) > 0].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("After preprocessing/filtering, one split became empty.")

    icd_mlb = MultiLabelBinarizer(classes=sorted(kept_labels))
    icd_mlb.fit(train_df["icd_labels"])

    print("\nAfter preprocessing and label filtering:")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape  : {val_df.shape}")
    print(f"Test shape : {test_df.shape}")
    print(f"Number of ICD labels: {len(icd_mlb.classes_)}")

    pos_weight = compute_pos_weights(
        train_df,
        "icd_labels",
        icd_mlb,
        CONFIG["min_pos_weight"],
        CONFIG["max_pos_weight"]
    )
    print("\nICD pos_weight shape:", pos_weight.shape)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["pretrained_model"])

    train_dataset = TemporalICDDataset(
        train_df,
        tokenizer,
        icd_mlb,
        max_event_length=CONFIG["max_event_length"],
        max_events=CONFIG["max_events"],
        append_discharge_as_last_event=CONFIG["append_discharge_as_last_event"]
    )

    val_dataset = TemporalICDDataset(
        val_df,
        tokenizer,
        icd_mlb,
        max_event_length=CONFIG["max_event_length"],
        max_events=CONFIG["max_events"],
        append_discharge_as_last_event=CONFIG["append_discharge_as_last_event"]
    )

    test_dataset = TemporalICDDataset(
        test_df,
        tokenizer,
        icd_mlb,
        max_event_length=CONFIG["max_event_length"],
        max_events=CONFIG["max_events"],
        append_discharge_as_last_event=CONFIG["append_discharge_as_last_event"]
    )

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

    model = TemporalICDClassifier(
        pretrained_model_name=CONFIG["pretrained_model"],
        num_labels=len(icd_mlb.classes_),
        hidden_dim=CONFIG["hidden_dim"],
        num_lstm_layers=CONFIG["num_lstm_layers"],
        task_hidden_dim=CONFIG["task_hidden_dim"],
        dropout=CONFIG["dropout"],
        bidirectional=CONFIG["bidirectional"]
    )

    print("\nTraining temporal ICD model...")
    best_state_dict, best_thresholds, best_topk, history_df = train_model(
        model,
        train_loader,
        val_loader,
        pos_weight
    )

    print("\nLoading best ICD model from memory...")
    model.load_state_dict(best_state_dict)
    model.to(CONFIG["device"])

    print(
        f"Evaluating on test set with avg threshold={float(np.mean(best_thresholds)):.4f} "
        f"and top-k={best_topk} ..."
    )

    test_metrics, y_true, y_probs = evaluate(
        model,
        test_loader,
        CONFIG["device"],
        best_thresholds,
        best_topk,
        CONFIG["max_predictions"]
    )

    print("\n" + "=" * 100)
    print("TEMPORAL ICD TEST RESULTS")
    print("=" * 100)
    for k, v in test_metrics.items():
        print(f"{k:24s}: {v:.4f}")
    print("=" * 100)

    def decode_predictions(y_probs, mlb, thresholds, top_k=None, hard_cap=None):
        y_pred = apply_labelwise_thresholds(y_probs, thresholds)
        y_pred = apply_topk_cap(y_pred, y_probs, top_k)
        y_pred = apply_hard_max_predictions(y_pred, y_probs, hard_cap)
        return mlb.inverse_transform(y_pred)

    predicted_icd_codes = decode_predictions(
        y_probs, icd_mlb, best_thresholds, best_topk, CONFIG["max_predictions"]
    )
    true_icd_codes = icd_mlb.inverse_transform(y_true)

    num_samples = min(5, len(predicted_icd_codes))
    random_indices = random.sample(range(len(predicted_icd_codes)), num_samples)

    print("\nRANDOM 5 TEST SAMPLES (ICD PREDICTIONS)\n")
    print("=" * 100)

    for i, idx in enumerate(random_indices, 1):
        print(f"Sample {i} (Index: {idx})")
        print("True ICD Codes     :", true_icd_codes[idx])
        print("Predicted ICD Codes:", predicted_icd_codes[idx])
        print("-" * 100)

    output_dir = CONFIG["output_dir"]

    full_config = copy.deepcopy(CONFIG)
    full_config["best_avg_threshold"] = float(np.mean(best_thresholds))
    full_config["best_topk"] = best_topk

    save_json(full_config, os.path.join(output_dir, "config.json"))
    save_json(icd_mlb.classes_.tolist(), os.path.join(output_dir, "label_classes.json"))
    save_json(best_thresholds.tolist(), os.path.join(output_dir, "best_labelwise_thresholds.json"))
    save_json({k: float(v) for k, v in test_metrics.items()}, os.path.join(output_dir, "test_metrics.json"))

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


# ============================================================
# 17. RUN SCRIPT
# ============================================================
if __name__ == "__main__":
    main()