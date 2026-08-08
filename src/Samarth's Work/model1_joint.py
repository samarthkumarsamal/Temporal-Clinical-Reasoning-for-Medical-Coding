# ============================================================
# JOINT ICD + CPT CODE PREDICTION USING BIOCLINICALBERT
# Folder-based split loading
#
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
#   ./model1_joint_icd_cpt_output
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
    "output_dir": "./model1_joint_icd_cpt_output",
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
    "icd_threshold": 0.3,
    "cpt_threshold": 0.3,
    "min_icd_label_freq": 2,
    "min_cpt_label_freq": 2,
    "top_k_icd_labels": 50,      # set None to keep all ICD labels
    "top_k_cpt_labels": 50,      # set None to keep all CPT labels
    "num_workers": 0,
    "fp16": True,
    "seed": 42,
    "icd_loss_weight": 1.0,
    "cpt_loss_weight": 1.0,
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


def extract_cpt_labels(label_obj):
    if not isinstance(label_obj, dict):
        return []
    return normalize_label_list(label_obj.get("cpt", []))


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
    df["icd_labels"] = df["labels"].apply(extract_icd_labels)
    df["cpt_labels"] = df["labels"].apply(extract_cpt_labels)

    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()

    df = df[
        (df["icd_labels"].map(len) > 0) | (df["cpt_labels"].map(len) > 0)
    ].copy()

    if len(df) == 0:
        raise ValueError(f"No usable rows left in {split_name} after preprocessing.")

    return df.reset_index(drop=True)


def build_datasets(train_raw: pd.DataFrame, val_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train_df = preprocess_split(train_raw, "train")
    val_df = preprocess_split(val_raw, "val")
    test_df = preprocess_split(test_raw, "test")

    kept_icd = get_kept_labels_from_train(
        train_df,
        label_col="icd_labels",
        min_freq=CONFIG["min_icd_label_freq"],
        top_k=CONFIG["top_k_icd_labels"]
    )
    kept_cpt = get_kept_labels_from_train(
        train_df,
        label_col="cpt_labels",
        min_freq=CONFIG["min_cpt_label_freq"],
        top_k=CONFIG["top_k_cpt_labels"]
    )

    if len(kept_icd) == 0:
        raise ValueError(
            "No ICD labels left after filtering train data. "
            "Try reducing min_icd_label_freq or setting top_k_icd_labels=None."
        )

    if len(kept_cpt) == 0:
        raise ValueError(
            "No CPT labels left after filtering train data. "
            "Try reducing min_cpt_label_freq or setting top_k_cpt_labels=None."
        )

    train_df = apply_label_filter(train_df, "icd_labels", kept_icd)
    val_df = apply_label_filter(val_df, "icd_labels", kept_icd)
    test_df = apply_label_filter(test_df, "icd_labels", kept_icd)

    train_df = apply_label_filter(train_df, "cpt_labels", kept_cpt)
    val_df = apply_label_filter(val_df, "cpt_labels", kept_cpt)
    test_df = apply_label_filter(test_df, "cpt_labels", kept_cpt)

    # Keep rows that still have at least one ICD or CPT label after filtering
    train_df = train_df[
        (train_df["icd_labels"].map(len) > 0) | (train_df["cpt_labels"].map(len) > 0)
    ].copy()
    val_df = val_df[
        (val_df["icd_labels"].map(len) > 0) | (val_df["cpt_labels"].map(len) > 0)
    ].copy()
    test_df = test_df[
        (test_df["icd_labels"].map(len) > 0) | (test_df["cpt_labels"].map(len) > 0)
    ].copy()

    if len(train_df) == 0:
        raise ValueError("No rows left in train split after label filtering.")
    if len(val_df) == 0:
        raise ValueError("No rows left in val split after label filtering.")
    if len(test_df) == 0:
        raise ValueError("No rows left in test split after label filtering.")

    icd_mlb = MultiLabelBinarizer(classes=sorted(kept_icd))
    cpt_mlb = MultiLabelBinarizer(classes=sorted(kept_cpt))

    train_icd_y = icd_mlb.fit_transform(train_df["icd_labels"])
    val_icd_y = icd_mlb.transform(val_df["icd_labels"])
    test_icd_y = icd_mlb.transform(test_df["icd_labels"])

    train_cpt_y = cpt_mlb.fit_transform(train_df["cpt_labels"])
    val_cpt_y = cpt_mlb.transform(val_df["cpt_labels"])
    test_cpt_y = cpt_mlb.transform(test_df["cpt_labels"])

    train_df["icd_vector"] = list(train_icd_y)
    val_df["icd_vector"] = list(val_icd_y)
    test_df["icd_vector"] = list(test_icd_y)

    train_df["cpt_vector"] = list(train_cpt_y)
    val_df["cpt_vector"] = list(val_cpt_y)
    test_df["cpt_vector"] = list(test_cpt_y)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        icd_mlb,
        cpt_mlb
    )


# ============================================================
# DATASET CLASS
# ============================================================
class JointDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts = df["text"].tolist()
        self.icd_labels = df["icd_vector"].tolist()
        self.cpt_labels = df["cpt_vector"].tolist()
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
            "icd_labels": torch.tensor(self.icd_labels[idx], dtype=torch.float),
            "cpt_labels": torch.tensor(self.cpt_labels[idx], dtype=torch.float),
        }


# ============================================================
# MODEL
# ============================================================
class JointClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_icd_labels: int, num_cpt_labels: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.icd_head = nn.Linear(hidden_size, num_icd_labels)
        self.cpt_head = nn.Linear(hidden_size, num_cpt_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_emb)

        icd_logits = self.icd_head(x)
        cpt_logits = self.cpt_head(x)

        return icd_logits, cpt_logits


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


def compute_metrics(y_true, y_probs, threshold=0.3, prefix=""):
    y_pred = (y_probs >= threshold).astype(int)

    return {
        f"{prefix}micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        f"{prefix}macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        f"{prefix}micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        f"{prefix}precision_at_5": precision_at_k(y_true, y_probs, 5),
        f"{prefix}precision_at_8": precision_at_k(y_true, y_probs, 8),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    icd_criterion = nn.BCEWithLogitsLoss()
    cpt_criterion = nn.BCEWithLogitsLoss()

    all_icd_probs = []
    all_icd_labels = []
    all_cpt_probs = []
    all_cpt_labels = []

    total_loss = 0.0
    total_icd_loss = 0.0
    total_cpt_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        icd_labels = batch["icd_labels"].to(device)
        cpt_labels = batch["cpt_labels"].to(device)

        icd_logits, cpt_logits = model(input_ids, attention_mask)

        icd_loss = icd_criterion(icd_logits, icd_labels)
        cpt_loss = cpt_criterion(cpt_logits, cpt_labels)
        loss = CONFIG["icd_loss_weight"] * icd_loss + CONFIG["cpt_loss_weight"] * cpt_loss

        icd_probs = torch.sigmoid(icd_logits)
        cpt_probs = torch.sigmoid(cpt_logits)

        total_loss += loss.item()
        total_icd_loss += icd_loss.item()
        total_cpt_loss += cpt_loss.item()

        all_icd_probs.append(icd_probs.detach().cpu().numpy())
        all_icd_labels.append(icd_labels.detach().cpu().numpy())
        all_cpt_probs.append(cpt_probs.detach().cpu().numpy())
        all_cpt_labels.append(cpt_labels.detach().cpu().numpy())

    y_icd_probs = np.vstack(all_icd_probs)
    y_icd_true = np.vstack(all_icd_labels)
    y_cpt_probs = np.vstack(all_cpt_probs)
    y_cpt_true = np.vstack(all_cpt_labels)

    metrics = {}
    metrics.update(compute_metrics(y_icd_true, y_icd_probs, CONFIG["icd_threshold"], prefix="icd_"))
    metrics.update(compute_metrics(y_cpt_true, y_cpt_probs, CONFIG["cpt_threshold"], prefix="cpt_"))
    metrics["loss"] = total_loss / max(1, len(loader))
    metrics["icd_loss"] = total_icd_loss / max(1, len(loader))
    metrics["cpt_loss"] = total_cpt_loss / max(1, len(loader))
    metrics["joint_score"] = (metrics["icd_micro_f1"] + metrics["cpt_micro_f1"]) / 2.0

    return metrics, y_icd_true, y_icd_probs, y_cpt_true, y_cpt_probs


# ============================================================
# TRAIN
# ============================================================
def train_model(model, train_loader, val_loader):
    device = CONFIG["device"]
    model = model.to(device)

    icd_criterion = nn.BCEWithLogitsLoss()
    cpt_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": CONFIG["encoder_lr"]},
            {
                "params": list(model.icd_head.parameters()) + list(model.cpt_head.parameters()),
                "lr": CONFIG["classifier_lr"]
            },
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

    best_joint_score = -1.0
    best_state_dict = None
    history = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            icd_labels = batch["icd_labels"].to(device)
            cpt_labels = batch["cpt_labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                icd_logits, cpt_logits = model(input_ids, attention_mask)

                icd_loss = icd_criterion(icd_logits, icd_labels)
                cpt_loss = cpt_criterion(cpt_logits, cpt_labels)

                loss = (
                    CONFIG["icd_loss_weight"] * icd_loss +
                    CONFIG["cpt_loss_weight"] * cpt_loss
                )

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
        val_metrics, _, _, _, _ = evaluate(model, val_loader, device)

        print("\n" + "=" * 80)
        print(f"Epoch {epoch+1}")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Val Total Loss  : {val_metrics['loss']:.4f}")
        print(f"Val ICD Loss    : {val_metrics['icd_loss']:.4f}")
        print(f"Val CPT Loss    : {val_metrics['cpt_loss']:.4f}")
        print(f"Val ICD MicroF1 : {val_metrics['icd_micro_f1']:.4f}")
        print(f"Val CPT MicroF1 : {val_metrics['cpt_micro_f1']:.4f}")
        print(f"Val Joint Score : {val_metrics['joint_score']:.4f}")
        print("=" * 80)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if val_metrics["joint_score"] > best_joint_score:
            best_joint_score = val_metrics["joint_score"]
            best_state_dict = copy.deepcopy(model.state_dict())
            print("Saved new best joint model in memory.")

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

    train_df, val_df, test_df, icd_mlb, cpt_mlb = build_datasets(train_raw, val_raw, test_raw)

    print("\nAfter preprocessing and label filtering:")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape  : {val_df.shape}")
    print(f"Test shape : {test_df.shape}")
    print(f"Number of ICD labels: {len(icd_mlb.classes_)}")
    print(f"Number of CPT labels: {len(cpt_mlb.classes_)}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["pretrained_model"])

    train_dataset = JointDataset(train_df, tokenizer, CONFIG["max_length"])
    val_dataset = JointDataset(val_df, tokenizer, CONFIG["max_length"])
    test_dataset = JointDataset(test_df, tokenizer, CONFIG["max_length"])

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

    model = JointClassifier(
        pretrained_model_name=CONFIG["pretrained_model"],
        num_icd_labels=len(icd_mlb.classes_),
        num_cpt_labels=len(cpt_mlb.classes_),
        dropout=CONFIG["dropout"]
    )

    print("\nTraining joint ICD + CPT model...")
    best_state_dict, history_df = train_model(model, train_loader, val_loader)

    print("\nLoading best joint model from memory...")
    model.load_state_dict(best_state_dict)
    model.to(CONFIG["device"])

    print("Evaluating joint model on test set...")
    test_metrics, y_icd_true, y_icd_probs, y_cpt_true, y_cpt_probs = evaluate(
        model,
        test_loader,
        CONFIG["device"]
    )

    print("\n" + "=" * 80)
    print("JOINT ICD + CPT TEST RESULTS")
    print("=" * 80)
    for k, v in test_metrics.items():
        print(f"{k:22s}: {v:.4f}")
    print("=" * 80)

    output_dir = CONFIG["output_dir"]

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    with open(os.path.join(output_dir, "icd_label_classes.json"), "w", encoding="utf-8") as f:
        json.dump(icd_mlb.classes_.tolist(), f, indent=2)

    with open(os.path.join(output_dir, "cpt_label_classes.json"), "w", encoding="utf-8") as f:
        json.dump(cpt_mlb.classes_.tolist(), f, indent=2)

    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    np.save(os.path.join(output_dir, "test_y_icd_true.npy"), y_icd_true)
    np.save(os.path.join(output_dir, "test_y_icd_probs.npy"), y_icd_probs)
    np.save(os.path.join(output_dir, "test_y_cpt_true.npy"), y_cpt_true)
    np.save(os.path.join(output_dir, "test_y_cpt_probs.npy"), y_cpt_probs)

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