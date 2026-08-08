"""Model-agnostic loader for the pipeline's JSONL output.

Demonstrates how to consume **every** field in an admission record:

    hadm_id, subject_id, admittime, dischtime
    events[]                        chronological clinical events
        event_type                  ed_note | admission_note | nursing |
                                    radiology | ecg | pharmacy |
                                    respiratory | lab
        charttime, relative_time_hrs
        text                        (notes only)
        value {                     (labs only)
            label, valuenum, valueuom, is_abnormal
        }
    discharge_text                  full summary — NEVER use as model input
    discharge_narrative             diagnosis-scrubbed version — safe
    labels {
        icd10, cpt, icd9_proc
    }

The file is organized as:

    1. Streaming JSONL reader
    2. Vocabulary builders
         - ICD-10 / CPT / ICD-9-PCS label vocabs
         - Lab itemid -> index vocab
         - Event-type -> index vocab
    3. Lab statistics (mean/std per itemid) for z-score normalization
    4. ``MimicAdmissionDataset`` — a PyTorch ``Dataset`` that returns a rich
       dict exposing text, lab tensors, time vectors, event-type ids, and
       all three label heads.
    5. Adapter examples showing four common consumer patterns:
         a. ``adapter_text_concat``   — flatten notes into one long string
         b. ``adapter_text_per_note`` — one sequence per note (hierarchical)
         c. ``adapter_multimodal``    — text + lab matrix + time vector
         d. ``adapter_timeline_text`` — OPTIONAL NARRATIVIZER: renders labs
            as compact natural-language "lab panels" and interleaves them
            with notes in chronological order, producing one unified
            timeline document. See the adapter's docstring for when to
            use this vs the structured ``adapter_multimodal``.
    6. ``collate_admissions`` — pads ragged sequences, produces masks.
    7. ``main`` — runs the full pipeline and prints one batch.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# paths, you'll have to change this
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def split_paths(split: str) -> list[Path]:
    split_dir = PROCESSED_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} does not exist — run `make all` first."
        )
    return sorted(split_dir.glob("part-*.jsonl"))


# json reader
def iter_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    """Yield one admission at a time from one or more JSONL shards."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


#vocabular builder
def build_label_vocab(
    paths: list[Path],
    label_field: str,             # "icd10" | "cpt" | "icd9_proc"
    top_k: int | None = None,     # None = keep everything seen
) -> list[str]:
    """Top-K label vocabulary, built from whichever shards you pass in.

    Always build from the TRAIN shards only, never from val/test — that
    keeps the held-out splits honest.
    """
    counter: Counter[str] = Counter()
    for rec in iter_records(paths):
        counter.update(rec.get("labels", {}).get(label_field, []))
    items = counter.most_common(top_k) if top_k else counter.most_common()
    return [code for code, _ in items]


def build_lab_vocab(paths: list[Path], min_count: int = 100) -> list[int]:
    """Lab itemids seen in at least ``min_count`` events across TRAIN."""
    counter: Counter[int] = Counter()
    for rec in iter_records(paths):
        for ev in rec.get("events", []):
            if ev.get("event_type") == "lab":
                label = ev.get("value", {}).get("label")
                if label is not None:
                    counter[label] += 1
    return sorted(
        [label for label, c in counter.items() if c >= min_count]
    )


# Fixed, known in advance — keeping it here as a function so callers don't
# have to hand-maintain the list.
def event_type_vocab() -> list[str]:
    return [
        "lab",
        "nursing",
        "radiology",
        "pharmacy",
        "respiratory",
        "ecg",
        "admission_note",
        "ed_note",
    ]

# Lab statistic, should also be built from TRAIN only to avoid leakage. Returns a dict mapping
def compute_lab_stats(
    paths: list[Path],
    lab_labels: list[str],
) -> dict[str, tuple[float, float]]:
    """Welford-style running mean/std per lab label, from TRAIN only.

    Returns {label: (mean, std)}. Labs with zero variance get std=1.0 so
    z-scoring becomes a no-op for them.
    """
    n: dict[str, int] = defaultdict(int)
    mean: dict[str, float] = defaultdict(float)
    m2: dict[str, float] = defaultdict(float)
    wanted = set(lab_labels)

    for rec in iter_records(paths):
        for ev in rec.get("events", []):
            if ev.get("event_type") != "lab":
                continue
            v = ev.get("value", {})
            label = v.get("label")
            val = v.get("valuenum")
            if label not in wanted or val is None:
                continue
            n[label] += 1
            delta = val - mean[label]
            mean[label] += delta / n[label]
            delta2 = val - mean[label]
            m2[label] += delta * delta2

    stats: dict[str, tuple[float, float]] = {}
    for label in lab_labels:
        if n[label] > 1:
            var = m2[label] / (n[label] - 1)
            std = math.sqrt(var) if var > 0 else 1.0
            stats[label] = (mean[label], std)
        else:
            stats[label] = (0.0, 1.0)
    return stats


#Dataset
class MimicAdmissionDataset(Dataset):
    """One item = one admission, exposed as a model-agnostic dict.

    __getitem__ returns::

        {
            "hadm_id":        int,
            "subject_id":     int,
            "admittime":      str (ISO),
            "dischtime":      str (ISO),

            # Chronologically ordered note events, text only:
            "note_texts":     list[str]                # len = n_notes
            "note_types":     LongTensor (n_notes,)    # event-type vocab ids
            "note_times":     FloatTensor (n_notes,)   # relative hours
            "note_types_str": list[str]                # raw names (debug)

            # Chronologically ordered lab events, numerical:
            "lab_values":     FloatTensor (n_labs,)    # z-scored valuenum
            "lab_item_ids":   LongTensor (n_labs,)     # lab-vocab ids
            "lab_flags":      FloatTensor (n_labs,)    # is_abnormal (0/1)
            "lab_times":      FloatTensor (n_labs,)    # relative hours
            "lab_has_value":  FloatTensor (n_labs,)    # 1 if valuenum!=null

            # Optional auxiliary context (diagnosis-scrubbed):
            "discharge_narrative": str

            # Multi-label targets:
            "icd10_labels":     FloatTensor (|icd10_vocab|,)
            "cpt_labels":       FloatTensor (|cpt_vocab|,)
            "icd9_proc_labels": FloatTensor (|proc_vocab|,)
        }

    Downstream adapters decide which of these fields they care about. The
    ``collate_admissions`` function below can pad any subset.
    """

    def __init__(
        self,
        jsonl_paths: list[Path],
        icd10_vocab: list[str],
        cpt_vocab: list[str],
        proc_vocab: list[str],
        lab_vocab: list[str],
        lab_stats: dict[str, tuple[float, float]],
        event_type_list: list[str] | None = None,
        in_memory: bool = True,
    ) -> None:
        self.icd10_idx = {c: i for i, c in enumerate(icd10_vocab)}
        self.cpt_idx = {c: i for i, c in enumerate(cpt_vocab)}
        self.proc_idx = {c: i for i, c in enumerate(proc_vocab)}
        self.lab_idx = {lbl: i for i, lbl in enumerate(lab_vocab)}
        self.lab_stats = lab_stats

        event_type_list = event_type_list or event_type_vocab()
        self.event_type_idx = {t: i for i, t in enumerate(event_type_list)}
        self.n_icd10 = len(icd10_vocab)
        self.n_cpt = len(cpt_vocab)
        self.n_proc = len(proc_vocab)

        self.paths = jsonl_paths
        if in_memory:
            self.records: list[dict[str, Any]] = list(iter_records(jsonl_paths))
        else:
            # For very large datasets, build an index (path, byte_offset)
            # and seek on __getitem__. Not shown here — swap to
            # IterableDataset if memory is an issue.
            raise NotImplementedError("Set in_memory=True for this example.")

    def __len__(self) -> int:
        return len(self.records)

    # label encoding helpers

    def _multihot(self, codes: list[str], idx: dict[str, int], size: int) -> torch.Tensor:
        y = torch.zeros(size, dtype=torch.float32)
        for c in codes:
            i = idx.get(c)
            if i is not None:
                y[i] = 1.0
        return y

    #  event splitting

    def _split_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        note_texts: list[str] = []
        note_types_str: list[str] = []
        note_types: list[int] = []
        note_times: list[float] = []

        lab_values: list[float] = []
        lab_item_ids: list[int] = []
        lab_flags: list[float] = []
        lab_times: list[float] = []
        lab_has_value: list[float] = []

        for ev in events:
            et = ev.get("event_type")
            t = ev.get("relative_time_hrs")
            t = float(t) if t is not None else 0.0

            if et == "lab":
                val = ev.get("value", {}) or {}
                label = val.get("label")
                idx = self.lab_idx.get(label)
                if idx is None:
                    continue  # lab itemid not in vocab — skip
                raw = val.get("valuenum")
                mean, std = self.lab_stats.get(label, (0.0, 1.0))
                if raw is None:
                    z = 0.0
                    has_val = 0.0
                else:
                    z = (raw - mean) / (std if std > 0 else 1.0)
                    has_val = 1.0
                lab_values.append(z)
                lab_item_ids.append(idx)
                lab_flags.append(1.0 if val.get("is_abnormal") else 0.0)
                lab_times.append(t)
                lab_has_value.append(has_val)
            else:
                text = ev.get("text")
                if not text:
                    continue
                note_texts.append(text)
                note_types_str.append(et or "unknown")
                note_types.append(self.event_type_idx.get(et, 0))
                note_times.append(t)

        return {
            "note_texts": note_texts,
            "note_types_str": note_types_str,
            "note_types": torch.tensor(note_types, dtype=torch.long),
            "note_times": torch.tensor(note_times, dtype=torch.float32),
            "lab_values": torch.tensor(lab_values, dtype=torch.float32),
            "lab_item_ids": torch.tensor(lab_item_ids, dtype=torch.long),
            "lab_flags": torch.tensor(lab_flags, dtype=torch.float32),
            "lab_times": torch.tensor(lab_times, dtype=torch.float32),
            "lab_has_value": torch.tensor(lab_has_value, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        events = rec.get("events", [])
        event_features = self._split_events(events)

        labels = rec.get("labels", {})
        item = {
            "hadm_id": int(rec["hadm_id"]),
            "subject_id": int(rec["subject_id"]),
            "admittime": rec.get("admittime"),
            "dischtime": rec.get("dischtime"),
            "discharge_narrative": rec.get("discharge_narrative", "") or "",
            # Preserve the raw chronologically ordered event list so that
            # adapters which need the original interleaving (e.g. the
            # narrativizer below) can walk it directly instead of having to
            # re-merge the split note / lab tensors.
            "raw_events": events,
            "icd10_labels": self._multihot(
                labels.get("icd10", []), self.icd10_idx, self.n_icd10
            ),
            "cpt_labels": self._multihot(
                labels.get("cpt", []), self.cpt_idx, self.n_cpt
            ),
            "icd9_proc_labels": self._multihot(
                labels.get("icd9_proc", []), self.proc_idx, self.n_proc
            ),
        }
        item.update(event_features)
        return item


# Adapters
def adapter_text_concat(
    item: dict[str, Any],
    sep: str = " [SEP] ",
    prepend_type: bool = True,
) -> str:
    """Flatten every note into one long string.

    Suitable for long-context models (Longformer, ClinicalBigBird, or any
    decoder with large max_len). Prepends the event type as a lightweight
    structural hint: '[nursing] ...body... [SEP] [radiology] ...'.
    """
    parts: list[str] = []
    for t, body in zip(item["note_types_str"], item["note_texts"]):
        parts.append(f"[{t}] {body}" if prepend_type else body)
    return sep.join(parts)


def adapter_text_per_note(item: dict[str, Any]) -> list[str]:
    """Return notes as a list of strings.

    Suitable for hierarchical encoders (ClinicalBERT + aggregator). The
    consumer tokenizes each note independently and pools per-note [CLS]
    vectors into an admission representation.
    """
    return list(item["note_texts"])


def adapter_multimodal(item: dict[str, Any]) -> dict[str, Any]:
    """Return a dict ready for a combined text + structured-lab model.

    The returned tensors are per-admission (not yet padded); use
    ``collate_admissions`` to batch them.
    """
    return {
        "note_texts": item["note_texts"],
        "note_types": item["note_types"],
        "note_times": item["note_times"],
        "lab_values": item["lab_values"],
        "lab_item_ids": item["lab_item_ids"],
        "lab_flags": item["lab_flags"],
        "lab_times": item["lab_times"],
        "lab_has_value": item["lab_has_value"],
        "discharge_narrative": item["discharge_narrative"],
        "labels": {
            "icd10": item["icd10_labels"],
            "cpt": item["cpt_labels"],
            "icd9_proc": item["icd9_proc_labels"],
        },
    }


# ----------------------------------------------------------------------------
# OPTIONAL: text narrativizer
# ----------------------------------------------------------------------------
#
# The adapters below are a DEMONSTRATION of the "everything as narrative"
# approach: lab results are rendered as compact natural-language lab panels
# and interleaved with clinical notes in chronological order, producing one
# unified timeline document that any text encoder can consume directly.
#
# When to use this:
#   - You want a single-modality text pipeline (one tokenizer, one encoder,
#     one loss) instead of a multimodal text + structured-lab model.
#   - You are using a clinical pretrained encoder (ClinicalBERT, BioBERT,
#     BioMedLM, clinical Llama, etc.) that was pretrained on text like
#     "creatinine elevated to 2.1 mg/dL" and already has good
#     representations for lab language.
#   - You want attention to operate across labs and surrounding notes
#     without building a multimodal fusion module.
#
# When NOT to use this:
#   - You want to run a multimodal ablation against a structured-lab baseline
#     (use ``adapter_multimodal`` for that; the structured tensors are still
#     available on every item regardless of which adapter you use).
#   - You are tight on context budget and have very lab-heavy admissions.
#     Even the compact panel format below uses ~2-5 tokens per lab, which
#     can add up to a few thousand tokens per admission.
#
# Design notes:
#   - Labs drawn at the same ``charttime`` are grouped into a single "panel"
#     (one blood draw -> one panel) to save tokens and mirror chart format.
#   - Abnormal values are marked inline with ``(abn)``. MIMIC-III's FLAG
#     column does not distinguish high from low, so we cannot emit ``(H)`` /
#     ``(L)`` directionally without re-computing reference ranges.
#   - Units are omitted by default for compactness; the lab name is usually
#     unambiguous enough. Flip ``include_uom=True`` if you prefer them.
#   - Null numeric values are skipped (no signal without a number).
# ----------------------------------------------------------------------------


def _fmt_rel_hours(h: float | None) -> str:
    """Render a relative_time_hrs value like '+03:12' or '-00:45'."""
    if h is None:
        return "??"
    sign = "-" if h < 0 else "+"
    h_abs = abs(h)
    hours = int(h_abs)
    minutes = int(round((h_abs - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{sign}{hours:02d}:{minutes:02d}"


def _fmt_lab_value(valuenum: float | None) -> str:
    """Compact numeric rendering: strip trailing zeros, keep small precision."""
    if valuenum is None:
        return "?"
    if abs(valuenum) >= 100:
        return f"{valuenum:.0f}"
    if abs(valuenum) >= 10:
        return f"{valuenum:.1f}".rstrip("0").rstrip(".")
    return f"{valuenum:.2f}".rstrip("0").rstrip(".")


def render_lab_panel(
    labs_at_same_time: list[dict[str, Any]],
    include_uom: bool = False,
) -> str:
    """Render a group of labs sharing one ``charttime`` as a single panel.

    Parameters
    ----------
    labs_at_same_time:
        List of raw lab event dicts from ``record["events"]`` (i.e.
        ``event_type == "lab"``) that all share the same ``charttime``.
    include_uom:
        If True, append ``valueuom`` after each value. Costs 1-2 tokens
        per lab.

    Returns
    -------
    A single string like::

        [lab panel +03:12] Hemoglobin 12.3; Sodium 128 (abn); Creatinine 2.1 (abn)

    Labs with a null ``valuenum`` are omitted. If all labs in the group
    have null values, returns an empty string.
    """
    if not labs_at_same_time:
        return ""

    rel = labs_at_same_time[0].get("relative_time_hrs")
    header = f"[lab panel {_fmt_rel_hours(rel)}]"

    parts: list[str] = []
    for ev in labs_at_same_time:
        v = ev.get("value") or {}
        label = v.get("label")
        val = v.get("valuenum")
        if label is None or val is None:
            continue
        piece = f"{label} {_fmt_lab_value(val)}"
        if include_uom and v.get("valueuom"):
            piece += f" {v['valueuom']}"
        if v.get("is_abnormal"):
            piece += " (abn)"
        parts.append(piece)

    if not parts:
        return ""
    return f"{header} " + "; ".join(parts)


def _render_note(ev: dict[str, Any]) -> str:
    """Render a single note event with a small header carrying type + time."""
    t = ev.get("event_type", "note")
    rel = ev.get("relative_time_hrs")
    body = ev.get("text") or ""
    return f"[{t} {_fmt_rel_hours(rel)}]\n{body}"


def adapter_timeline_text(
    item: dict[str, Any],
    include_uom: bool = False,
    include_discharge_narrative: bool = False,
    sep: str = "\n\n",
) -> str:
    """Render an admission as a single chronological text document.

    Walks ``item["raw_events"]`` in chronological order, rendering each
    note directly and grouping consecutive labs that share a charttime
    into one ``render_lab_panel`` call. The resulting document is ready
    to be tokenized by any text encoder.

    Parameters
    ----------
    item:
        An admission dict from ``MimicAdmissionDataset.__getitem__``.
    include_uom:
        Forwarded to ``render_lab_panel``; set True to include units.
    include_discharge_narrative:
        If True, append the diagnosis-scrubbed discharge narrative at the
        very end of the document (clearly delimited). This is the
        auxiliary-context pathway. Off by default — only turn it on if
        you have audited the scrub output for leakage and your task
        tolerates the signal from a hindsight document.
    sep:
        String between adjacent events in the output. ``"\\n\\n"`` keeps
        the document readable; ``" [SEP] "`` is more token-efficient.

    Returns
    -------
    One large string representing the entire admission timeline.
    """
    events = item.get("raw_events") or []
    pieces: list[str] = []

    i = 0
    while i < len(events):
        ev = events[i]
        if ev.get("event_type") == "lab":
            # Greedily consume the full run of labs sharing this charttime.
            t0 = ev.get("charttime")
            group: list[dict[str, Any]] = []
            while i < len(events) and events[i].get("event_type") == "lab" \
                    and events[i].get("charttime") == t0:
                group.append(events[i])
                i += 1
            rendered = render_lab_panel(group, include_uom=include_uom)
            if rendered:
                pieces.append(rendered)
        else:
            pieces.append(_render_note(ev))
            i += 1

    timeline = sep.join(pieces)

    if include_discharge_narrative:
        narr = item.get("discharge_narrative") or ""
        if narr:
            timeline = (
                f"{timeline}{sep}[discharge narrative (scrubbed, auxiliary)]\n{narr}"
            )

    return timeline


# ============================================================================
# 6. Collate — pads ragged sequences and builds masks
# ============================================================================

def _pad_1d(seqs: list[torch.Tensor], pad_value: float | int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) > 0 and lengths.max() > 0 else 1
    dtype = seqs[0].dtype if seqs else torch.float32
    out = torch.full((len(seqs), max_len), pad_value, dtype=dtype)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = s.shape[0]
        if n > 0:
            out[i, :n] = s
            mask[i, :n] = 1
    return out, mask


def collate_admissions(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad all ragged per-admission tensors to the batch max length."""
    B = len(batch)

    note_types, note_mask = _pad_1d([b["note_types"] for b in batch])
    note_times, _ = _pad_1d([b["note_times"] for b in batch])

    lab_values, lab_mask = _pad_1d([b["lab_values"] for b in batch])
    lab_item_ids, _ = _pad_1d([b["lab_item_ids"] for b in batch])
    lab_flags, _ = _pad_1d([b["lab_flags"] for b in batch])
    lab_times, _ = _pad_1d([b["lab_times"] for b in batch])
    lab_has_value, _ = _pad_1d([b["lab_has_value"] for b in batch])

    return {
        # Identity
        "hadm_ids": [b["hadm_id"] for b in batch],
        "subject_ids": [b["subject_id"] for b in batch],
        "admittimes": [b["admittime"] for b in batch],
        "dischtimes": [b["dischtime"] for b in batch],

        # Notes — text stays as a ragged list[list[str]] for the adapter
        # of choice to tokenize.
        "note_texts": [b["note_texts"] for b in batch],
        "note_types": note_types,          # (B, N_max)
        "note_times": note_times,          # (B, N_max)
        "note_mask": note_mask,            # (B, N_max)

        # Labs — fully tensorized
        "lab_values": lab_values,          # (B, L_max)
        "lab_item_ids": lab_item_ids,      # (B, L_max)
        "lab_flags": lab_flags,            # (B, L_max)
        "lab_times": lab_times,            # (B, L_max)
        "lab_has_value": lab_has_value,    # (B, L_max)
        "lab_mask": lab_mask,              # (B, L_max)

        # Optional auxiliary context
        "discharge_narrative": [b["discharge_narrative"] for b in batch],

        # Labels (always dense — vocab size is fixed)
        "icd10_labels": torch.stack([b["icd10_labels"] for b in batch]),
        "cpt_labels": torch.stack([b["cpt_labels"] for b in batch]),
        "icd9_proc_labels": torch.stack([b["icd9_proc_labels"] for b in batch]),
    }


# ============================================================================
# 7. Demo
# ============================================================================

def _describe(batch: dict[str, Any]) -> None:
    def shape(x):
        return tuple(x.shape) if isinstance(x, torch.Tensor) else f"len={len(x)}"

    print("\n--- one collated batch ---")
    for k in (
        "hadm_ids",
        "note_texts",
        "note_types",
        "note_times",
        "note_mask",
        "lab_values",
        "lab_item_ids",
        "lab_flags",
        "lab_times",
        "lab_has_value",
        "lab_mask",
        "discharge_narrative",
        "icd10_labels",
        "cpt_labels",
        "icd9_proc_labels",
    ):
        v = batch[k]
        extra = ""
        if isinstance(v, torch.Tensor):
            extra = f"  dtype={v.dtype}"
            if v.dtype in (torch.float32, torch.float64):
                extra += f"  sum={v.sum().item():.1f}"
        print(f"  {k:<22} {shape(v)}{extra}")


def main() -> None:
    train_paths = split_paths("train")
    val_paths = split_paths("val")
    print(f"train shards: {len(train_paths)}, val shards: {len(val_paths)}")

    # --- Build vocabularies from the TRAIN split only ------------------
    print("\nbuilding vocabularies from train...")
    icd10_vocab = build_label_vocab(train_paths, "icd10", top_k=200)
    cpt_vocab = build_label_vocab(train_paths, "cpt", top_k=200)
    proc_vocab = build_label_vocab(train_paths, "icd9_proc", top_k=200)
    lab_vocab = build_lab_vocab(train_paths, min_count=100)
    print(f"  icd10: {len(icd10_vocab)}, cpt: {len(cpt_vocab)}, "
          f"icd9_proc: {len(proc_vocab)}, labs: {len(lab_vocab)}")

    print("computing lab stats...")
    lab_stats = compute_lab_stats(train_paths, lab_vocab)

    # --- Datasets ------------------------------------------------------
    train_ds = MimicAdmissionDataset(
        train_paths,
        icd10_vocab=icd10_vocab,
        cpt_vocab=cpt_vocab,
        proc_vocab=proc_vocab,
        lab_vocab=lab_vocab,
        lab_stats=lab_stats,
    )
    val_ds = MimicAdmissionDataset(
        val_paths,
        icd10_vocab=icd10_vocab,
        cpt_vocab=cpt_vocab,
        proc_vocab=proc_vocab,
        lab_vocab=lab_vocab,
        lab_stats=lab_stats,
    )
    print(f"train admissions: {len(train_ds):,}, val admissions: {len(val_ds):,}")

    # --- Inspect a single example ------------------------------------
    example = train_ds[0]
    print("\n--- single example (train[0]) ---")
    print(f"  hadm_id={example['hadm_id']}  subject={example['subject_id']}")
    print(f"  n_notes={len(example['note_texts'])}  n_labs={len(example['lab_values'])}")
    print(f"  first note type: "
          f"{example['note_types_str'][0] if example['note_texts'] else 'n/a'}")
    print(f"  icd10 positives: {int(example['icd10_labels'].sum().item())}")
    print(f"  discharge_narrative[:100]: "
          f"{example['discharge_narrative'][:100]!r}")

    # --- Adapter demos ------------------------------------------------
    print("\n--- adapter_text_concat (first 200 chars) ---")
    concat = adapter_text_concat(example)
    print(f"  total length = {len(concat)} chars")
    print(f"  {concat[:200]!r}")

    print("\n--- adapter_text_per_note ---")
    per_note = adapter_text_per_note(example)
    print(f"  {len(per_note)} notes, first note length = "
          f"{len(per_note[0]) if per_note else 0} chars")

    print("\n--- adapter_multimodal ---")
    mm = adapter_multimodal(example)
    print(f"  lab_values shape   = {tuple(mm['lab_values'].shape)}")
    print(f"  lab_item_ids shape = {tuple(mm['lab_item_ids'].shape)}")
    print(f"  note count         = {len(mm['note_texts'])}")

    print("\n--- adapter_timeline_text (optional narrativizer) ---")
    timeline = adapter_timeline_text(example)
    print(f"  total length = {len(timeline)} chars")
    # Show the first lab panel in the timeline to illustrate the format.
    for line in timeline.splitlines():
        if line.startswith("[lab panel"):
            print(f"  first lab panel: {line[:200]}")
            break
    print(f"  head:\n{timeline[:400]}")
    print("  ...")
    print(f"  tail:\n{timeline[-400:]}")

    # --- Batched via DataLoader --------------------------------------
    loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_admissions,
    )
    batch = next(iter(loader))
    _describe(batch)

    # --- How downstream models consume the batch --------------------
    # A few recipes:
    #
    # (a) Text-only, long-context model (e.g. Longformer):
    #     texts = [adapter_text_concat({'note_types_str': ..., 'note_texts': n})
    #              for n in batch['note_texts']]
    #     enc = tokenizer(texts, padding=True, truncation=True,
    #                     max_length=4096, return_tensors='pt')
    #     logits = model(**enc).logits
    #
    # (b) Hierarchical text encoder (ClinicalBERT + aggregator):
    #     for i, notes in enumerate(batch['note_texts']):
    #         enc = tokenizer(notes, padding='max_length', truncation=True,
    #                         max_length=512, return_tensors='pt')
    #         cls = bert(**enc).last_hidden_state[:, 0]         # (n_notes, H)
    #         # pool per-admission with batch['note_mask'][i] ...
    #
    # (c) Multimodal: text encoder + lab transformer + time features:
    #     lab_emb = lab_item_embedding(batch['lab_item_ids'])   # (B, L, D)
    #     lab_emb = lab_emb + value_proj(batch['lab_values'].unsqueeze(-1))
    #     lab_emb = lab_emb + time_embed(batch['lab_times'])
    #     lab_vec = transformer(lab_emb, mask=batch['lab_mask'])
    #     # concat with pooled text vec, then multi-head classification
    #
    # (d) Multi-task: all three label heads share an encoder:
    #     loss = (
    #         bce(icd10_logits,     batch['icd10_labels'])
    #         + bce(cpt_logits,     batch['cpt_labels'])
    #         + bce(proc_logits,    batch['icd9_proc_labels'])
    #     )
    #
    # (e) Unified timeline (OPTIONAL narrativizer path):
    #     texts = [adapter_timeline_text(ex) for ex in batch_of_raw_items]
    #     enc = tokenizer(texts, padding=True, truncation=True,
    #                     max_length=4096, return_tensors='pt')
    #     logits = model(**enc).logits      # one encoder, one loss, done.


if __name__ == "__main__":
    main()
