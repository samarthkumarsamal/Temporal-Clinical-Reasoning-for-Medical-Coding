## 📊 Results

Our model was evaluated on the MIMIC-III dataset for joint ICD and CPT prediction using a temporal multi-note modeling framework.

### Experimental Results

| Model | ICD Micro-F1 | ICD Macro-F1 | CPT Micro-F1 | CPT Macro-F1 | P@5 (ICD) | P@5 (CPT) | Joint Score |
|------|-------------|-------------|-------------|-------------|-----------|-----------|-------------|
| Baseline (BioClinicalBERT) | 0.4506 | 0.3130 | 0.5380 | 0.1751 | 0.4315 | 0.4775 | 0.4593 |
| Longformer | 0.5946 | 0.4394 | 0.5904 | 0.3001 | 0.5563 | 0.4963 | 0.5925 |
| Temporal (BERT + LSTM) | 0.5343 | 0.4122 | 0.6096 | 0.3971 | 0.4979 | 0.5010 | 0.5720 |

These results demonstrate that temporal multi-note modeling improves performance while explicitly modeling patient progression across sequential clinical notes.

---

### 🔍 Main Findings

- Temporal multi-note modeling substantially improves performance over the baseline
- Temporal modeling enables more context-aware predictions across clinical notes
- Longformer achieves the strongest overall performance through long-context modeling
- ICD prediction remains more challenging than CPT prediction
- The temporal model provides an effective and clinically meaningful representation of patient progression over time
