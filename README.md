
# chinese-medical-empathy

**Description:**
Automatic detection of empathy and politeness in Chinese medical conversations, combining rule-based features and transformer-based models.

---

## 1. Overview

**Goal:**
Detect **empathy** and **politeness** expressions in **Chinese medical dialogues**, focusing on doctor-patient conversations.
We integrate **linguistic rules**, **lexical resources**, and **pre-trained transformer models** to achieve accurate and interpretable classification.

---

## 2. Background

Empathy and politeness are crucial in medical communication.
Prior research (e.g., Yeomans et al., *The R Journal*, 2018) introduced structured frameworks for **politeness detection** in English, combining feature engineering and machine learning. We adapt and extend this approach to Chinese medical contexts, emphasizing **multi-label detection** of empathy strategies and politeness markers.

---

## 3. Research Questions

1. How does **empathy** manifest linguistically in Chinese medical dialogues?
2. How do **empathy** and **politeness** interact in medical consultations?
3. What is the performance gap between **rule-based**, **traditional ML**, and **transformer-based** methods?
4. How does **contextual information** (previous turns) improve detection?

---

## 4. Data & Privacy

* Data format: doctor-patient conversation transcripts.
* Example: `Sample Data.xlsx` (an anonymized demo file) placed in `data/raw/` or `data/external/`.
* All data must be **anonymized** before use.
* Ethics: see `docs/ethics_privacy.md`.

---

## 5. Label Schema

**Empathy (multi-label):**

* Emotional acknowledgment
* Reassurance/comfort
* Encouragement/empowerment
* Shared responsibility
* Positive reframing
* Apology

**Politeness (multi-label / degree):**

* Greetings
* Gratitude
* Apology
* Hedges (e.g., "could you", "is it possible")
* Softened imperatives
* Negations

---

## 6. Methods

* **Rule-based & Lexicon Features** (`src/features/`, `src/rules/`)
* **Traditional ML Baselines** (LR, SVM + TF-IDF/n-grams)
* **Transformer Models** (BERT, RoBERTa, MacBERT for Chinese)
* **Multi-label Classification** with contextual input
* **Evaluation Metrics**: micro/macro F1, PR-AUC, dialogue-level consistency

---

## 7. Quick Start

```bash
# 1) Environment setup
pip install -r requirements.txt

# 2) Place data
# Example file: Sample Data.xlsx -> data/raw/

# 3) Preprocess
python -m src.data.preprocess --in data/raw --out data/processed --anonymize

# 4) Train model
python -m src.training.train --config configs/roberta_multilabel.yaml

# 5) Evaluate
python -m src.training.evaluate --ckpt outputs/ckpt.pt --data data/processed/dev.jsonl

# 6) Inference
python -m src.training.inference --ckpt outputs/ckpt.pt --in data/processed/test.jsonl --out outputs/preds.jsonl
```

---

## 8. Outputs

* Model weights, metrics, and visualizations in `outputs/`
* Model Card in `docs/model_card.md`

---

## 9. Ethics & Limitations

* Strict de-identification required.
* For **research & education** only — not for clinical decision-making.

---

## 10. Citation

If you use this repo, please cite:

```yaml
cff-version: 1.2.0
title: "chinese-medical-empathy"
message: "If you use this repository, please cite it as below."
authors:
  - family-names: YourSurname
    given-names: YourName
date-released: 2025-08-10
repository-code: "https://github.com/<your-org>/chinese-medical-empathy"
```

---

## 11. License

MIT License (code) + separate license for data (research-only).

---

我可以帮你把这一套直接打包成一个 **可上传到 GitHub 的初始化压缩包**，包括 README、目录树和空文件，这样你只需要解压 + git push 就行。
你要我帮你生成这个初始化包吗？这样你可以马上建仓库并运行基本框架。
