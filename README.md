# Genomic Sequence Classification

A deep learning framework for genomic sequence classification using transformer-based DNA language models (DNABERT).  
The project supports multiple model architectures (Transformer, CNN, DNN) and compares different fine-tuning strategies for regulatory element prediction tasks.

---

## Overview

This repository implements machine learning models for classifying genomic sequences into functional regulatory elements.  
The current implementation focuses on enhancer vs non-enhancer classification using a DNABERT-based transformer model, with support for:

- Full fine-tuning
- Partial fine-tuning (last N layers)
- Frozen backbone + trainable classifier head

The framework is designed to be extensible to additional genomic tasks such as promoter prediction and multi-class regulatory element classification.

---

## Current Task

### Enhancer Classification
Binary classification of DNA sequences:

- **Enhancer**
- **Non-enhancer**

Input sequences are encoded using **k-mer tokenization (k=6)** and processed by a pretrained DNA language model.

---

## Model Architecture

### Transformer Model (DNABERT)

- Base model: `zhihan1996/DNA_bert_6`
- Input: k-mer tokenized DNA sequences
- Encoder: Transformer (BERT-style architecture)
- Classifier head:
  - Linear → ReLU → Dropout → Linear (binary output)

---

## Training Strategies

The project supports three fine-tuning strategies:

| Strategy | Description |
|----------|------------|
| Frozen backbone | Only classifier head trained |
| Partial fine-tuning | Last N transformer layers trainable |
| Full fine-tuning | Entire model trainable |

---

## Results Summary

| Tuning Strategy | Trainable Params | Test Accuracy | ROC-AUC | PR-AUC | Training Time |
|----------------|------------------|--------------|---------|--------|---------------|
| Frozen | ~0.6M | 0.56 | 0.57 | 0.57 | ~2 min |
| Partial (last 2 layers) | ~14.8M | 0.87 | 0.94 | 0.94 | ~3–5 min |
| Full fine-tuning | ~89M | TBD | TBD | TBD | overnight |

---

## Key Features

- DNABERT-based genomic sequence modeling
- k-mer DNA tokenization pipeline
- PyTorch Lightning training framework
- Mixed precision training (16-bit AMP)
- Early stopping + model checkpointing
- ROC-AUC / PR-AUC / confusion matrix evaluation
- Experiment tracking (train/val/test metrics)
- Fine-tuning strategy comparison

---

## Project Structure

```
genomic-sequence-classification/

├── data/
│   └── easy.csv

├── outputs/
│   ├── checkpoints/
│   ├── model/
│   ├── tokenizer/
│   └── figures/

├── Generate_SeqData.py
├── Transformer.py
├── requirements.txt
└── README.md
```

## Installation
```
git clone https://github.com/XuejianXiong/genomic-sequence-classification.git
cd genomic-sequence-classification

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Training

Run training with:

```python Transformer.py```

To change tuning strategy:
```
tuning_value = 0   # frozen backbone
tuning_value = -2  # last 2 layers
tuning_value = 1   # full fine-tuning
```

## Outputs

After training, the pipeline generates:

- Best model checkpoint
- Saved tokenizer + model
- Test predictions (test_predictions.csv)
- ROC curve
- Precision-recall curve
- Confusion matrix
- Full evaluation report

## Future Work

- Extend to promoter prediction
- Multi-class regulatory element classification
- Integration with ENCODE / FANTOM5 datasets
- Chromosome-level train/test splitting
- Add CNN and hybrid CNN–Transformer models
- Improve interpretability (attention analysis)

### Notes

This project is designed as a modular deep learning framework for genomic sequence classification and can be extended to real-world regulatory genomics datasets.
