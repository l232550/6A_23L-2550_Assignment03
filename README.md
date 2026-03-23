# HSRIS — Hybrid Semantic Retrieval & Intelligence System

A multi-stage NLP pipeline for intelligent customer support ticket retrieval, 
built from scratch using base PyTorch — no sklearn.

## Live Demo
🚀 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/tooba-1234/hybrid-ticket-search)

## Overview
This system retrieves similar past support tickets given a new query by 
blending two retrieval methods:
- **TF-IDF (keyword)** — finds tickets sharing the same words
- **GloVe (semantic)** — finds tickets with the same meaning

## Pipeline
```
Dataset (8,469 tickets)
→ Label Encoding (Ticket Priority)
→ One-Hot Encoding (Ticket Channel)  
→ Custom Tokenizer + N-grams
→ TF-IDF (torch.sparse tensor)
→ GloVe 300-d Embeddings (weighted pooling)
→ Hybrid Search (α blending)
→ Dual T4 GPU optimization
```

## Key Results
| Metric | Value |
|--------|-------|
| Precision@5 (best) | 0.213 |
| GPU throughput | 231,000 queries/sec |
| TF-IDF sparsity | 98.78% |
| GloVe vocabulary | 400,000 words |

## Tech Stack
- Python, PyTorch, NumPy, Pandas
- GloVe 300-d pretrained embeddings
- Kaggle Dual T4 x2 GPU
- Gradio + Hugging Face Spaces

## Files
| File | Description |
|------|-------------|
| `DS_ASS03_23L_2550.ipynb` | Main Kaggle notebook |
| `app.py` | Gradio app for deployment |
| `requirements.txt` | Dependencies |

## Assignment
DS Assignment 3 — Hybrid Semantic Retrieval & Intelligence System  
Platform: Kaggle Dual T4 x2 GPU
