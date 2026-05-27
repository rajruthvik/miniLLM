# MiniLLM

Modular Transformer Inference Engine built from scratch using PyTorch.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-GPU_Acceleration-green?style=for-the-badge&logo=nvidia)
![FastAPI](https://img.shields.io/badge/FastAPI-API_Server-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?style=for-the-badge&logo=docker)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=for-the-badge&logo=numpy)

---

## Overview

MiniLLM is a modular transformer inference engine focused on understanding and engineering the internals of modern Large Language Model systems.

The project explores:
- Transformer architecture
- Self-attention mechanisms
- Autoregressive generation
- KV cache optimization
- Quantized inference
- Runtime benchmarking
- Local GPU inference
- FastAPI-based serving

Instead of relying entirely on external APIs, MiniLLM focuses on building and understanding the inference pipeline itself.

---

## System Architecture

```text
                 ┌────────────────────┐
                 │    User Prompt     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │     Tokenizer      │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   Embedding Layer  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Transformer Blocks │
                 │  Self Attention    │
                 │ Feed Forward Net   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Inference Runtime  │
                 │  KV Cache Engine   │
                 │ Sampling Pipeline  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Generated Tokens   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ FastAPI Serving    │
                 └────────────────────┘
```

---

## Features

### Transformer Architecture
- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Feedforward layers
- Residual connections
- Layer normalization
- Autoregressive generation

---

### Inference Runtime
- Token-by-token generation
- Greedy decoding
- Top-K sampling
- Top-P sampling
- Temperature scaling
- CUDA acceleration
- CPU/GPU execution support

---

### Optimization & Systems Engineering
- KV cache support
- Runtime benchmarking
- Throughput analysis
- Latency profiling
- VRAM monitoring
- Quantization experiments

---

### API & Serving
- FastAPI inference server
- Async request handling
- Local deployment pipeline
- Streaming generation support

---

## Tech Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| GPU Acceleration | CUDA |
| API Framework | FastAPI |
| Numerical Computing | NumPy |
| Deployment | Docker |
| Vector Search | FAISS (planned) |

---

## Project Structure

```text
miniLLM/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/
│   ├── model/
│   ├── tokenizer/
│   ├── inference/
│   ├── training/
│   ├── config/
│   └── utils/
│
├── api/
│   ├── routes/
│   ├── schemas/
│   └── main.py
│
├── benchmarks/
├── tests/
├── docs/
├── scripts/
└── assets/
```

---

## Current Goals

- Build transformer blocks from scratch
- Implement custom tokenizer
- Add KV cache optimization
- Build optimized inference runtime
- Benchmark latency and throughput
- Add quantized inference support
- Implement FastAPI serving layer
- Add streaming token generation

---

## Planned Extensions

- Retrieval-Augmented Generation (RAG)
- FAISS vector database integration
- Concurrent inference batching
- Distributed inference experiments
- Dockerized deployment
- Local web-based inference UI

---

## Development Environment

- NVIDIA RTX 4060 Laptop GPU
- CUDA-enabled PyTorch
- Python 3.x

---

## Motivation

Most projects stop at consuming LLM APIs.

MiniLLM focuses on understanding and engineering the systems behind transformer inference, runtime optimization, scalable serving, and local AI infrastructure.
