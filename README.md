# Running Cogni-Brain on DGX Spark · Muse-Glimmer-30B + DFlash Speculative Decoding

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Base Model](https://img.shields.io/badge/base%20model-Muse--Glimmer--30B--NVFP4--W4A4-cyan)
![Drafter](https://img.shields.io/badge/drafter-DFlash--Speculative--Decoding-magenta)
![Runtime](https://img.shields.io/badge/runtime-vllm%2Fvllm--openai%3Amuse--glimmer-orange)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Context](https://img.shields.io/badge/context-128K-blue)
![Tool Calling](https://img.shields.io/badge/tool--calling-muse__glimmer-green)
![Reasoning](https://img.shields.io/badge/reasoning-muse__glimmer-black)
![Quantization](https://img.shields.io/badge/quantization-NVFP4%20W4A4-purple)

This repository contains the production-grade DGX Spark inference configuration for [Inferact/Muse-Glimmer-30B-NVFP4-W4A4](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4) paired with **DFlash Speculative Decoding**.

> ⚠️ **Personal workstation setup. Not for enterprise use. Use at your own risk.**

---

## Why Muse-Glimmer 30B + DFlash on DGX Spark?

### 1. Solving the DGX Spark Memory Bandwidth Bottleneck
On unified-memory hardware like the NVIDIA DGX Spark (Grace-Blackwell GB10 with 128 GB unified memory), autoregressive decoding for dense 30B models is bottlenecked by memory bandwidth rather than compute. Streaming 30B weights sequentially results in ~5–8 tok/s baseline.

**The DFlash Breakthrough:**
By pairing the dense 30B model with a lightweight DFlash assistant drafter (`meta-models/Muse-Glimmer-30B-assistant`) with $K = 15$ speculative draft tokens:
- The drafter proposes candidate tokens at high compute speeds.
- The 30B model verifies the batch in a single forward pass, reading model weights once across multiple accepted tokens.
- **Generation speed leaps from ~5–8 tok/s to 23+ tok/s** without losing 1 bit of target model intelligence or accuracy!

```
Standard Autoregressive (Dense 30B):
[Token 1: Read 16GB] ──> [Token 2: Read 16GB] ──> [Token 3: Read 16GB]  (~5–8 tok/s)

DFlash Speculative Decoding (Dense 30B + Drafter):
[Draft 15 tokens fast] ──> [Single Batched 30B Verification: Read 16GB once] ──> (23+ tok/s)
```

### 2. Frontier Intelligence vs MoE / Omni Models
According to the **Artificial Analysis Intelligence Index** and empirical agent benchmarks:
- **Muse-Glimmer (high reasoning)** achieves an intelligence rating of **35**, outperforming Qwen 3.5 122B MoE (33), Qwen 3.6 35B MoE (32), Claude 4.5 Haiku (30), and Nemotron 3 Super (26).
- **Zero Routing Degradation:** Unlike Mixture-of-Experts models that suffer expert drift over long multi-turn contexts, Muse-Glimmer's dense 30B parameters participate in every reasoning step.
- **100% Tool Parameter Precision:** 100% precision on multi-step chains and parameter extraction via native `muse_glimmer` parsers.

---

## Quick Start

> ⚠️ **Warning:** `setup/install.sh` disables system swap permanently to prevent unified-memory thrashing. This is a system-level change that survives reboots.

```bash
# 1. Set your Hugging Face token
export HF_TOKEN="your_hf_token_here"

# 2. System prerequisites (swap disable, uv install, docker check)
bash setup/install.sh

# 3. Download target model + DFlash drafter weights (~20 GB total)
bash setup/download_model.sh

# 4. Preflight check — validates GPU, memory, weights, Docker image, and DFlash flags
bash setup/preflight.sh

# 5. Start vLLM container with DFlash speculative decoding
bash docker/start.sh

# 6. Check container status & speculative health
bash docker/status.sh

# 7. Run benchmarks
uv run benchmark/benchmark_speed.py
uv run benchmark/benchmark_smarts.py
# Overnight full sweep:
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv
```

---

## Benchmarks

### Speed Benchmark (`benchmark_speed.py`)

```bash
# Full run (TPS, TTFT, concurrent sessions, context window)
uv run benchmark/benchmark_speed.py

# Skip slower subtests for a quicker check
uv run benchmark/benchmark_speed.py --skip-context
uv run benchmark/benchmark_speed.py --skip-context --skip-concurrent

# Custom endpoint or model alias
uv run benchmark/benchmark_speed.py --host localhost --port 8000 --model Cogni-Brain
```

> **Metrics Tracked:** Single-session TPS, peak TPS, TTFT under deep reasoning chains, concurrent session throughput (1–4 clients), and 128K context scaling.

### Smarts Benchmark (`benchmark_smarts.py` · `tool-eval-bench`)

```bash
# Quick smoke test (15 scenarios)
uv run benchmark/benchmark_smarts.py

# Throughput sweep
uv run benchmark/benchmark_smarts.py --mode perf

# Deterministic multi-trial evaluation
uv run benchmark/benchmark_smarts.py --mode trials --seed 42 --trials 3
```

### Spark Arena Sweep (`benchmark_speed_arena.py` · `llama-benchy`)

```bash
# Standard spark-arena sweep (context depths up to 128K with prefix caching)
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv
```

---

## Benchmark Results Summary

> Benchmarks run on DGX Spark GB10 · August 2026  
> Model: `Inferact/Muse-Glimmer-30B-NVFP4-W4A4` + `meta-models/Muse-Glimmer-30B-assistant`  
> vLLM `v0.1.dev19075+gd89ec6d6a` · tool-eval-bench `v2.3.1.dev8+g7fa6dd70b`

### Speed Benchmark (`benchmark_speed.py`)

| Test | Metric | Standard NVFP4 | NVFP4 + DFlash Speculative |
|---|---|---|---|
| Single Session Baseline | Average TPS | 8.2 tok/s | **22.4 tok/s** |
| Single Session Baseline | Peak TPS | 11.5 tok/s | **24.8 tok/s** |
| Output Length 600 tok | TPS | 12.1 tok/s | **29.9 tok/s** |
| Concurrent — 2 sessions | Total TPS | 14.5 tok/s | **26.8 tok/s** |
| Concurrent — 4 sessions | Total TPS | 22.0 tok/s | **48.2 tok/s** |
| Context window | Max verified | **130,753 tokens ✓** | **130,753 tokens ✓** |

> **Note on TTFT & Reasoning:** Muse-Glimmer executes a deep internal `<thought>` chain before emitting answer tokens. Token calculations include both reasoning and content tokens.

### Smarts Benchmark (`tool-eval-bench` — 15 Scenarios)

| Category | Score | Result |
|---|---|---|
| Tool Selection | 67% | 4/6 |
| Parameter Precision | **100%** | 6/6 |
| Multi-Step Chains | **100%** | 6/6 |
| Restraint & Refusal | **100%** | 6/6 |
| Error Recovery | 67% | 4/6 |
| **Overall Quality** | **87 / 100** | **13 pass · 0 partial · 2 fail** |

---

## Hardware & Architecture

- **NVIDIA DGX Spark Mini PC** (GB10 Grace-Blackwell Superchip)
- **128 GB Unified Memory** (CPU + GPU shared)
- **NVFP4 W4A4 Quantization** — Weights and activations quantized to 4-bit, executed on Grace-Blackwell Tensor Cores
- **DFlash Drafter Assistant** — Speculative drafting eliminating memory bandwidth stalls
- **Native Parsers** — Zero regex overhead with `muse_glimmer` tool-call and reasoning parsers
- **128K Context Window** (`--max-model-len 131072`)

---

## Key Configuration Reference

| Parameter | Value | Technical Rationale |
|---|---|---|
| Target Model | `Inferact/Muse-Glimmer-30B-NVFP4-W4A4` | Dense 30B reasoning model with NVFP4 weights + activations |
| Drafter Model | `meta-models/Muse-Glimmer-30B-assistant` | Speculative assistant model for DFlash acceleration |
| `--speculative-config` | `{"method":"dflash","num_speculative_tokens":15,...}` | 15 draft tokens per verification step |
| `--load-format` | `fastsafetensors` | Zero-copy mmap loading, drops container init to <60s |
| `--attention-backend` | `triton_attn` | Blackwell GB10 optimized Triton attention kernel |
| `--enable-chunked-prefill` | on | Prevents TTFT spikes during concurrent multi-turn requests |
| `--gpu-memory-utilization` | `0.92` | Optimal VRAM allocation on 128 GB DGX Spark |
| `--kv-cache-dtype` | `fp8` | Native Blackwell FP8 KV cache (~25% memory savings) |
| `--kv-cache-memory-bytes` | `4147483648` | Dedicated 4 GB KV pool boundary |
| `--max-num-batched-tokens` | `8196` | High throughput batching threshold |
| `--max-model-len` | `131072` | Full 128K context window |
| `--override-generation-config` | `{"temperature": 0.7, "top_p": 0.8, "top_k": 20, ...}` | Balances creative reasoning with deterministic tool payload schema |
| `--disable-log-stats` | on | Removes Prometheus polling locks from hot inference loop |

---

## Raw Docker Command (Reference)

```bash
docker run -d \
  --name spark-brain \
  --gpus all \
  --shm-size=16gb \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:muse-glimmer \
  Inferact/Muse-Glimmer-30B-NVFP4-W4A4 \
  --served-model-name Cogni-Brain \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 4147483648 \
  --max-num-batched-tokens 8196 \
  --max-num-seqs 8 \
  --load-format fastsafetensors \
  --attention-backend triton_attn \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dflash","num_speculative_tokens": 15, "model": "meta-models/Muse-Glimmer-30B-assistant"}' \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0, "repetition_penalty": 1.0}' \
  --disable-log-stats
```

---

## Repository Structure

```text
dgx-spark-muse-glimmer-agent/
├── README.md                     ← this file
├── LICENSE                       ← MIT
├── CITATION.cff                  ← citation metadata
├── setup/
│   ├── install.sh                ← prerequisites, swap disable, uv install
│   ├── download_model.sh         ← fetch target NVFP4 model & DFlash drafter weights
│   └── preflight.sh              ← pre-inference system, memory, and flag validation
├── docker/
│   ├── start.sh                  ← launch spark-brain with DFlash & GB10 tuning
│   ├── stop.sh                   ← stop and remove container
│   └── status.sh                 ← check container, speculative status, KV cache, VmSwap
└── benchmark/
    ├── benchmark_speed.py        ← TPS, TTFT, context-window benchmark
    ├── benchmark_speed_arena.py  ← overnight spark-arena llama-benchy sweep
    └── benchmark_smarts.py       ← tool-eval-bench capability benchmark
```

---

**Author:** Rajendra Rawat · August 2026
