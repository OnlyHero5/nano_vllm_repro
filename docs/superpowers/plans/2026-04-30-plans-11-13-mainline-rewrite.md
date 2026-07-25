# Plans 11-13 Mainline Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `plans/11` through `plans/13` so they are concrete nano-vLLM learning-project guides based on this repository's actual code boundaries.

**Architecture:** Only Markdown guides change. Each guide must describe file-level mainline modifications, complete code blocks to add or replace, tests to create, and exact verification commands. Production vLLM details are cited only as learning references, not as required scope.

**Tech Stack:** Markdown, Python code blocks for the existing nano-vLLM toy project, PyTorch, current repository modules under `config.py`, `models/`, `layers/`, `engine/`, `utils/`, and `tests/`.

---

## File Structure

- Modify: `plans/11-实现MoE推理主线与专家并行认知篇.md`
  - Rewrite from standalone prototype to mainline-compatible teaching patch.
  - Document modifications to `config.py`, `models/qwen3.py`, `utils/loader.py`, and `tests/test_Day11_moe.py`.
- Modify: `plans/12-实现FP8与KV-Cache量化实验篇.md`
  - Rewrite from standalone proto to optional mainline KV cache quantization teaching path.
  - Document modifications to `config.py`, `utils/kvcache_quant.py`, `engine/model_runner.py`, `utils/context.py`, `layers/attention.py`, and `tests/test_Day12_kvcache_quant.py`.
- Modify: `plans/13-实现GPU-Offload与跨后端扩展总览.md`
  - Rewrite from offload/cross-backend overview to mainline CPU KV block offload teaching path.
  - Document modifications to `engine/sequence.py`, `engine/block_manager.py`, `engine/model_runner.py`, `engine/scheduler.py`, and `tests/test_Day13_kv_offload.py`.

## Task 1: Rewrite Day11 MoE Guide

**Files:**
- Modify: `plans/11-实现MoE推理主线与专家并行认知篇.md`

- [ ] Replace future/prototype framing with current-repo mainline patch framing.
- [ ] Include real references: Hugging Face Qwen3MoE, vLLM Qwen3MoE/FusedMoE, nano-vLLM MoE PR.
- [ ] Add complete code blocks for `config.py` fields, `models/qwen3.py` MoE classes, dense/MoE layer selection, loader mapping notes, and `tests/test_Day11_moe.py`.
- [ ] Add exact commands:

```bash
python -m py_compile models/qwen3.py tests/test_Day11_moe.py
python tests/test_Day11_moe.py
```

## Task 2: Rewrite Day12 KV Cache Quantization Guide

**Files:**
- Modify: `plans/12-实现FP8与KV-Cache量化实验篇.md`

- [ ] Replace standalone-only proto framing with optional mainline `int8_sim` / `pseudo_fp8_sim` KV cache path.
- [ ] State clearly that this is not production FP8 kernel support.
- [ ] Add complete code blocks for config, quantization utility, allocation changes, context typing, attention store/dequant hooks, and tests.
- [ ] Add exact commands:

```bash
python -m py_compile utils/kvcache_quant.py tests/test_Day12_kvcache_quant.py
python tests/test_Day12_kvcache_quant.py
```

## Task 3: Rewrite Day13 KV Offload Guide

**Files:**
- Modify: `plans/13-实现GPU-Offload与跨后端扩展总览.md`

- [ ] Replace overview/prototype framing with current mainline CPU KV block offload teaching patch.
- [ ] Exclude production KV connector, LMCache, async prefetch, and cross-backend implementation from required scope.
- [ ] Add complete code blocks for sequence `SWAPPED`, block residency metadata, swap-in/out block manager logic, model runner CPU KV buffer copy, scheduler queue changes, and tests.
- [ ] Add exact commands:

```bash
python -m py_compile engine/sequence.py engine/block_manager.py engine/model_runner.py engine/scheduler.py tests/test_Day13_kv_offload.py
python tests/test_Day13_kv_offload.py
```

## Task 4: Self-Review

**Files:**
- Check: `plans/11-实现MoE推理主线与专家并行认知篇.md`
- Check: `plans/12-实现FP8与KV-Cache量化实验篇.md`
- Check: `plans/13-实现GPU-Offload与跨后端扩展总览.md`

- [ ] Scan for banned vague wording: `未来如果`, `以后真要`, `原型不是主线`, `省略`, `TODO`, `TBD`, `自行补`.
- [ ] Check each guide names exact files to create or modify.
- [ ] Check each guide contains complete code blocks for the named new test files.
- [ ] Check each guide remains scoped to a nano-vLLM learning project.
