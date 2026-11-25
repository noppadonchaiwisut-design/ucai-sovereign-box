# `ucai-sovereign-box` (v0.2.0 — Tier-2 Full-Corpus Single-GPU Edition)

[![License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Benchmark-CC_BY_4.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

**Single-GPU Air-Gapped Research-in-a-Box** uniting full-corpus hybrid retrieval (**Inverted Okapi BM25 + `all-MiniLM-L6-v2`**), **Spoke 1** cross-document numeric provenance + `DOC_NOT_INDEXED` abstention, **Spoke 2** prompt-injection defense, optional **local open-weight LLM verification** (`Qwen/Qwen2.5-1.5B-Instruct` on Tesla T4), and **`RO-Crate 1.1` + `SHA-256`** reproducibility packaging at **`$0.00` cloud cost**.

Developed as **Paper 3 (Spoke 3)** of the **UCAI Flagship 4** monograph (*Open Verifiable AI Co-Scientist & Research-in-a-Box*).

---

## Quick Start

```bash
pip install -e ".[eval]"
python -m unittest discover -s tests -v

# Run full-corpus benchmark (BM25-Only, Dense-Only, Sovereign Box + Local Qwen2.5-1.5B on T4 GPU):
python cli.py --pdf-dir /path/to/pdfs --out-dir results_tier2 --run-local-llm
```
