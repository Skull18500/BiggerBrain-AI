# BiggerBrain-AI

Official repository for **BiggerBrain**, an implementation of the 
**MoLaMaRT** (Mixture of Layers and Memory augmented Recurrent Transformer) 
architecture — a custom LLM designed for maximum reasoning density per parameter.

## What is MoLaMaRT?

MoLaMaRT is a novel transformer architecture built around the idea that 
iterative refinement of token representations produces better reasoning 
than a single deep forward pass.

### Architecture Overview
```
Input → Linguistic Blocks → [Thinking Loop × N] → Output Block → Logits
              ↑                      ↑
         (single pass)        (shared weights,
         syntax/morphology     run 1-3 times)
```

### Key Features

**Memory System**
- `FastMemoryCell` — learned scratchpad with write gate
  (only writes when something important happened)
- `GRU` — long-term sequence context
- Linguistic anchor — prevents thinking loop from drifting 
  away from input meaning

**Mixture of Layers (MoL)**
- Novel routing mechanism distinct from standard MoE
- Routes tokens to different expert FFN layers based on 
  thinking quality signals (delta, drift, iteration stage)
- More parameter-efficient than MoE — compute cost of 1 expert,
  capacity of N experts

**Recurrent Transformer (RT)**
- ~30-40% single-pass linguistic pre-blocks
- ~50% shared-weight thinking blocks (run 1-3 iterations)
- ~10-20% post-loop output refinement
- Random iteration training — model learns to be correct 
  at ALL thinking depths, not just maximum depth

**RoPE Attention**
- Rotary Positional Embeddings throughout
- FlashCrossAttention for memory queries (O(seq) not O(seq²))
- SwiGLU activations

### Architecture Taxonomy

This project introduces several named concepts:

| Name | Definition |
|------|-----------|
| RT (Recurrent Transformer) | Transformer with ~50% of blocks in a recurrent loop |
| MaRT (Memory augmented RT) | RT with persistent scratchpad and long-term memory |
| MoLaRT (Mixture of Layers RT) | RT with per-iteration expert routing |
| MoLaMaRT | Full combination of all the above |

## Usage

Run `AI_main.py` to open the CLI (case-insensitive):

| Command | Description |
|---------|-------------|
| `pretrain` | Train on pretraining dataset |
| `train` | Train on finetuning dataset |
| `debugmode` | Toggle debug printing |
| `profile` | Run PyTorch profiler |
| `speedtest` | Benchmark model initialization |
| `filesize` | Check dataset file size |
| `quit` | Exit safely |
| *(any text)* | Run inference on your input |

## Requirements
```bash
pip install torch tiktoken bitsandbytes numpy datasets umap-learn plotly
```

- Python 3.12+
- CUDA-compatible GPU (tested on RTX 4060 8GB)
- PyTorch 2.x

## Project Structure
```
BiggerBrain-AI/
├── AI_main.py          # Entry point, CLI
├── biggerbrain.py      # Model architecture (MoLaMaRT)
├── ai_extras.py        # Custom layers (RoPE, FlashCrossAttention, 
│                       #   MoLLayer, FastMemoryCell, RMSNorm, etc.)
├── training_utils.py   # Dataset tokenization utilities
└── viewing.py          # 3D thought trajectory visualization
```

## Visualization

BiggerBrain includes a 3D visualization of token trajectories 
across thinking iterations using UMAP dimensionality reduction.
Run `viewing.py` to see how token representations evolve through 
the thinking loop.

## Status

Currently in pretraining phase on ~6B tokens of mixed text data
(Project Gutenberg, Wikipedia, OpenWebText, TinyStories).

## Author

Built by a 13-year-old ML enthusiast.
