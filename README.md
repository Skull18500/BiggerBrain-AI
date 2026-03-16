# BiggerBrain-AI  
The official repository for BiggerBrain, an implementation of the MoLaMaRT (Mixture of Layers and Memory, augmented Recurrent Transformer) architecture.

What is MoLaMaRT? MoLaMaRT is a hyper-advanced Recurrent Transformer designed for maximal reasoning density per parameter. Key features include:

# Memory: 
Uses a GRU and a custom memory manager to maintain and gate long-term dependencies.

# RoPE Attention: 
Implements Rotary Positional Embeddings for superior relative position awareness.

# Mixture of Layers (MoL): 
A novel concept designed for this project. MoL is more parameter-efficient than standard Mixture of Experts (MoE), allowing for a higher layer count without the usual parameter bloat.

# Recurrent Transformer: 
Approximately 50% of the architecture consists of recurrent blocks that allow the model to "iterate" on a token's representation.20-40% of layers are "single-shot" pre-loop.The remainder are dedicated to post-loop language clarification and output refining.

# Usage:
To run, run AI_main.py to test the project. It will open the CLI. The CLI is case-insensitive and supports the following commands:

train : Trains on the file specified in trainfilename using default settings.

pretrain : Trains on a custom filename with a specified learning rate.

debugmode : Enables debug printing (WIP).

profile : Runs the PyTorch profiler to identify performance bottlenecks.

speedtest : Benchmarks model loading and initialization speeds.

filesize : Returns the size of the specified dataset file.

quit : Safely exits the program.

[Any Prompt] : Entering any other text will run the AI on your input.


The actual code behind the AI is in biggerbrain.py.

Dependencies:

Python 3.12,

pip,

Pytorch newest version,

tiktoken,

bitsandbytes,

Probably a CUDA compatable GPU?
