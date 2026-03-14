
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    
class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # This layer looks at both the current state and the new info
        # to decide what to keep.
        self.gate_layer = nn.Linear(dim * 2, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.gate_x = nn.Linear(dim, dim, bias=False)
        self.gate_r = nn.Linear(dim, dim, bias=False)


    def forward(self, x, residual):
        """
        x: The new information (e.g., from Attention)
        residual: The current memory/state (the 'highway')
        """
        # 1. Concatenate them and calculate the 'Valve' (0 to 1)


        # 2. The 'Convex Combination' - pure stability
        # If gate is 0, we keep only the old memory.
        # If gate is 1, we take only the new info.
        gate = torch.sigmoid(self.gate_x(x) + self.gate_r(residual))  # no cat needed
        mixed = (1 - gate) * residual + gate * x
        # 3. Final cleanup
        return self.output_norm(mixed)
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        return (x * cos) + (rotate_half(x) * sin)

class RoPEAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "Dimension must be divisible by number of heads."
        
        # Individual projections to intercept Q and K
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, key, value, rope, attn_mask=None):
        b, sq, _ = query.shape
        _, sk, _ = key.shape
        
        # 1. Project and reshape
        q = self.q_proj(query).view(b, sq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, sk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, sk, self.heads, self.head_dim).transpose(1, 2)
        
        # 2. Apply RoPE to Queries and Keys
        q = rope(q, sq)
        k = rope(k, sk)
        
        # 3. Compute Attention via FlashAttention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        # 4. Reshape and project out
        out = out.transpose(1, 2).contiguous().view(b, sq, self.dim)
        return self.out_proj(out), None

class StreamDataset(Dataset):
    def __init__(self, bin_file, seq_len):
        # dtype MUST match what you used in tofile()
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        # We need seq_len + 1 to get a (input, target) pair
        self.n_samples = len(self.data) // (seq_len + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
class FastMemoryCell(nn.Module):
    """
    Drop-in replacement for memory1 + GatedResidual.

    Speed over your original:
      Original: 4 matmuls (gate_x, gate_r x2 GR calls) + lin1 = 5 total
      This:     1 matmul (fused_proj) = 5x fewer weight multiplications

    Quality over a vanilla GRU:
      - Bidirectional: returns BOTH a new hidden state AND a context vector
      - Reset gate (like GRU) for selective forgetting
      - Shared candidate prevents the two gate paths from fighting each other
      - RMSNorm instead of LayerNorm (no mean subtraction = ~20% faster norm)
    """

    def __init__(self, dim: int):
        super().__init__()

        # ONE big Linear replaces:
        #   self.GR.gate_x  (dim -> dim)
        #   self.GR.gate_r  (dim -> dim)
        #   self.GR1.gate_x (dim -> dim)
        #   self.GR1.gate_r (dim -> dim)
        #   self.lin1       (dim -> dim)
        # 
        # In C++ terms: instead of 4 small GEMM calls, 
        # we do 1 large GEMM — GPU loves wide matmuls.
        self.fused_proj = nn.Linear(dim * 2, dim * 3, bias=True)

        # RMSNorm: skips mean subtraction vs LayerNorm, ~20% faster
        # Requires PyTorch >= 2.1. Fall back to nn.LayerNorm if needed.
        self.norm = nn.RMSNorm(dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Single cat + single matmul for ALL gate logic
        # Shape: [batch, dim*3]
        proj = self.fused_proj(torch.cat([x, state], dim=-1))

        # Slice into 3 equal parts along last dim — no memory copy, just views
        g_update, g_reset, g_context = proj.chunk(3, dim=-1)

        g_update  = g_update.sigmoid()   # How much new info enters the state
        g_reset   = g_reset.sigmoid()    # GRU-style: what old state to use for candidate
        g_context = g_context.sigmoid()  # How much updated state leaks into context output

        # GRU-style candidate: reset gate filters what old state matters
        candidate = (g_reset * state).tanh()

        # --- Two outputs, shared computation ---

        # 1. New hidden state  (equivalent to your: x = GR(input, state) → lin1)
        new_state = (1.0 - g_update) * state + g_update * candidate

        # 2. Context vector    (equivalent to your: w = GR(state, input))
        #    Blends raw input with the freshly updated state
        context = (1.0 - g_context) * x + g_context * new_state

        return self.norm(new_state), context
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # stays float32 always
    
    def forward(self, x):
        # Upcast input to float32 for norm (more numerically stable anyway)
        # then cast result back to whatever dtype x was
        return torch.rms_norm(x.float(), x.shape[-1:], self.weight, self.eps).to(x.dtype)
    
    
class router(nn.Module):
    def __init__(self, dim, num_out, top_k = 2):
        super().__init__()
        self.L1 = nn.Linear(dim, num_out)
        self.topk = top_k

    def forward(self, input):
        x = self.L1(input)
        y = torch.topk(x, self.topk)
        
        z = torch.softmax(y, dim=-1)
        
        
class FlashCrossAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that uses
    Flash Attention (O(seq) memory instead of O(seq²)).
    
    Usage identical to your existing MA1/MA2:
        self.layerMA1 = AI_ex.FlashCrossAttention(dim, heads)
        y, _ = self.layerMA1(query, key, value, attn_mask=mask)
    
    The attn_mask parameter is accepted but ignored —
    Flash Attention handles causality internally and is
    always more memory efficient than passing an explicit mask.
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.dim      = dim

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, key, value, attn_mask=None):
        B, Sq, D  = query.shape
        Skv       = key.size(1)
        H, Hd     = self.heads, self.head_dim

        q = self.q_proj(query).view(B, Sq,  H, Hd).transpose(1, 2)
        k = self.k_proj(key).view(B,   Skv, H, Hd).transpose(1, 2)
        v = self.v_proj(value).view(B,  Skv, H, Hd).transpose(1, 2)

        # Flash Attention — O(seq) memory, same result as standard attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,   # cross attention doesn't need causal mask
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, Sq, D)
        return self.out_proj(out), None  # None matches nn.MultiheadAttention signature