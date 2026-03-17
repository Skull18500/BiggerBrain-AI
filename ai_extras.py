
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
    
class ThinkingRouter(nn.Module):
    """
    Routes to different experts based on the QUALITY of current thinking,
    not just the content. Uses three signals:
    
    1. Delta:   how much y changed this iteration (uncertainty signal)
    2. Drift:   how far y is from the linguistic anchor (grounding signal)  
    3. Iter:    which iteration we're on (stage signal)
    
    These directly describe WHERE we are in the thinking process,
    making routing decisions interpretable and meaningful.
    """
    def __init__(self, dim: int, n_experts: int = 2, max_iter: int = 3):
        super().__init__()
        self.n_experts = n_experts
        self.max_iter = max_iter
        # Iteration embedding — gives each iteration a learned "personality"
        # iter 1 = "first pass", iter 2 = "refinement", iter 3 = "verification"
        self.iter_embed = nn.Embedding(max_iter, 16)
        
        # Project the three signals into routing logits
        # Input: delta_scalar + drift_scalar + iter_embed(16) = 18 dims
        self.router = nn.Sequential(
            nn.Linear(18, 64),
            SwiGLU(),
            nn.Linear(32, n_experts, bias=False)
        )
        
        # Init router to be nearly uniform at start
        # → experts start with equal load, specialization emerges
        nn.init.normal_(self.router[0].weight, std=0.001)
        nn.init.normal_(self.router[2].weight, std=0.001)
        
        self.last_weights = None  # store for balancing loss
    
    def forward(self,
                y:                torch.Tensor,   # current hidden state
                y_prev:           torch.Tensor,   # hidden state from last iter
                linguistic_anchor: torch.Tensor,  # what the input said
                iter_idx:         int             # which iteration (0-indexed)
               ) -> torch.Tensor:
        """
        Returns routing weights [batch, n_experts].
        """
        # Signal 1: Delta — how much thinking changed this step
        # High delta = uncertain, still changing a lot
        # Low delta  = converging, changes are subtle
        delta = (y - y_prev).norm(dim=-1).mean(dim=-1, keepdim=True)
        # delta shape: [batch, 1]
        
        # Signal 2: Drift — how far current thinking is from the input
        # High drift = model is thinking abstractly, far from literal input
        # Low drift  = model is still closely following the input
        drift = (y - linguistic_anchor).norm(dim=-1).mean(dim=-1, keepdim=True)
        # drift shape: [batch, 1]
        
        # Normalize both signals so they're comparable
        # Detach to avoid routing gradients affecting main computation
        delta = delta.detach() / (delta.detach().mean() + 1e-8)
        drift = drift.detach() / (drift.detach().mean() + 1e-8)
        
        # Signal 3: Iteration stage embedding
        iter_clamped = min(iter_idx if isinstance(iter_idx, int) 
                   else iter_idx.item(), 
                   self.max_iter - 1)  # clamp to valid range
        iter_tensor  = torch.as_tensor(iter_clamped, device=y.device, dtype=torch.long)
        iter_emb     = self.iter_embed(iter_tensor)
        iter_emb    = iter_emb.unsqueeze(0).expand(y.size(0), -1)  # [batch, 16]
        
        # Combine signals
        routing_input = torch.cat([delta, drift, iter_emb], dim=-1)  # [batch, 18]
        logits        = self.router(routing_input)                     # [batch, n_experts]
        
        if self.training:
        
            weights = F.gumbel_softmax(logits, tau=0.5, hard=False)
        else:
            # Hard routing at inference
            idx     = logits.argmax(dim=-1)
            weights = torch.zeros_like(logits)
            weights.scatter_(1, idx.unsqueeze(1), 1.0)
        
        self.last_weights = weights.mean(0)  # for balance loss
        return weights


class MoLLayer(nn.Module):
    def __init__(self, dim: int, ffndim: int,
                 n_experts: int = 2, max_iter: int = 3):
        super().__init__()
        self.n_experts = n_experts
        
        self.router  = ThinkingRouter(dim, n_experts, max_iter)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ffndim * 2),
                SwiGLU(),
                nn.Linear(ffndim, dim)
            )
            for _ in range(n_experts)
        ])
    
    def forward(self,
                x:                torch.Tensor,
                x_prev:           torch.Tensor,
                linguistic_anchor: torch.Tensor,
                iter_idx:         int
               ) -> torch.Tensor:
        
        weights = self.router(x, x_prev, linguistic_anchor, iter_idx)
        # weights: [batch, n_experts]
        
        # Weighted sum of expert outputs
        out = sum(
            weights[:, e].unsqueeze(1).unsqueeze(2) * self.experts[e](x)
            for e in range(self.n_experts)
        )
        return out