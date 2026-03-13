from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tiktoken
import ai_extras as AI_ex
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, random_split
import numpy as np
import torch._dynamo
torch._dynamo.config.suppress_errors = False  # make errors visible

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('high')

enc = tiktoken.get_encoding("gpt2")

class biggerbrain(nn.Module):
    def __init__(self, device):
        super(biggerbrain, self).__init__()
        self.dim = 768
        self.dim1 = 768
        self.ffndim = int(self.dim1 * 2)# for the first and last layers.
        self.heads = 8
        self.sequencelength = 1024
        self.device = device
        
        self.debugprints = False
        
        self.embed = nn.Embedding(enc.max_token_value + 1, self.dim)
        self.layerO1 = nn.Linear(self.dim1, enc.max_token_value + 1, bias=False)#O1 is output1. This is the layer that produces the final output. It takes the processed data from the attention layer and produces a probability distribution over the vocabulary for the next word prediction.
        
        self.rope = AI_ex.RoPE(self.dim1 // self.heads, max_seq_len=self.sequencelength)
        
        self.layerMe1 = AI_ex.FastMemoryCell(self.dim1)#scratchpad
        self.layerMe2 = nn.GRU(self.dim1, self.dim1, batch_first=True)#long term memory
        
        self.layerMA1 = nn.MultiheadAttention(self.dim1, self.heads, batch_first=True)#Memory Attention #1
        self.layerMA2 = nn.MultiheadAttention(self.dim1, self.heads, batch_first=True)
        
        self.layerPreA1 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerPre1 = nn.Sequential(nn.Linear(self.dim1, self.ffndim * 2), AI_ex.SwiGLU(), nn.Linear(self.ffndim, self.dim1))
        self.normPre1 = AI_ex.RMSNorm(self.dim1)
        
        self.layerPostA1 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerPost1 = nn.Sequential(nn.Linear(self.dim1, self.ffndim * 2), AI_ex.SwiGLU(), nn.Linear(self.ffndim, self.dim1))
        self.normPost1 = AI_ex.RMSNorm(self.dim1)
        
        self.layerA1 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerA2 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerA3 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerA4 = AI_ex.RoPEAttention(self.dim1, self.heads)
        
        self.layerMi1 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        self.layerMi2 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        self.layerMi3 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        self.layerMi4 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        
        self.norm0 = AI_ex.RMSNorm(self.dim1)
        self.norm1 = AI_ex.RMSNorm(self.dim1) # Layer normalization for stabilizing training. This is a common technique used in transformer models to improve convergence and performance.
        self.norm2 = AI_ex.RMSNorm(self.dim1)
        self.norm3 = AI_ex.RMSNorm(self.dim1)
        self.norm4 = AI_ex.RMSNorm(self.dim1)
        
        self.scratchpad_gate = nn.Linear(self.dim1, 1, bias=True)
        
        self.MM1 = AI_ex.GatedResidual(self.dim1)#gated residual sigmoid thing. Memory Managment #1
        self.MM2 = AI_ex.GatedResidual(self.dim1)  # for linguistic anchor pull
        
        self._initialize_weights()
        
        nn.init.zeros_(self.scratchpad_gate.weight)
        nn.init.constant_(self.scratchpad_gate.bias, -1.0)
        
        self.layerO1.weight = self.embed.weight  # tied weights

    def pick_word(self, output, k=50, temperature=0.8):
        # output shape: (batch_size, vocab_size)
        # 1. Apply Temperature (higher = more random, lower = more confident)
        logits = output / (temperature + 1e-8)
        
        # 2. Top-K filtering
        v, _ = torch.topk(logits, min(k, logits.size(-1)))
        # Any logit smaller than the K-th value gets set to -inf
        logits[logits < v[:, [-1]]] = float('-inf')
        
        # 3. Softmax and Sample
        probabilities = torch.softmax(logits, dim=-1)
        word_id = torch.multinomial(probabilities, num_samples=1)
        
        return word_id

    def trainingloop(self, data, epochs=100, lr=3e-4, batchsize=32,
                 accumulation_steps=4, max_batches=None, 
                 subset_fraction=1.0):   # ← add this parameter
        self.train()

        # Build full index list once
        dataset_size = len(data)
        all_indices  = np.arange(dataset_size)
        np.random.shuffle(all_indices)  # shuffle once upfront

        # How many samples per epoch
        subset_size  = int(dataset_size * subset_fraction)
    
        # Split into chunks — one chunk per epoch
        # [0:25%], [25%:50%], [50%:75%], [75%:100%], then wraps around
        chunks = [
            all_indices[i * subset_size : (i + 1) * subset_size]
            for i in range(int(1.0 / subset_fraction))
        ]

        criterion   = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer   = torch.optim.AdamW(
            self.parameters(), lr=lr,
            betas=(0.9, 0.95), weight_decay=0.05
        )
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * (subset_size // batchsize)
        )
        best_loss   = 1000.0

        for epoch in range(epochs):
            epoch_loss  = 0.0
            batches_run = 0
            optimizer.zero_grad(set_to_none=True)

            # Pick which chunk this epoch uses — wraps after 4 epochs
            chunk_idx = epoch % len(chunks)
            sampler   = SubsetRandomSampler(chunks[chunk_idx])

            loader = DataLoader(
                data,
                batch_size=batchsize,
                sampler=sampler,       # ← uses this epoch's chunk
                num_workers=0,
                pin_memory=True,
                drop_last=True
            )

            if epoch % len(chunks) == 0 and epoch > 0:
                # Re-shuffle chunks every full cycle so next 4 epochs
                # see different groupings than the last 4
                np.random.shuffle(all_indices)
                chunks = [
                    all_indices[i * subset_size : (i + 1) * subset_size]
                    for i in range(int(1.0 / subset_fraction))
                ]
                print(f"  ↻ Reshuffled data chunks for next cycle")

            for i, (batch_inputs, batch_targets) in enumerate(loader):
                if i == 0:
                    print(f"Epoch {epoch} | Chunk {chunk_idx+1}/{len(chunks)} | "
                        f"Starting training...")

                batch_inputs  = batch_inputs.to(self.device,  non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    num_iter     = random.randint(1, 3)
                    logits, _    = self(batch_inputs, iter=num_iter, 
                                    is_training=True)
                    lm_loss      = criterion(
                        logits.view(-1, enc.max_token_value + 1),
                        batch_targets.reshape(-1)
                    )
                    loss         = lm_loss / accumulation_steps
                    epoch_loss  += lm_loss.detach()
                    batches_run += 1

                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            avg_loss  = (epoch_loss / batches_run).item()
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Chunk: {chunk_idx+1}/{len(chunks)}")

            current_loss = loss.detach().item()
            if best_loss > current_loss:
                best_loss = current_loss
                torch.save(self.state_dict(), "model_best.pt")
                
                
    def forward_with_hooks(self, input_ids, iter=3):
        """
        Same as forward but returns hidden states at each iteration.
        Use this for visualization only, not training.
        """
        self.eval()
        batch_size, seq_len = input_ids.size()
        x    = self.embed(input_ids)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
            diagonal=1
        )

        mem1 = torch.zeros(batch_size, self.dim1, device=self.device)
        mem2 = torch.zeros(1, batch_size, self.dim1, device=self.device)

        y, _ = self.layerA1(x, x, x, attn_mask=mask, rope=self.rope)

        last         = y[:, -1:, :]
        new_state, _ = self.layerMe1(last.squeeze(1), mem1)
        write_gate   = torch.sigmoid(self.scratchpad_gate(last.squeeze(1)))
        mem1         = write_gate * new_state + (1.0 - write_gate) * mem1
        m1_out       = mem1.unsqueeze(1).expand(-1, seq_len, -1)
        z            = y

        y = self.normPre1(y)
        o = y
        y, _ = self.layerPreA1(y, y, y, attn_mask=mask, rope=self.rope)
        y = y + o
        y = y + self.layerPre1(y)

        linguistic_anchor = y
        seq_context, _    = self.layerMe2(y, mem2)

        # ── Capture states at each iteration ────────────────────────────────
        states = []
        states.append(y.detach().cpu())  # state before loop = iter 0

        for j in range(iter):
            last         = y[:, -1:, :]
            new_state, _ = self.layerMe1(last.squeeze(1), mem1)
            write_gate   = torch.sigmoid(self.scratchpad_gate(last.squeeze(1)))
            mem1         = write_gate * new_state + (1.0 - write_gate) * mem1
            m1_out       = mem1.unsqueeze(1).expand(-1, seq_len, -1)

            residual = y
            y        = self.norm0(y)
            y, _     = self.layerA1(y, y, y, attn_mask=mask, rope=self.rope)
            y        = y + residual
            y, _     = self.layerMA1(y, m1_out,      m1_out,      attn_mask=mask)
            y, _     = self.layerMA2(y, seq_context, seq_context, attn_mask=mask)

            y = self.norm1(y);  y = y + self.layerMi1(y)
            y = self.norm2(y);  o = y
            y, _ = self.layerA2(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o;          y = y + self.layerMi2(y)
            y = self.norm3(y);  o = y
            y, _ = self.layerA3(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o;          y = y + self.layerMi3(y)
            y = self.norm4(y);  o = y
            y, _ = self.layerA4(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o;          y = y + self.layerMi4(y)

            y = self.MM1(y, z)
            y = self.MM2(y, linguistic_anchor)
            z = y

            states.append(y.detach().cpu())  # capture after each iter

        # states is now a list of [iter+1] tensors, each [batch, seq, 768]
        return states
    
    
    def _initialize_weights(self):
        for m in self.modules():
        # 1. Linear Layers (The Muscles)
            if isinstance(m, nn.Linear):
            # For LLMs, a small normal distribution is often more stable than Xavier
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 2. Embedding Layer (The Ears)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # 3. GRU Layers (The Memory)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        # Input-to-hidden weights
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        # Hidden-to-hidden weights - USE ORTHOGONAL!
                        # This is the secret to making recurrent layers actually remember things
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

            # 4. LayerNorm (The Stabilizer)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, outlength=1, iter=3, is_training=False):
        batch_size, seq_len = input_ids.size()
        x = self.embed(input_ids)

        # ── TRAINING LANE ────────────────────────────────────────────────────
        if is_training:
            total_iters = torch.tensor(float(iter), device=self.device)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
                diagonal=1
            )

            mem1 = torch.zeros(batch_size, self.dim1, device=self.device)
            mem2 = torch.zeros(1, batch_size, self.dim1, device=self.device)

            # Initial attention pass — separate from loop (no shared weights issue)
            y, _ = self.layerA1(x, x, x, attn_mask=mask, rope=self.rope)

            # Initial scratchpad update — feed last token only
            # (full sequence would break FastMemoryCell)
            last             = y[:, -1:, :]
            new_state, _     = self.layerMe1(last.squeeze(1), mem1)
            mem1             = new_state
            m1_out           = mem1.unsqueeze(1).expand(-1, seq_len, -1)

            z = y

            # Pre-block — linguistic feature extraction
            y = self.normPre1(y)
            o = y
            y, _ = self.layerPreA1(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o
            y = y + self.layerPre1(y)

            # ── LINGUISTIC ANCHOR ────────────────────────────────────────────
            # Save here — AFTER pre-block refines raw embeddings into
            # actual linguistic features (syntax, morphology, relationships)
            # but BEFORE the thinking loop starts potentially drifting.
            # Each iteration will be pulled back toward this representation.
            linguistic_anchor = y

            # Long-term memory context — computed once, read-only during loop
            seq_context, _ = self.layerMe2(y, mem2)

            for j in range(iter):

                residual = y
                y        = self.norm0(y)
                y, _     = self.layerA1(y, y, y, attn_mask=mask, rope=self.rope)
                y        = y + residual

                y, _ = self.layerMA1(y, m1_out,     m1_out,     attn_mask=mask)
                y, _ = self.layerMA2(y, seq_context, seq_context, attn_mask=mask)

                # ── SCRATCHPAD WRITE GATE ────────────────────────────────────
                # Instead of always updating memory, the gate learns WHEN
                # something worth remembering happened.
                # Gate near 0 = "nothing new, keep old memory"
                # Gate near 1 = "important update, write to scratchpad"
                last      = y[:, -1:, :]
                new_state, _ = self.layerMe1(last.squeeze(1), mem1)

                # Gate is conditioned on the last token's current representation
                write_gate = torch.sigmoid(
                    self.scratchpad_gate(last.squeeze(1))
                )  # [batch, 1]

                mem1   = write_gate * new_state + (1.0 - write_gate) * mem1
                m1_out = mem1.unsqueeze(1).expand(-1, seq_len, -1)

                # Block 1
                y = self.norm1(y)
                y = y + self.layerMi1(y)

                # Block 2
                y = self.norm2(y)
                o = y
                y, _ = self.layerA2(y, y, y, attn_mask=mask, rope=self.rope)
                y = y + o
                y = y + self.layerMi2(y)

                # Block 3
                y = self.norm3(y)
                o = y
                y, _ = self.layerA3(y, y, y, attn_mask=mask, rope=self.rope)
                y = y + o
                y = y + self.layerMi3(y)

                # Block 4
                y = self.norm4(y)
                o = y
                y, _ = self.layerA4(y, y, y, attn_mask=mask, rope=self.rope)
                y = y + o
                y = y + self.layerMi4(y)

                # Ground to previous iteration (MM1 — unchanged)
                y = self.MM1(y, z)

                # ── LINGUISTIC ANCHOR PULL (MM2) ─────────────────────────────
                # Pull thinking back toward what the input actually said.
                # Prevents iter3 from being pure "thinking about thinking."
                y = self.MM2(y, linguistic_anchor)

                z = y

            # Update long-term memory once after loop — last token summary
            _, mem2 = self.layerMe2(y[:, -1:, :], mem2)

            z    = self.normPost1(z)
            o    = z
            z, _ = self.layerPostA1(z, z, z, attn_mask=mask, rope=self.rope)
            z    = z + o
            z    = z + self.layerPost1(z)

            return self.layerO1(z), total_iters

        # ── CHAT LANE ────────────────────────────────────────────────────────
        else:
            output = torch.zeros(
                batch_size, outlength, enc.max_token_value + 1, 
                device=self.device
            )
            mem1 = torch.zeros(batch_size, self.dim1, device=self.device)
            mem2 = torch.zeros(1, batch_size, self.dim1, device=self.device)

            for word in range(outlength):
                current_input_ids = input_ids[:, -self.sequencelength:]
                curr_seq_len      = current_input_ids.size(1)
                w    = self.embed(current_input_ids)
                mask = torch.triu(
                    torch.ones(curr_seq_len, curr_seq_len, device=self.device) * float('-inf'),
                    diagonal=1
                )

                y, _ = self.layerA1(w, w, w, attn_mask=mask, rope=self.rope)

                # ── Fixed: FastMemoryCell takes single token, not full sequence
                last         = y[:, -1:, :]
                new_state, _ = self.layerMe1(last.squeeze(1), mem1)
                write_gate   = torch.sigmoid(self.scratchpad_gate(last.squeeze(1)))
                mem1         = write_gate * new_state + (1.0 - write_gate) * mem1
                m1_out       = mem1.unsqueeze(1).expand(-1, curr_seq_len, -1)

                z = y

                y    = self.normPre1(y)
                o    = y
                y, _ = self.layerPreA1(y, y, y, attn_mask=mask, rope=self.rope)
                y    = y + o
                y    = y + self.layerPre1(y)

                # Linguistic anchor — same as training lane
                linguistic_anchor = y

                seq_context, _ = self.layerMe2(y, mem2)

                for j in range(iter):
                    residual = y
                    y        = self.norm0(y)
                    y, _     = self.layerA1(y, y, y, attn_mask=mask, rope=self.rope)
                    y        = y + residual

                    y, _ = self.layerMA1(y, m1_out,     m1_out,     attn_mask=mask)
                    y, _ = self.layerMA2(y, seq_context, seq_context, attn_mask=mask)

                # Scratchpad write gate
                    last         = y[:, -1:, :]
                    new_state, _ = self.layerMe1(last.squeeze(1), mem1)
                    write_gate   = torch.sigmoid(self.scratchpad_gate(last.squeeze(1)))
                    mem1         = write_gate * new_state + (1.0 - write_gate) * mem1
                    m1_out       = mem1.unsqueeze(1).expand(-1, curr_seq_len, -1)

                    # Block 1
                    y = self.norm1(y)
                    y = y + self.layerMi1(y)

                    # Block 2
                    y = self.norm2(y)
                    o = y
                    y, _ = self.layerA2(y, y, y, attn_mask=mask, rope=self.rope)
                    y = y + o
                    y = y + self.layerMi2(y)

                    # Block 3
                    y = self.norm3(y)
                    o = y
                    y, _ = self.layerA3(y, y, y, attn_mask=mask, rope=self.rope)
                    y = y + o
                    y = y + self.layerMi3(y)

                    # Block 4
                    y = self.norm4(y)
                    o = y
                    y, _ = self.layerA4(y, y, y, attn_mask=mask, rope=self.rope)
                    y = y + o
                    y = y + self.layerMi4(y)

                    y = self.MM1(y, z)
                    y = self.MM2(y, linguistic_anchor)
                    z = y

                _, mem2 = self.layerMe2(y[:, -1:, :], mem2)

                z    = self.normPost1(z)
                o    = z
                z, _ = self.layerPostA1(z, z, z, attn_mask=mask, rope=self.rope)
                z    = z + o
                z    = z + self.layerPost1(z)

                last_token_logits      = self.layerO1(z[:, -1, :])
                output[:, word, :]     = last_token_logits

                next_token = self.pick_word(last_token_logits)
                input_ids  = torch.cat([input_ids, next_token], dim=1)

            return output, None

def initmodel(device):
    model = biggerbrain(device).to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    return model

def think(prompt, model, max_length=100, iter=3):
    formatted = f"user: {prompt}\nassistant:"
    input_ids = torch.tensor([enc.encode(formatted, allowed_special={'<|endoftext|>'})]).to(model._orig_mod.device)
    output = []
    
    model._orig_mod.eval()
    with torch.no_grad():
        logits, _ = model._orig_mod.forward(input_ids, outlength=max_length, iter=iter, is_training=False)
        
        for i in range(logits.size(1)):
            probs = torch.softmax(logits[0, i], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == enc.eot_token:
                break
                
            output.append(next_token)
    
    print("Output:", enc.decode(output))