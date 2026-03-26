from datetime import datetime
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import torch
import torch.nn as nn
import torch.optim as optim
import random
import tiktoken
import ai_extras as AI_ex
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, random_split
import numpy as np
import torch._dynamo
import bitsandbytes as bnb
import torch.utils.checkpoint as cp

torch._dynamo.config.suppress_errors = False  # make errors visible
torch._inductor.config.triton.cudagraphs = True

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_float32_matmul_precision('medium')

enc = tiktoken.get_encoding("gpt2")

class biggerbrain(nn.Module):
    def __init__(self, device):
        super(biggerbrain, self).__init__()
        self.dim = 768
        self.dim1 = 768
        self.ffndim = int(self.dim1 * 2)
        self.heads = 8
        self.sequencelength = 512
        self.device = device
        
        self.debugprints = False
        
        self.embed = nn.Embedding(enc.max_token_value + 1, self.dim)
        self.layerO1 = nn.Linear(self.dim1, enc.max_token_value + 1, bias=False)#O1 is output1. This is the layer that produces the final output. It takes the processed data from the attention layer and produces a probability distribution over the vocabulary for the next word prediction.
        
        self.rope = AI_ex.RoPE(self.dim1 // self.heads, max_seq_len=self.sequencelength)
        
        self.layerMe1 = AI_ex.FastMemoryCell(self.dim1)#scratchpad
        self.layerMe2 = nn.GRU(self.dim1, self.dim1, batch_first=True)#long term memory
        
        self.layerMA1 = AI_ex.FlashCrossAttention(self.dim1, self.heads)#Memory Attention #1
        self.layerMA2 = AI_ex.FlashCrossAttention(self.dim1, self.heads)
        
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
        
        self.layerMi1 = AI_ex.MoLLayer(self.dim1, self.dim1 * 4, n_experts=2)
        self.layerMi2 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        self.layerMi3 = AI_ex.MoLLayer(self.dim1, self.dim1 * 4, n_experts=2)
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

    def pick_word(self, output, k=50, temperature=0.8,
              prev_tokens=None, rep_penalty=1.5):
        logits = output / (temperature + 1e-8)
    
        # Permanently blacklist chat-format artifact tokens
        # These have high probability from training data contamination
        # but should never appear in normal prose output
        BAD_TOKENS = [
            25,    # ":"  single colon
            3712,     # "::" double colon  
            1058,  # ":" alternate encoding
        ]
        for tok in BAD_TOKENS:
            if tok < logits.size(-1):
                logits[0, tok] = float('-inf')
    
        # Repetition penalty
        if prev_tokens is not None:
            for tok in set(prev_tokens[-64:]):
                logits[0, tok] /= rep_penalty
    
        # Top-K filtering
        v, _ = torch.topk(logits, min(k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    
        # Sample
        probabilities = torch.softmax(logits, dim=-1)
        word_id = torch.multinomial(probabilities, num_samples=1)
    
        return word_id

    def trainingloop(self, data, epochs=50, lr=3e-4, batchsize=32,
                 accumulation_steps=4, max_batches=None, 
                 subset_fraction=1.0):   # ← add this parameter
        self.train()
        batchloss = 1000
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
        optimizer = bnb.optim.AdamW8bit([
            {
            'params': [p for n, p in self.named_parameters() 
                    if 'embed' not in n and 'layerO1' not in n],
            'weight_decay': 0.05
            },
            {
            'params': [p for n, p in self.named_parameters() 
                   if 'embed' in n or 'layerO1' in n],
            'weight_decay': 0.0
            }
        ],
        lr=lr,
        betas=(0.9, 0.95)
        # no weight_decay here — it's set per group above
        )
        
        batches_per_epoch = subset_size // batchsize
        total_updates_per_epoch = batches_per_epoch // accumulation_steps
        T_max = total_updates_per_epoch * epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = T_max,
            eta_min = 1e-5       # floor — never goes below this
        )

        best_loss   = 1000.0
        
        #self.forward_training = torch.compile(self.forward_training, backend ='eager')#, options=['shape_padding':True] model
        if os.path.exists("checkpoint_full.pth"):
            checkpoint = torch.load("checkpoint_full.pth", weights_only=False, map_location='cpu')
            self.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            #all_indices = checkpoint['all_indices']
            batchloss = checkpoint['batchloss']
            print(f"Loaded checkpoint for training. {scheduler.get_last_lr()}")
            
            
        for epoch in range(epochs):
            
            #torch.cuda.empty_cache()  # ← add this line
            #torch.cuda.reset_peak_memory_stats()
            
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
                if i % 500 == 0 and i > 0:
                    std = self.embed.weight.std().item()
                    print(f"  Embed std: {std:.4f}  (healthy = ~0.02, bad = <0.005)")
                    if std < 0.005:
                        print("  WARNING: embedding collapse detected!")



                batch_inputs  = batch_inputs.to(self.device,  non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    num_iter  = random.randint(1, 3)
                    logits, _ = self.forward_training(batch_inputs, 
                                      max_iters=3, 
                                      active_iters=num_iter)
                    lm_loss = criterion(
                        logits.view(-1, enc.max_token_value + 1),
                        batch_targets.reshape(-1)
                    )
    
                    # MoL balance loss — prevents expert collapse
                    balance_loss = torch.tensor(0.0, device=self.device)
                    for name, module in self.named_modules():
                        if isinstance(module, AI_ex.MoLLayer):
                            if hasattr(module, 'balance_loss') and module.balance_loss is not None:
                                balance_loss = balance_loss + module.balance_loss
                    loss = (lm_loss + 0.2 * balance_loss) / accumulation_steps

                loss.backward()


                if (i) % accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if self.debugprints:
                        (f"finished batch: {i}")
                    if batchloss > lm_loss.detach().item():
                        batchloss = lm_loss.detach().item()
                        torch.save(self.state_dict(), "model_best.pth")
                        print(f"saved best batch weights loss: {batchloss:.4f}")
                if (i + 1) % 20 == 0:
                    print(f"Epoch {epoch} | Batch {i} | loss={lm_loss.detach().item():.4f}")
                    # Save full training state:
                    torch.save({
                    'model': self.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'chunk_idx': chunk_idx,
                    #'all_indices': all_indices,  # save the shuffle order
                    'batchloss': batchloss,
                    }, "checkpoint_full.pth")
                    print(f"Saved state.")
                    
            avg_loss  = (epoch_loss / batches_run)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Chunk: {chunk_idx+1}/{len(chunks)}")
            
    
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

    def forward_training(self, input_ids, active_iters=3, max_iters=3):
        batch_size, seq_len = input_ids.size()
        x = self.embed(input_ids)
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

        # ── THINKING LOOP ────────────────────────────────────────────
        def run_thinking_block(y, z, mem1, m1_out, y_prev, iter_tensor):
            iter_idx = iter_tensor.item()  # ← use the parameter, not a ghost variable

            residual = y
            y        = self.norm0(y)
            y, _     = self.layerA1(y, y, y, attn_mask=mask, rope=self.rope)
            y        = y + residual

            y, _ = self.layerMA1(y, m1_out,      m1_out,      attn_mask=mask)
            y, _ = self.layerMA2(y, seq_context, seq_context, attn_mask=mask)

            last         = y[:, -1:, :]
            new_state, _ = self.layerMe1(last.squeeze(1), mem1)
            write_gate   = torch.sigmoid(self.scratchpad_gate(last.squeeze(1)))
            mem1         = write_gate * new_state + (1.0 - write_gate) * mem1
            m1_out       = mem1.unsqueeze(1).expand(-1, y.size(1), -1)

            y = self.norm1(y)
            y = y + self.layerMi1(y, y_prev, linguistic_anchor, iter_idx)

            y = self.norm2(y);  o = y
            y, _ = self.layerA2(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o;          y = y + self.layerMi2(y)

            y = self.norm3(y);  o = y
            y, _ = self.layerA3(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o
            y = y + self.layerMi3(y, y_prev, linguistic_anchor, iter_idx)

            y = self.norm4(y);  o = y
            y, _ = self.layerA4(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o;          y = y + self.layerMi4(y)

            y = self.MM1(y, z)
            y = self.MM2(y, linguistic_anchor)
            return y, mem1, m1_out

        for j in range(active_iters):            # ← the missing loop
            y_prev      = y.detach()             # ← snapshot before this iter
            iter_tensor = torch.tensor(j, device=self.device)

            y, mem1, m1_out = cp.checkpoint(
                run_thinking_block,
                y, z, mem1, m1_out, y_prev, iter_tensor,  # ← y_prev added, comma fixed
                use_reentrant=False
            )
            z = y
        # ─────────────────────────────────────────────────────────────

        _, mem2 = self.layerMe2(y[:, -1:, :], mem2)

        z    = self.normPost1(z)
        o    = z
        z, _ = self.layerPostA1(z, z, z, attn_mask=mask, rope=self.rope)
        z    = z + o
        z    = z + self.layerPost1(z)

        return self.layerO1(z), torch.tensor(float(active_iters))
    
    def forward_chat(self, input_ids, outlength=1, iter=3, top_k=10, temperature=0.8):
        """
        Dynamic loop for chat inference. Does not compute gradients.
        Allows for flexible thought-depth on the fly.
        """
        batch_size, _ = input_ids.size()
        states = [] # for the iter states.
        generated_tokens = [] # Store the actual token IDs here
        prev_tokens = input_ids[0].tolist()
        mem1 = torch.zeros(batch_size, self.dim1, device=self.device)
        mem2 = torch.zeros(1, batch_size, self.dim1, device=self.device)

        for word in range(outlength):
            # Crop to max sequence length to prevent out-of-bounds positional issues
            current_input_ids = input_ids[:, -self.sequencelength:]
            curr_seq_len = current_input_ids.size(1)
            
            w = self.embed(current_input_ids)
            mask = torch.triu(
                torch.ones(curr_seq_len, curr_seq_len, device=self.device) * float('-inf'),
                diagonal=1
            )

            y, _ = self.layerA1(w, w, w, attn_mask=mask, rope=self.rope)

            last = y[:, -1:, :]
            new_state, _ = self.layerMe1(last.squeeze(1), mem1)
            write_gate = torch.sigmoid(self.scratchpad_gate(last.squeeze(1)))
            mem1 = write_gate * new_state + (1.0 - write_gate) * mem1
            m1_out = mem1.unsqueeze(1).expand(-1, curr_seq_len, -1)

            z = y

            y = self.normPre1(y)
            o = y
            y, _ = self.layerPreA1(y, y, y, attn_mask=mask, rope=self.rope)
            y = y + o
            y = y + self.layerPre1(y)

            linguistic_anchor = y
            seq_context, _ = self.layerMe2(y, mem2)

            states.append(y.detach().cpu()) 
            
            for j in range(iter):
                # --- FIX: Track the previous state and iteration for the MoL layers ---
                y_prev = y.detach()
                iter_tensor = torch.tensor(j, device=self.device)

                residual = y
                y = self.norm0(y)
                y, _ = self.layerA1(y, y, y, attn_mask=mask, rope=self.rope)
                y = y + residual

                y, _ = self.layerMA1(y, m1_out, m1_out, attn_mask=mask)
                y, _ = self.layerMA2(y, seq_context, seq_context, attn_mask=mask)

                last = y[:, -1:, :]
                new_state, _ = self.layerMe1(last.squeeze(1), mem1)
                write_gate = torch.sigmoid(self.scratchpad_gate(last.squeeze(1)))
                mem1 = write_gate * new_state + (1.0 - write_gate) * mem1
                m1_out = mem1.unsqueeze(1).expand(-1, curr_seq_len, -1)

                # Block 1
                y = self.norm1(y)
                # --- FIX: Pass the missing arguments to layerMi1 ---
                y = y + self.layerMi1(y, y_prev, linguistic_anchor, iter_tensor) 

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
                # --- FIX: Pass the missing arguments to layerMi3 ---
                y = y + self.layerMi3(y, y_prev, linguistic_anchor, iter_tensor) 

                # Block 4
                y = self.norm4(y)
                o = y
                y, _ = self.layerA4(y, y, y, attn_mask=mask, rope=self.rope)
                y = y + o
                y = y + self.layerMi4(y)

                y = self.MM1(y, z)
                y = self.MM2(y, linguistic_anchor)
                z = y
                states.append(y.detach().cpu()) 

            _, mem2 = self.layerMe2(y[:, -1:, :], mem2)

            z = self.normPost1(z)
            o = z
            z, _ = self.layerPostA1(z, z, z, attn_mask=mask, rope=self.rope)
            z = z + o
            z = z + self.layerPost1(z)
            states.append(z.detach().cpu())
            
            # --- FIX: Only look at the logits for the very last token ---
            last_token_logits = self.layerO1(z[:, -1, :])
            
            # --- FIX: Pick the word HERE, once. ---
            next_token = self.pick_word(
                last_token_logits,
                k=top_k,
                temperature=temperature,
                prev_tokens=prev_tokens,
                rep_penalty=3.0
            )
            prev_tokens.append(next_token.item())
            
            # Stop early if the model generates the end-of-text token
            if next_token.item() == enc.eot_token:
                break
                
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # Return the clean list of generated tokens, not the logits
        return generated_tokens, states

def initmodel(device):
    model = biggerbrain(device).to(device)
    #model = torch.compile(model, fullgraph=False, mode='eager')
    
    return model

def think(prompt, model, max_length=100, iter=3, top_k=25, temperature=0.8, raw=False):
    
    if raw:
        formatted = prompt
    else:
        formatted = f"user: {prompt}\nassistant:"
        
    input_ids = torch.tensor([enc.encode(formatted, allowed_special={'<|endoftext|>'})]).to(model.device)
    
    model.eval()
    with torch.no_grad():
        # Pass the sampling parameters directly to forward_chat
        generated_tokens, _ = model.forward_chat(
            input_ids, 
            outlength=max_length, 
            iter=iter, 
            top_k=top_k, 
            temperature=temperature
        )
        
        
    print("Output:", enc.decode(generated_tokens))