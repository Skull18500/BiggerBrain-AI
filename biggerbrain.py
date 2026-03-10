from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
import ai_extras as AI_ex
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, random_split
import numpy as np
import torch._dynamo
torch._dynamo.config.suppress_errors = False  # make errors visible

torch.set_float32_matmul_precision('high')

enc = tiktoken.get_encoding("gpt2")

class biggerbrain(nn.Module):
    def __init__(self, device):
        super(biggerbrain, self).__init__()
        self.dim = 768
        self.dim1 = 768
        self.ffndim = int(self.dim1 * 2)# for the first and last layers.
        self.heads = 8
        self.sequencelength = 256
        self.device = device
        
        self.debugprints = False
        
        self.embed = nn.Embedding(enc.max_token_value + 1, self.dim)
        self.layerO1 = nn.Linear(self.dim1, enc.max_token_value + 1, bias=False)#O1 is output1. This is the layer that produces the final output. It takes the processed data from the attention layer and produces a probability distribution over the vocabulary for the next word prediction.
        
        self.rope = AI_ex.RoPE(self.dim1 // self.heads, max_seq_len=self.sequencelength)
        
        self.layerMe1 = nn.GRU(self.dim1, self.dim1, batch_first=True)#memory GRU
        
        self.layerMA1 = nn.MultiheadAttention(self.dim1, self.heads, batch_first=True)#Memory Attention #1
        
        self.layerPreA1 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerPre1 = nn.Sequential(nn.Linear(self.dim1, self.ffndim * 2), AI_ex.SwiGLU(), nn.Linear(self.ffndim, self.dim1))
        self.normPre1 = nn.LayerNorm(self.dim1)
        
        self.layerPostA1 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerPost1 = nn.Sequential(nn.Linear(self.dim1, self.ffndim * 2), AI_ex.SwiGLU(), nn.Linear(self.ffndim, self.dim1))
        self.normPost1 = nn.LayerNorm(self.dim1)
        
        self.layerA1 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerA2 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerA3 = AI_ex.RoPEAttention(self.dim1, self.heads)
        self.layerA4 = AI_ex.RoPEAttention(self.dim1, self.heads)
        
        self.layerMi1 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        self.layerMi2 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        self.layerMi3 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        self.layerMi4 = nn.Sequential(nn.Linear(self.dim1, self.dim1 * 4 * 2), AI_ex.SwiGLU(), nn.Linear(self.dim1 * 4, self.dim1))
        
        self.norm0 = nn.LayerNorm(self.dim1)
        self.norm1 = nn.LayerNorm(self.dim1) # Layer normalization for stabilizing training. This is a common technique used in transformer models to improve convergence and performance.
        self.norm2 = nn.LayerNorm(self.dim1)
        self.norm3 = nn.LayerNorm(self.dim1)
        self.norm4 = nn.LayerNorm(self.dim1)
        
        self.MM1 = AI_ex.GatedResidual(self.dim1)#gated residual sigmoid thing. Memory Managment #1
        
        self.ponder_gate = nn.Linear(self.dim1, 1, bias=False)
        
        self._initialize_weights()
        
        self.layerO1.weight = self.embed.weight  # tied weights
        
        nn.init.normal_(self.ponder_gate.weight, mean=0.0, std=0.01)

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

    def trainingloop(self, data, epochs=100, lr=3e-4, batchsize=32, accumulation_steps=4):
        self.train()
        
        # Check if data is a dataset or pre-made batches
        if isinstance(data, list):
            # Pre-made batches
            batches = data
            use_loader = False
            num_batches = len(batches)
        else:
            # Dataset
            loader = DataLoader(
                data, 
                batch_size=batchsize, 
                shuffle=True,
                num_workers=4,        # CPU prepares next batch while GPU works
                pin_memory=True,      # Faster CPU -> GPU transfer
                drop_last=True        # Essential for torch.compile stability
            )
            use_loader = True
            num_batches = len(loader)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * num_batches)
        scaler = torch.amp.GradScaler('cuda')
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        best_loss = 1000.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            if use_loader:
                batch_iter = enumerate(loader)
            else:
                batch_iter = enumerate(batches)

            for i, (batch_inputs, batch_targets) in batch_iter:
                # non_blocking=True is the key to fixing the 40/60 CPU/GPU split
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    # Force fixed 'iter' for training to keep the GPU graph static
                    logits, iters = self(batch_inputs, iter=3, is_training=True)
                
                    lm_loss = criterion(logits.view(-1, enc.max_token_value + 1), batch_targets.reshape(-1))
                    ponder_loss = 0.001 * iters.mean()
                    loss = (lm_loss + ponder_loss) / accumulation_steps
                    epoch_loss += lm_loss.detach().item()
                    
                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # We don't call .item() inside the loop anymore. 
                
            current_loss = loss.detach().item()
            # End of Epoch Stats (The only time we sync with CPU)
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            avg_loss = epoch_loss / num_batches
            
            print(f"[{timestamp}] Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

            # Save Checkpoint
            if best_loss > current_loss:
                best_loss = current_loss
                torch.save(self.state_dict(), f"model_best.pt")


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
        total_iters = torch.zeros(batch_size, device=self.device)
    
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids)

        # ----------- TRAINING LANE ----------------
        if is_training:
            active_mask = torch.ones(batch_size, device=self.device)
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(self.device)

            # FIX #1: Initial attention and GRU are now inside this branch,
            # and use the causal mask so future tokens can't leak in.
            mem1 = torch.zeros(1, batch_size, self.dim1, device=self.device)
            y, _ = self.layerA1(x, x, x, attn_mask=mask, rope=self.rope)
            m1_out, mem1 = self.layerMe1(y, mem1)
            z = y
            
            y = self.normPre1(y)
            o = y
            y, _ = self.layerPreA1(y,y,y, mask=mask, rope=self.rope)
            y = y + o
            y = y + self.layerPre1(y)
            
            for j in range(iter):
                y_prev = y

                # FIX #2: Added the missing residual connection around layerA1.
                # Before, y was just replaced. Now the input is added back after attention.
                residual = y
                y = self.norm0(y)
                y, _ = self.layerA1(y, y, y, attn_mask=mask, rope=self.rope)
                y = y + residual  # <-- THE FIX

                y, _ = self.layerMA1(y, m1_out, m1_out, attn_mask=mask)
                m1_out, mem1 = self.layerMe1(y, mem1)

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

                total_iters += active_mask

                mask_3d = active_mask.bool().view(-1, 1, 1).expand_as(y)
                y = torch.where(mask_3d, y, y_prev)

                exit_prob = torch.sigmoid(self.ponder_gate(y[:, -1, :]))
                still_thinking = (exit_prob < 0.5).float().squeeze(-1)
                active_mask = active_mask * still_thinking

                z = y
            z = self.normPost1(z)
            o = z
            z, _ = self.layerPostA1(z,z,z, mask=mask, rope=self.rope)
            z = z + o
            z = z + self.layerPost1(z)
            
            return self.layerO1(z), total_iters

        # ------------- CHAT LANE ------------------
        else:
            if self.debugprints:
                iters = []
            output = torch.zeros(batch_size, outlength, enc.max_token_value + 1, device=self.device)
            mem1 = torch.zeros(1, batch_size, self.dim1, device=self.device)
            

            
            for word in range(outlength):
                current_input_ids = input_ids[:, -self.sequencelength:]
                curr_seq_len = current_input_ids.size(1)
                w = self.embed(current_input_ids)
                mask = torch.triu(torch.ones(curr_seq_len, curr_seq_len) * float('-inf'), diagonal=1).to(self.device)

                # Match training order exactly:
                y, _ = self.layerA1(w, w, w, attn_mask=mask, rope=self.rope)  # 1st
                m1_out, mem1 = self.layerMe1(y, mem1)                          # 2nd
                z = y

                y = self.normPre1(y)                                           # 3rd
                o = y
                y, _ = self.layerPreA1(y, y, y, attn_mask=mask, rope=self.rope)
                y = y + o
                y = y + self.layerPre1(y)

                currentiter = 0

                for j in range(iter):
                    if self.debugprints:
                        currentiter += 1

                    # FIX #2: Added the missing residual connection (matches training).
                    residual = y
                    y = self.norm0(y)
                    y, _ = self.layerA1(y, y, y, attn_mask=mask, rope=self.rope)
                    y = y + residual

                    y, _ = self.layerMA1(y, m1_out, m1_out, attn_mask=mask)
                    m1_out, mem1 = self.layerMe1(y, mem1)

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
                    z = y

                    exit_prob = torch.sigmoid(self.ponder_gate(y[:, -1, :]))
                    if exit_prob >= 0.5:
                        break
                
                #post block
                z = self.normPost1(z)
                o = z
                z, _ = self.layerPostA1(z,z,z, mask=mask, rope=self.rope)
                z = z + o
                z = z + self.layerPost1(z)
                
                last_token_logits = self.layerO1(z[:, -1, :])
                output[:, word, :] = last_token_logits

                next_token = self.pick_word(last_token_logits)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if self.debugprints:
                    iters.append(currentiter)

            if self.debugprints:
                print(iters)
            return output, None
                #enc.n_vocab

def initmodel(device):
    model = biggerbrain(device).to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
    
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