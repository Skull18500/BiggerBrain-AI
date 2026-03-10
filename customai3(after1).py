#from csv import writer
from dataclasses import dataclass
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tokenizer import numbers, id_to_word, tokenize, clean_input, trim_history, load_file
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast#use autocast for mixed precision.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

writer = SummaryWriter('runs/skullai')

pretrainfile = "c:/Users/chand/OneDrive/Documents/pytorchplayground/AI/pretraining.txt"
trainingfile = "c:/Users/chand/OneDrive/Documents/pytorchplayground/AI/training.txt"
weightsfile = "c:/Users/chand/OneDrive/Documents/pytorchplayground/AI/model_best.pth"
weightfile = "c:/Users/chand/OneDrive/Documents/pytorchplayground/AI/model.pth"

print(torch.cuda.is_available())
print(torch.__version__)

device = torch.device('cpu')#other code: 'cuda' if torch.cuda.is_available() else , but cuda is slower.
print(f'Using device: {device}')

class bigbrain(nn.Module):
    
    def __init__(self):
        super().__init__()
        global numbers
        self.dim = 16
        self.dim1 = 48
        
        self.layerI1 = nn.Linear(self.dim, self.dim1)
        self.layerM1 = nn.Linear(self.dim1 + self.dim + self.dim, self.dim1)
        self.layerM2 = nn.Linear(self.dim1 + self.dim * 3 + self.dim, self.dim1)
        self.layerM3 = nn.Linear(self.dim1, self.dim1)
        self.layerN1 = nn.LayerNorm(self.dim1)# normilize the values to prevent them from getting too high or too low, which can cause problems with learning and stability. This is especially important in a model like this where the context and thoughts can have a big impact on the output.
        self.layerO1 = nn.Linear(self.dim1, self.dim)
        
        self.layerT1 = nn.MultiheadAttention(self.dim, 4)#thoughts layer.
        self.layerC1 = nn.GRU(self.dim, self.dim)#context layer. Uses GRU as a brain cell for writing data to context.
        self.layerA1 = nn.Linear(self.dim1, self.dim)#attention projection?
        self.layerE1 = nn.Linear(self.dim1, 1)# this is for the decision of when to End the thinking process. Key to the thinking.
        
        self.embed = nn.Embedding(len(numbers), self.dim)
        self.thoughts = []
        self.maxthoughts = 20
        self.context = torch.zeros(self.dim).to(device)
        self.context_words = []
        
        nn.init.zeros_(self.layerE1.weight)#make it output 0.5
        nn.init.zeros_(self.layerE1.bias)#make it output 0.5
        
        
    def think_train(self, batch, context1):
        # batch: [batch_size, seq_len] word IDs
        # context1: [batch_size, 16]
        batch_size, seq_len = batch.shape
    
        # Get embeddings for whole batch at once
        embeddings = self.embed(batch)  # [batch_size, seq_len, 16]
    
        # Self attention over whole sentence
        embeddings_t = embeddings.transpose(0, 1)  # [seq_len, batch_size, 16]
        attended, _ = self.layerT1(embeddings_t, embeddings_t, embeddings_t)
        attended = attended.transpose(0, 1)  # [batch_size, seq_len, 16]
    
        # Update context word by word using GRU
        for t in range(seq_len):
            context1 = self.layerC1(embeddings[:, t, :], context1)  # [batch_size, 16]
    
        # Return attended embeddings as thoughts, final context
        return attended, context1  # [batch_size, seq_len, 16], [batch_size, 16]

    def forward_step(self, word_embed, context1, thoughts1):
        # word_embed: [batch_size, 16]
        # context1:   [batch_size, 16]
        # thoughts1:  [batch_size, seq_len, 16]
        batch_size = word_embed.shape[0]
    
        out = self.layerI1(word_embed)  # [batch_size, 48]
    
        # Project out to dim for attention scoring
        out_proj = self.layerA1(out)  # [batch_size, 16]
    
        # M1 — current state + context + attention projection
        combined_m1 = torch.cat([out, context1, out_proj], dim=1)  # [batch_size, 80]
        out = self.layerM1(combined_m1)  # [batch_size, 48]
        out = self.layerN1(out)
        out = self.calc(out)
    
        # Score all thoughts against current state
        scores = torch.bmm(thoughts1, out_proj.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
    
        # Get top 3 thoughts
        k = min(3, thoughts1.shape[1])
        top_idx = torch.topk(scores, k, dim=1).indices  # [batch_size, k]
        top = thoughts1[torch.arange(batch_size).unsqueeze(1), top_idx]  # [batch_size, k, 16]
    
        # Pad to exactly 3 if needed
        if k < 3:
            padding = torch.zeros(batch_size, 3 - k, self.dim).to(device)
            top = torch.cat([top, padding], dim=1)
    
        top_flat = top.reshape(batch_size, self.dim * 3)  # [batch_size, 48]
    
        # M2 — state + thoughts + context
        combined_m2 = torch.cat([out, top_flat, context1], dim=1)  # [batch_size, 112]
        out = self.layerM2(combined_m2)  # [batch_size, 48]
        out = self.layerN1(out)
        out = self.calc(out)
    
        # M3
        out = self.layerM3(out)  # [batch_size, 48]
        out = self.layerN1(out)
        out = self.calc(out)
    
        out = self.layerO1(out)  # [batch_size, 16]
    
        logits = torch.matmul(out, self.embed.weight.T)  # [batch_size, vocab_size]
        return logits

    def make_batches(self, sentences, batch_size=8):
        batches = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            id_batch = [[numbers[w] for w in s if w in numbers] for s in batch]
            id_batch = [s for s in id_batch if len(s) >= 2]
            if len(id_batch) == 0:
                continue
            max_len = max(len(s) for s in id_batch)
            padded = [s + [0] * (max_len - len(s)) for s in id_batch]
            batches.append(torch.tensor(padded).to(device))
        return batches

    def training_loop(self, sentences, epochs=100, lr=0.001, is_second_phase=False):
        global id_to_word, numbers
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding tokens
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        best_loss = float('inf')
        for epoch in range(epochs):
            total_epoch_loss = 0
            random.shuffle(sentences)
            batches = self.make_batches(sentences, batch_size=8)
        
            for batch in batches:
                    batch_size, seq_len = batch.shape
                    if seq_len < 2:
                        continue
            
                    optimizer.zero_grad()
            
                    # Local context, never persists during training
                    context1 = torch.zeros(batch_size, self.dim).to(device)
            
                    # Think over whole batch at once
                    thoughts1, context1 = self.think_train(batch, context1)
            
                    all_logits = []
                    all_targets = []
            
                    for t in range(seq_len - 1):
                        word_embed = self.embed(batch[:, t])  # [batch_size, 16]
                        target = batch[:, t + 1]              # [batch_size]
                
                        # Update context step by step with real words (teacher forcing!)
                        context1 = self.layerC1(word_embed, context1)
                
                        logits = self.forward_step(word_embed, context1, thoughts1)
                        all_logits.append(logits)
                        all_targets.append(target)
            
                    # Single loss for entire batch and sequence
                    logits_tensor = torch.stack(all_logits).view(-1, len(numbers))
                    targets_tensor = torch.stack(all_targets).view(-1)
            
                    loss = criterion(logits_tensor, targets_tensor)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_epoch_loss += loss.item()
        
            writer.add_scalar('Loss/train', total_epoch_loss,
                            epoch if not is_second_phase else epoch + 500)
        
            if total_epoch_loss < best_loss:
                    best_loss = total_epoch_loss
                    torch.save(self.state_dict(), weightsfile)
        
            if epoch % 5 == 0:
                    print(f"Epoch {epoch} | Loss: {total_epoch_loss:.4f}")
            scheduler.step()
    
        torch.save(self.state_dict(), weightfile)
    
    def calc(self, x):#this was doint relu until i realised that that squashed numbers. Possibly a better formula could be found, but for now this is what im using.
        y = torch.nn.functional.silu(x) # SiLU (Sigmoid Linear Unit) is a smooth, non-monotonic activation function that can help with gradient flow and learning complex patterns. It allows for small negative values, which can be beneficial for certain types of data and tasks, while still providing non-linearity to the model.
        return y
        
    #make this better!
    def think(self, x, is_training=True):
        global numbers, device
        for i in clean_input(x).split(" "):
            if i not in numbers:
                continue
            
            y = self.embed(torch.tensor(numbers[i], device=device)).view(-1)#this makes gets the embedding for the word.
            self.context_words.append(y)#we add to the context words, which are used for attention in the forward pass

            words = torch.stack(self.context_words)  # [num_words_so_far, 16]#we smoosh all context words together into a matrix for the attention layer.
            words = words.unsqueeze(1)               # [num_words, 1, 16] — batch_size=1
            attended, _ = self.layerT1(words, words, words)  #we use attention on the context words to generate a new thought, and add that thought to the list of thoughts. This allows the model to generate thoughts that are relevant to the current context, which can help it make better predictions.
            z = attended[-1, 0, :]  # [16] — just the last word's attended vector
            
            relevance = torch.nn.functional.cosine_similarity(self.context.unsqueeze(0), z.unsqueeze(0)) >= 0.5  # arbitrary threshold for "relevance", mesures the closeness of thought to the current context. This mesures the correctnes of the thought so it wont think random stuff that has nothing to do with the current context.
            if relevance:
                self.thoughts.append(z)
                if len(self.thoughts) > self.maxthoughts:
                    #make this sort by cosine similarity on the current word/context and then only pop the least relevent one.
                    similarities = [torch.nn.functional.cosine_similarity(self.context.unsqueeze(0), t.unsqueeze(0)) for t in self.thoughts]
                    min_sim = torch.argmin(torch.stack(similarities))
                    self.thoughts.pop(min_sim)
            self.calc_context(y)
        if not is_training:#this made my terminal explode with too much output, so i made it only print outside training.
            print("Thoughts:", self.thoughts)#im not printing words because the vectors wont always be word, they will encompass vibes and feelings and stuff, so it would be hard to interpret.
        
    def calc_context(self, x):
        # I add .unsqueeze(0) to make them [1, 12] for the GRU logic. GRU might not be the best choice, but i got no clue.
        new_context = self.layerC1(x.unsqueeze(0), self.context.unsqueeze(0)).squeeze(0)
    
        # I squeeze(0) to bring it back to a flat (12) vector
        self.context = new_context.squeeze(0)#.detach()
        
    #ideally this wouldnt be necisary, and it would just run through one final layer that outputs the word.
    def pickword(self, x, training):
        global id_to_word
        logits = torch.matmul(x, self.embed.weight.T)
        if training:
            return logits
        else:
        # Safety checks
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            probs = torch.softmax(logits, -1)
            probs = probs + 1e-8  # prevent exact zeros
            probs = probs / probs.sum()  # renormalize
            word_id = torch.multinomial(probs, num_samples=1).item()
        return id_to_word[word_id]
        
    def forward(self, x, training=True, loadfromfile=False, maxwords=50):
        last_words = []
        global numbers, device, id_to_word
        out = torch.zeros(self.dim).to(device)
        x1 = torch.zeros(self.dim).to(device)
        output = ""
        logits = None
        should_continue = True
        wordfinished = False
        self.think(x, training)#think!
        x1 = self.context.clone()
        current_iter = 0    
        self.context_words = []#clear context words after using them for the first prediction, so that it can focus on the new words it generates and not just repeat the same ones over and over.
        while should_continue == True:
            if current_iter >= maxwords:
                should_continue = False
            current_iter += 1
            out = self.layerI1(x1)# I + 1 means Input #1
            
            if len(self.context_words) > 0:
                context_matrix1 = (torch.stack(self.context_words))
                out_projected = self.layerA1(out)
                scores1 = torch.softmax(torch.matmul(context_matrix1, out_projected), dim=0)#i should really just make this a projection layer.
                input_attention1 = torch.matmul(scores1, context_matrix1)#i should really just make this a projection layer.
            else:
                input_attention1 = torch.zeros(self.dim).to(device)
            
            while wordfinished == False:
                out = self.layerM1(torch.cat([out, self.context, input_attention1])) # M + 1 means middle #1
                out = self.layerN1(out)
            
                out = self.calc(out)
                if len(self.thoughts) >= 3:
                    thought_tensor = torch.stack(self.thoughts) # [N, 12]
                    # Compare 'out' to all thoughts at once
                    scores = torch.nn.functional.cosine_similarity(out[:self.dim].unsqueeze(0), thought_tensor)
                    top3_idx = torch.topk(scores, min(3, len(self.thoughts))).indices
                    top3 = torch.cat([self.thoughts[i] for i in top3_idx])
                    combined = torch.cat([out, top3, self.context])
                    out = self.layerM2(combined)
                elif len(self.thoughts) > 0:
                    # pad with zeros if less than 3 thoughts
                    padding = torch.zeros(self.dim * (3 - len(self.thoughts))).to(device)
                    top_thoughts = torch.cat(self.thoughts + [padding])
                    combined = torch.cat([out, top_thoughts, self.context])
                    out = self.layerM2(combined)
                else:
                    out = self.layerM2(torch.cat([out, torch.zeros(self.dim * 3).to(device), self.context]))  # no thoughts, just pad with zeros
                out = self.layerN1(out) # normilize before next layer
            
                out = self.calc(out)
                out = self.layerM3(out) # M + 3 means middle #3
                out = self.layerN1(out)
                out = self.calc(out)
                if torch.sigmoid(self.layerE1(out)).item() > 0.5:
                    wordfinished = True
            out = self.layerO1(out)# O + 1 means output #1
            x1 = out
            
            logits = self.pickword(out, training)
            last_words.append(logits)
            if len(last_words) >= 4 and last_words[-1] == last_words[-3] and last_words[-2] == last_words[-4]:
                should_continue = False
            if training:
                    return logits#this isnt working due to some error with picking the word.
            else:
                output += " "
                #reducing variable count
                output += logits
                out = numbers[logits]
                if logits == "<end>":
                    should_continue = False
                    
            self.calc_context(out)
            
        if not training:
            return output
        elif logits is not None:
            return logits

text1 = []

model = bigbrain()
model = model.to(device)
#model = torch.compile(model, mode="reduce-overhead")

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total}")

pretrain_data = load_file(pretrainfile)
training_data = load_file(trainingfile)

old_weights = model.embed.weight.data
new_embed = nn.Embedding(len(numbers), model.dim).to(device)
new_embed.weight.data[:old_weights.shape[0]] = old_weights
model.embed = new_embed

model.training_loop(pretrain_data, 500, 0.0001)
model.load_state_dict(torch.load(weightsfile, weights_only=True), strict=False)
model.training_loop(training_data, 500, 0.00008, True)

model.load_state_dict(torch.load(weightsfile, weights_only=True), strict=False)#probably not necessary but just to be safe, load the best weights from training before starting the chat loop.

history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    history = trim_history(history, 400) + f" [user] {user_input} [ai]"
    model.context_words = []  # clear per-turn but keep thoughts/context
    response = model(clean_input(history), False)
    history += response
    print(f"SkullAI: {response}")