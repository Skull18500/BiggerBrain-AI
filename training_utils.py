import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
from datasets import load_dataset
from itertools import islice
import random
import os


enc = tiktoken.get_encoding("gpt2")



def make_batches(text, chunk_size=32, batch_size=8):
    tokens = enc.encode(text)
    batches = []
    documents = text.split("\n\n")
    
    # Encode each document with EOT separator
    all_tokens = []
    for doc in documents:
        if doc.strip():
            all_tokens += enc.encode(doc.strip()) + [enc.eot_token]
            
    input_ids  = all_tokens[:-1]
    target_ids = all_tokens[1:]
    currentbatch = []

    for i in range(0, len(input_ids), chunk_size):
        
        input_batch = input_ids[i:i+chunk_size]
        target_batch = target_ids[i:i+chunk_size]
        if len(target_batch) < chunk_size:
            
            target_batch = target_batch + [enc.eot_token] * (chunk_size - len(target_batch))
            
        if len(input_batch) < chunk_size:
            
            input_batch = input_batch + [enc.eot_token] * (chunk_size - len(input_batch))
            
        
        # always append first
        currentbatch.append((torch.tensor(input_batch), torch.tensor(target_batch)))
        
        # then flush when full
        if len(currentbatch) == batch_size:
            input_tensor = torch.stack([c[0] for c in currentbatch])
            target_tensor = torch.stack([c[1] for c in currentbatch])
            batches.append((input_tensor, target_tensor))
            currentbatch = []

    if len(currentbatch) > 0:
        input_tensor = torch.stack([c[0] for c in currentbatch])
        target_tensor = torch.stack([c[1] for c in currentbatch])
        batches.append((input_tensor, target_tensor))

    
    print(f"Number of batches: {len(batches)}")
    print(f"Batch shape: {batches[0][0].shape}")
    return batches

    
def make_batches_streaming(filename, chunk_size=256, batch_size=256, num_streams=4, max_batches=1000):
    global enc
    file_size = os.path.getsize(filename)
    all_chunks = []
    buffers = [[] for _ in range(num_streams)]
    files = []

    # 1. Open multiple 'viewpoints' into the file
    for _ in range(num_streams):
        f = open(filename, "r", encoding="utf-8", errors="ignore")
        
        # We stay within the first 75% to ensure there's enough text to read
        max_start = int(file_size * 0.75)
        if max_start > 0:
            f.seek(random.randint(0, max_start))
            f.readline() # Align to the next clean line
        
        files.append(f)

    # 2. Fill the chunks
    read_size = 1024 * 32 # 32KB chunks
    while len(all_chunks) < max_batches * batch_size:
        any_text_read = False
        for i in range(num_streams):
            raw_data = files[i].read(read_size)
        
            # If EOF, wrap around
            if not raw_data:
                files[i].seek(0)
                raw_data = files[i].read(read_size)
        
            # Explicitly check that we have a non-empty string
            if raw_data and isinstance(raw_data, str):
                any_text_read = True
                new_tokens = enc.encode(raw_data)
                buffers[i].extend(new_tokens)
            
                # Slice the buffer into chunks
                while len(buffers[i]) >= chunk_size + 1:
                    all_chunks.append(buffers[i][:chunk_size + 1])
                    buffers[i] = buffers[i][chunk_size + 1:]
        
        if not any_text_read: break # Safety break

    for f in files: f.close()

    # 3. Final Shuffling and Tensor Creation
    random.shuffle(all_chunks)
    batches = []
    for i in range(0, len(all_chunks) - batch_size, batch_size):
        batch = all_chunks[i:i + batch_size]
        x = torch.tensor([c[:-1] for c in batch])
        y = torch.tensor([c[1:] for c in batch])
        batches.append((x, y))

    return batches

def build_dataset(output_file="pretraining.txt"):
    
    with open(output_file, "a", encoding="utf-8") as f:
        
        print("Adding Wikipedia...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for example in ds:
            if example["text"].strip():
                f.write(example["text"] + "\n\n")
        print("Wikipedia done!")

        print("Adding GSM8K...")
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for example in ds:
            f.write(example["question"] + "\n" + example["answer"] + "\n\n")
        print("GSM8K done!")

        print("Adding SmolTalk...")
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
        for example in ds:
            for message in example["messages"]:
                f.write(f"{message['role']}: {message['content']}\n")
            f.write("\n")
        print("SmolTalk done!")

        print("Adding Orca (500k limit)...")
        ds = load_dataset("Open-Orca/OpenOrca", split="train")
        for example in islice(ds, 500000):
            f.write(f"system: {example['system_prompt']}\n")
            f.write(f"user: {example['question']}\n")
            f.write(f"assistant: {example['response']}\n\n")
        print("Orca done!")

    print("Dataset complete!")
    import os
    size = os.path.getsize(output_file)
    print(f"Final file size: {size / 1024 / 1024:.1f} MB")
    
    
def make_batches_simple(filename, enc, chunk_size=256, batch_size=256, max_batches=1000):
    # Load the whole file or stream it? 
    # For a few MBs, loading to memory is fastest.
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Encode the entire corpus once
    tokens = enc.encode(text)
    
    all_chunks = []
    # Slide through the tokens to create training examples
    for i in range(0, len(tokens) - (chunk_size + 1), chunk_size + 1):
        all_chunks.append(tokens[i : i + chunk_size + 1])
    
    print(f"Total chunks available: {len(all_chunks)}")
    random.shuffle(all_chunks)
    
    # Limit to max_batches if requested
    num_to_take = min(len(all_chunks), max_batches * batch_size)
    all_chunks = all_chunks[:num_to_take]
    
    batches = []
    for i in range(0, len(all_chunks) - batch_size, batch_size):
        batch = all_chunks[i : i + batch_size]
        x = torch.tensor([c[:-1] for c in batch])
        y = torch.tensor([c[1:] for c in batch])
        batches.append((x, y))
        
    return batches


def build_ultimate_dataset(output_path, total_samples=5000):
    """
    Downloads, mixes, and saves a high-quality curriculum.
    Ratio: 40% Logic (GSM8K) / 60% Conversation (SmolTalk/Instruct)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Download GSM8K (The Logic)
    print("Downloading GSM8K...")
    gsm_data = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
    
    # 2. Download SmolTalk or similar (The Conversation)
    # Note: 'lucasmccabe-lmi/CodeAlpaca-20k' or 'HuggingFaceH4/no_robots' are great
    print("Downloading SmolTalk-style instructions...")
    instruct_data = load_dataset("HuggingFaceH4/no_robots", split="train")

    num_gsm = int(total_samples * 0.4)
    num_instruct = int(total_samples * 0.6)

    final_text = []

    # Process GSM8K (Logic)
    for i in range(min(num_gsm, len(gsm_data))):
        q = gsm_data[i]['question']
        a = gsm_data[i]['answer']
        final_text.append(f"User: {q}\nAssistant: {a}\n---\n")

    # Process Instruction Data (English Grammar/Conversation)
    for i in range(min(num_instruct, len(instruct_data))):
        # no_robots uses a 'messages' list or 'prompt'/'messages'
        prompt = instruct_data[i]['messages'][0]['content']
        response = instruct_data[i]['messages'][1]['content']
        final_text.append(f"User: {prompt}\nAssistant: {response}\n---\n")

    # Shuffle so the model doesn't get 2000 math problems in a row
    random.shuffle(final_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(final_text)
    
    print(f"🔥 Success! Created {len(final_text)} samples at {output_path}")

def add_wiki_knowledge(output_path, num_articles=10000, chunk_size=600):
    """
    Pulls articles and breaks them into smaller chunks to maximize data volume.
    """
    print(f"📚 Gathering a massive Wiki dataset ({num_articles} articles)...")
    
    wiki_data = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    wiki_samples = []
    count = 0
    
    # We use streaming=True so we don't have to download 20GB at once
    for article in wiki_data:
        if count >= num_articles:
            break
            
        title = article['title']
        text = article['text'].replace('\n', ' ')
        
        # Break the article into chunks of 'chunk_size' characters
        # This ensures we get the FULL article data in 128-token bites
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if len(chunk) < 100: continue # Skip tiny fragments
            
            wiki_samples.append(f"Wiki Topic: {title}\nContent: {chunk}\n---\n")
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} articles...")

    with open(output_path, 'a', encoding='utf-8') as f:
        f.writelines(wiki_samples)
    
    print(f"🔥 Massive Wiki expansion complete! Added approx {len(wiki_samples)} chunks.")
    
def create_balanced_shuffled_dataset(file_paths, output_path="C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\DATA\\combined.txt"):
    """
    Reads multiple files, mixes all their lines, and shuffles them 
    to prevent the model from getting stuck in one 'mode'.
    """
    all_lines = []
    
    for label, path in file_paths.items():
        if os.path.exists(path):
            print(f"📖 Reading {label}...")
            with open(path, 'r', encoding='utf-8') as f:
                # We split by '---' if you used my previous separator, 
                # or just readlines for raw text.
                content = f.read().split('---')
                # Clean up empty strings and add the separator back for training
                entries = [e.strip() + "\n---\n" for e in content if len(e.strip()) > 10]
                all_lines.extend(entries)
        else:
            print(f"⚠️ Warning: {path} not found. Skipping.")

    print(f"🎲 Shuffling {len(all_lines)} total training samples...")
    random.shuffle(all_lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    
    print(f"✨ Balanced dataset created at: {output_path}")

# --- YOUR FILE PATHS ---
paths = {
    "Pretrain (Rabbit)": r"C:\Users\chand\OneDrive\Documents\pytorchplayground\AI\DATA\pretrain.txt",
    "Wiki (Facts)": r"C:\Users\chand\OneDrive\Documents\pytorchplayground\AI\DATA\wiki.txt",
    "Train1 (Sherlock)": r"C:\Users\chand\OneDrive\Documents\pytorchplayground\AI\DATA\train1.txt",
    "Train2 (SmolTalk)": r"C:\Users\chand\OneDrive\Documents\pytorchplayground\AI\DATA\train2.txt"
}

output = r"C:\Users\chand\OneDrive\Documents\pytorchplayground\AI\DATA\MASTER_TRAIN.txt"

def make_batches_fast(filename, chunk_size=256, batch_size=48, max_batches=1000):
    # Check for pre-tokenized cache first
    cache_path = filename.replace('.txt', f'_tokens_{chunk_size}.pt')
    
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}...")
        all_tokens = torch.load(cache_path).tolist()
    else:
        print(f"Tokenizing {filename} for the first time (will cache for next run)...")
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        all_tokens = enc.encode(text)
        torch.save(torch.tensor(all_tokens, dtype=torch.long), cache_path)
        print(f"Cached {len(all_tokens):,} tokens to {cache_path}")
    
    # Build chunks
    all_chunks = []
    for i in range(0, len(all_tokens) - (chunk_size + 1), chunk_size + 1):
        all_chunks.append(all_tokens[i : i + chunk_size + 1])
    
    print(f"Total chunks available: {len(all_chunks):,}")
    random.shuffle(all_chunks)
    
    # Limit to max_batches
    num_to_take = min(len(all_chunks), max_batches * batch_size)
    all_chunks = all_chunks[:num_to_take]
    
    batches = []
    for i in range(0, len(all_chunks) - batch_size, batch_size):
        batch = all_chunks[i : i + batch_size]
        x = torch.tensor([c[:-1] for c in batch], dtype=torch.long)
        y = torch.tensor([c[1:] for c in batch], dtype=torch.long)
        batches.append((x, y))
    
    print(f"Created {len(batches)} batches of size {batch_size}")
    return batches

import numpy as np

def prepare_data_fast(input_folder, output_file):
    enc = tiktoken.get_encoding("gpt2")
    
    # Open the file in 'append binary' mode
    with open(output_file, 'wb') as f_out:
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                print(f"Tokenizing {filename}...")
                try:
                    with open(os.path.join(input_folder, filename), 'r', encoding='utf-8', errors='ignore') as f_in:
                        # Process in chunks if a single file is massive
                        text = f_in.read()
                        tokens = enc.encode_ordinary(text)
                        
                        # Convert to uint16 and write directly to disk
                        np.array(tokens, dtype=np.uint16).tofile(f_out)
                except Exception as e:
                    print(f"Skipped {filename} due to error: {e}")

    print(f"Done! Binary file saved to {output_file}")
    
    
"""
Dataset Builder for biggerbrain LLM Training
=============================================
Builds two files:
  - pretrain_data.txt   (raw text, separated by <|endoftext|>)
  - finetune_data.txt   (prompt/response pairs: "user: ...\nassistant: ...")

Run:
    pip install datasets tqdm tiktoken
    python build_dataset.py

The script streams everything — no massive downloads needed upfront.
Estimated output sizes:
    pretrain_data.txt  ~10-20 GB  (depending on limits set below)
    finetune_data.txt  ~1-2 GB
"""

import os
import re
from datasets import load_dataset
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIG — tweak these to fit your disk space
# ─────────────────────────────────────────────

OUTPUT_DIR       = "./training_data"
SEPARATOR        = "<|endoftext|>"      # GPT-2 style document separator

# How many examples to take from each dataset (None = take all)
LIMITS = {
    # Pretraining
    "fineweb_edu"   : 500_000,   # ~1.5B tokens
    "gutenberg"     : None,      # ~3B tokens (take everything, it's good)
    "tinystories"   : None,      # ~500M tokens (take everything)
    "openwebtext"   : 300_000,   # ~2B tokens
    "wikipedia"     : 200_000,   # ~1B tokens

    # Finetuning
    "smoltalk"      : 100_000,   # conversational
    "orca_math"     : 50_000,    # math reasoning
    "gsm8k"         : None,      # math word problems (small dataset, take all)
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove null bytes and normalize whitespace."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{4,}", "\n\n\n", text)   # collapse excessive newlines
    text = re.sub(r" {3,}", "  ", text)         # collapse excessive spaces
    return text.strip()


def write_separator(f):
    f.write(f"\n{SEPARATOR}\n")


def safe_load(name, **kwargs):
    """Load a dataset with a friendly error message on failure."""
    try:
        print(f"\n{'─'*50}")
        print(f"Loading: {name}")
        ds = load_dataset(**kwargs, streaming=True, trust_remote_code=True)
        return ds
    except Exception as e:
        print(f"  ⚠ Could not load {name}: {e}")
        print(f"  Skipping.")
        return None


# ─────────────────────────────────────────────
#  PRETRAINING DATASETS
# ─────────────────────────────────────────────

def build_pretrain(out_path: str):
    print("\n" + "="*50)
    print("  BUILDING PRETRAINING DATA")
    print("="*50)

    total_docs = 0

    with open(out_path, "w", encoding="utf-8") as f:

        # ── 1. FineWeb-Edu ──────────────────────────────
        # High quality educational web text.
        # Best single pretraining source for a GPT-2 competitor.
        ds = safe_load(
            "FineWeb-Edu",
            path="HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train"
        )
        if ds:
            limit = LIMITS["fineweb_edu"]
            desc  = f"FineWeb-Edu (limit={limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc, total=limit)):
                if limit and i >= limit:
                    break
                text = clean_text(sample.get("text", ""))
                if len(text) > 100:
                    f.write(text)
                    write_separator(f)
                    total_docs += 1

        # ── 2. Project Gutenberg ────────────────────────
        # All public domain books. Excellent long-form reasoning data.
        ds = safe_load(
            "Project Gutenberg",
            path="sedthh/gutenberg_english",
            split="train"
        )
        if ds:
            limit = LIMITS["gutenberg"]
            desc  = f"Gutenberg (limit={'all' if not limit else limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc, total=limit)):
                if limit and i >= limit:
                    break
                text = clean_text(sample.get("TEXT", ""))
                if len(text) > 500:
                    f.write(text)
                    write_separator(f)
                    total_docs += 1

        # ── 3. TinyStories ──────────────────────────────
        # Short stories written for small LMs.
        # Proven to make 117M models coherent and fluent.
        ds = safe_load(
            "TinyStories",
            path="roneneldan/TinyStories",
            split="train"
        )
        if ds:
            limit = LIMITS["tinystories"]
            desc  = f"TinyStories (limit={'all' if not limit else limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc, total=limit)):
                if limit and i >= limit:
                    break
                text = clean_text(sample.get("text", ""))
                if len(text) > 50:
                    f.write(text)
                    write_separator(f)
                    total_docs += 1

        # ── 4. OpenWebText ──────────────────────────────
        # Open recreation of GPT-2's actual training data (WebText).
        # This is essentially what made GPT-2 work.
        ds = safe_load(
            "OpenWebText",
            path="Skylion007/openwebtext",
            split="train"
        )
        if ds:
            limit = LIMITS["openwebtext"]
            desc  = f"OpenWebText (limit={limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc, total=limit)):
                if limit and i >= limit:
                    break
                text = clean_text(sample.get("text", ""))
                if len(text) > 200:
                    f.write(text)
                    write_separator(f)
                    total_docs += 1

        # ── 5. Wikipedia ────────────────────────────────
        # Factual grounding. Helps the model learn world knowledge.
        ds = safe_load(
            "Wikipedia",
            path="wikimedia/wikipedia",
            name="20231101.en",
            split="train"
        )
        if ds:
            limit = LIMITS["wikipedia"]
            desc  = f"Wikipedia (limit={limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc, total=limit)):
                if limit and i >= limit:
                    break
                text = clean_text(sample.get("text", ""))
                if len(text) > 200:
                    f.write(text)
                    write_separator(f)
                    total_docs += 1

    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"\n✅ Pretraining data saved: {out_path}")
    print(f"   Documents : {total_docs:,}")
    print(f"   File size : {size_gb:.2f} GB")


# ─────────────────────────────────────────────
#  FINETUNING DATASETS
# ─────────────────────────────────────────────

def format_conversation(user_msg: str, assistant_msg: str) -> str:
    """
    Format a conversation pair to match the model's think() function format.
    This MUST match: f"user: {prompt}\nassistant:"
    """
    user_msg      = clean_text(user_msg)
    assistant_msg = clean_text(assistant_msg)
    if not user_msg or not assistant_msg:
        return ""
    return f"user: {user_msg}\nassistant: {assistant_msg}"


def build_finetune(out_path: str):
    print("\n" + "="*50)
    print("  BUILDING FINETUNING DATA")
    print("="*50)

    total_pairs = 0

    with open(out_path, "w", encoding="utf-8") as f:

        # ── 1. SmolTalk ─────────────────────────────────
        # High quality conversational data.
        # Teaches the model to be helpful and coherent in chat format.
        ds = safe_load(
            "SmolTalk",
            path="HuggingFaceTB/smoltalk",
            name="all",
            split="train"
        )
        if ds:
            limit = LIMITS["smoltalk"]
            desc  = f"SmolTalk (limit={limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc, total=limit)):
                if limit and i >= limit:
                    break
                messages = sample.get("messages", [])
                # Extract consecutive user/assistant pairs
                for j in range(len(messages) - 1):
                    if (messages[j].get("role") == "user" and
                        messages[j+1].get("role") == "assistant"):
                        pair = format_conversation(
                            messages[j]["content"],
                            messages[j+1]["content"]
                        )
                        if pair:
                            f.write(pair)
                            write_separator(f)
                            total_pairs += 1

        # ── 2. Orca Math ────────────────────────────────
        # Math word problems with chain-of-thought reasoning.
        # Critical for teaching the model step-by-step thinking.
        ds = safe_load(
            "Orca Math",
            path="microsoft/orca-math-word-problems-200k",
            split="train"
        )
        if ds:
            limit = LIMITS["orca_math"]
            desc  = f"Orca Math (limit={limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc, total=limit)):
                if limit and i >= limit:
                    break
                question = sample.get("question", "")
                answer   = sample.get("answer", "")
                pair = format_conversation(question, answer)
                if pair:
                    f.write(pair)
                    write_separator(f)
                    total_pairs += 1

        # ── 3. GSM8K ────────────────────────────────────
        # Grade school math. Small but high quality reasoning dataset.
        # Great for teaching the model basic arithmetic logic.
        ds = safe_load(
            "GSM8K",
            path="openai/gsm8k",
            name="main",
            split="train"
        )
        if ds:
            limit = LIMITS["gsm8k"]
            desc  = f"GSM8K (limit={'all' if not limit else limit})"
            for i, sample in enumerate(tqdm(ds, desc=desc)):
                if limit and i >= limit:
                    break
                question = sample.get("question", "")
                answer   = sample.get("answer", "")
                pair = format_conversation(question, answer)
                if pair:
                    f.write(pair)
                    write_separator(f)
                    total_pairs += 1

    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"\n✅ Finetuning data saved: {out_path}")
    print(f"   Pairs     : {total_pairs:,}")
    print(f"   File size : {size_gb:.2f} GB")


# ─────────────────────────────────────────────
#  TOKENIZATION HELPER (optional but useful)
# ─────────────────────────────────────────────

def estimate_tokens(file_path: str):
    """
    Rough token count estimate without loading entire file.
    GPT-2 averages ~4 characters per token.
    """
    size_bytes = os.path.getsize(file_path)
    chars      = size_bytes  # UTF-8 ASCII text ≈ 1 byte/char
    tokens     = chars / 4
    print(f"   Estimated tokens: {tokens/1e9:.2f}B  ({tokens/1e6:.0f}M)")


# ─────────────────────────────────────────────
#  DATALOADER HELPER
#  Drop this into your training file to use the
#  text files as a streaming dataset.
# ─────────────────────────────────────────────

DATALOADER_SNIPPET = '''
# ── How to use the output files in your training loop ──────────────────────
#
# In your training script:
#
# import tiktoken
# import torch
# from torch.utils.data import IterableDataset, DataLoader
#
# enc = tiktoken.get_encoding("gpt2")
# SEP_TOKEN = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
# SEQ_LEN   = 1024  # match your model's self.sequencelength
#
# class TextFileDataset(IterableDataset):
#     def __init__(self, path, seq_len=SEQ_LEN):
#         self.path    = path
#         self.seq_len = seq_len
#
#     def __iter__(self):
#         buffer = []
#         with open(self.path, "r", encoding="utf-8") as f:
#             for line in f:
#                 tokens = enc.encode(line, allowed_special={"<|endoftext|>"})
#                 buffer.extend(tokens)
#                 # Yield seq_len+1 chunks (input + target)
#                 while len(buffer) >= self.seq_len + 1:
#                     chunk = buffer[:self.seq_len + 1]
#                     x = torch.tensor(chunk[:-1], dtype=torch.long)
#                     y = torch.tensor(chunk[1:],  dtype=torch.long)
#                     yield x, y
#                     buffer = buffer[self.seq_len:]  # slide window
#
# # Usage:
# pretrain_ds = TextFileDataset("training_data/pretrain_data.txt")
# loader      = DataLoader(pretrain_ds, batch_size=32, num_workers=4)
# model.trainingloop(loader, epochs=1)
'''


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pretrain_path  = os.path.join(OUTPUT_DIR, "pretrain_data.txt")
    finetune_path  = os.path.join(OUTPUT_DIR, "finetune_data.txt")
    snippet_path   = os.path.join(OUTPUT_DIR, "dataloader_snippet.py")

    # Build datasets
    build_pretrain(pretrain_path)
    build_finetune(finetune_path)

    # Print token estimates
    print("\n── Token Estimates ──────────────────────────────")
    print(f"Pretrain:")
    estimate_tokens(pretrain_path)
    print(f"Finetune:")
    estimate_tokens(finetune_path)

    # Save the dataloader snippet
    with open(snippet_path, "w") as f:
        f.write(DATALOADER_SNIPPET.strip())
    print(f"\n── Saved dataloader snippet: {snippet_path}")

    print("\n" + "="*50)
    print("  ALL DONE")
    print("="*50)
    print(f"  {pretrain_path}")
    print(f"  {finetune_path}")
    print(f"  {snippet_path}")
    print()
    print("  Training order:")
    print("  1. python train.py --data pretrain_data.txt --lr 3e-4")
    print("  2. python train.py --data finetune_data.txt --lr 3e-5")
    print()
    print("  Upload to your rented 5090 with:")
    print("  rsync -avz training_data/ user@<instance-ip>:~/data/")