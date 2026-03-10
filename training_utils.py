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