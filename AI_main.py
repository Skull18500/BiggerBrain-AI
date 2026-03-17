
if __name__ == '__main__':
    
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from biggerbrain import think, initmodel # Import the specific class
    import training_utils as t_u
    
    import time
    
    import random
    import ai_extras as A_E
    # Also force compile errors to be visible:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.verbose = True
    torch.set_float32_matmul_precision('medium')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:",device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if device == "cuda":
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"Free:      {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
    
    #import subprocess
    #subprocess.run(["nvidia-smi"], shell=True)
    
    #-----Vars
    filename = os.path.join(BASE_DIR, "DATA", "pretrain.txt")
    trainfilename = os.path.join(BASE_DIR, "DATA", "wiki.txt")
    train1filename = os.path.join(BASE_DIR, "DATA", "train1.txt")
    train2filename = os.path.join(BASE_DIR, "DATA", "combined.txt")
    bin = os.path.join(BASE_DIR, "DATA", "training_data.bin")
    lr = 0.0002
    train_lr = 0.00001
    subsetfraction = 0.1
    epochs = 2
    batchsize = 64
    chunksize= 512
    #-----

    model = initmodel(device)
    model.sequencelength = chunksize

    print("Parameters: ", sum(p.numel() for p in model.parameters()))#._orig_mod

    print(torch.cuda.get_device_name(0))
    print(f"Compiled: {hasattr(model, '_orig_mod')}")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "cpu":
            model = model.to_device("cpu")
            
        elif user_input.lower() == "view":
            model.forward
        elif user_input.lower() == "debug mode" or user_input.lower() == "debugmode":
            model.debugprints = True
            print(f"Debug mode enabled on model : {model}")
        elif user_input.lower() == "train":
            if 'train_ds' not in locals():
                train_ds = A_E.StreamDataset(bin_file=bin, seq_len=chunksize)
                print(f"Dataset loaded: {len(train_ds)} samples")
            model.trainingloop(train_ds, epochs=epochs, lr=train_lr, batchsize=batchsize, subset_fraction=subsetfraction, accumulation_steps=2)#._orig_mod

        elif user_input.lower() == "pretrain":
            
            #bin_dir = t_u.tokenize_to_binary("C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\pretrain\\pretraining.txt")
            if 'pretrain_ds' not in locals():
                pretrain_ds = A_E.StreamDataset(bin_file="C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\pretrain\\pretraining.bin", seq_len=chunksize)
                print(f"Dataset loaded: {len(pretrain_ds)} samples")
            
            model.trainingloop(pretrain_ds, epochs=epochs, lr=lr, batchsize=batchsize, accumulation_steps=6, subset_fraction=subsetfraction)#TODO: implement profiling.
        elif user_input.lower() == "profile":
        
            from torch.profiler import profile, ProfilerActivity
            
            test_input = []
            test_input.append(random.randint(1, t_u.enc.max_token_value+1))
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with torch.no_grad():
                    model.forward_chat(test_input, 1, 3)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
        elif user_input.lower() == "dataset copy":
        
            t_u.build_dataset(filename)
        elif user_input.lower() == "load":
        
            model.load_state_dict(torch.load("C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\model_best.pth", weights_only=True))# .pth or .pt?
            print("Weights loaded!")
        elif user_input.lower() == "save":
        
            torch.save(model.state_dict(), "C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\manual_save.pt")
            print("Saved!")
        elif user_input.lower() == "filesize":
        
            size = os.path.getsize(filename) / 1024 / 1024
            print(f"Training file: {size:.1f} MB")
        elif user_input.lower() == "makefiles":
            t_u. build_pretrain("C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\pretraining.txt")
            train_ds = A_E.StreamDataset(
            bin_file=bin, 
            seq_len=chunksize
            )
            
        elif user_input.lower() == "speedtest":
            print("Testing raw forward pass speed...")
            test_input = torch.randint(0, t_u.enc.max_token_value + 1 - 1, (48, 256)).to(device)
    
            # Warmup - compile needs a few passes to finish compiling
            print("Warming up torch.compile (this takes 30-60 seconds first time)...")
            for i in range(3):
                with torch.no_grad():
                    model.forward(test_input, iter=3, is_training=True)
    
            # Actual timed test
            torch.cuda.synchronize()
            start = time.time()
            for i in range(10):
                with torch.amp.autocast('cuda'):
                    logits, iters = model._orig_mod.forward(test_input, iter=3, is_training=False)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 10
            print(f"Average forward pass: {elapsed*1000:.0f}ms")
            print(f"Estimated epoch time with 200 batches: {elapsed*200/60:.1f} minutes")
            
            
        elif user_input.lower() == "greed":
            think_greedy("The old man", model)
            think_greedy("user: hello\nassistant:", model)
        elif user_input.lower() == "check":
            model.eval()
            with torch.no_grad():
                prompt = "The man ran"
                ids = torch.tensor([t_u.enc.encode(prompt)]).to(model.device)
                w = model.embed(ids)
                mask = torch.triu(
                torch.ones(ids.size(1), ids.size(1), 
                      device=model.device) * float('-inf'),
                diagonal=1
                )
            y, _ = model.layerA1(w, w, w, attn_mask=mask, rope=model.rope)
            out = model.layerO1(y[:, -1, :])
            top10 = out[0].topk(10)
            print("Top 10 predicted next tokens:")
            for val, idx in zip(top10.values, top10.indices):
                tok = t_u.enc.decode([idx.item()])
                print(f"  '{tok}' (id={idx.item()}) = {val.item():.3f}")
        
            print(f"\nEmbed std: {model.embed.weight.std().item():.4f}")
            
        elif user_input.lower() == "mol":
            model.eval()
            with torch.no_grad():
                prompt = "The man ran"
                ids = torch.tensor([t_u.enc.encode(prompt)]).to(model.device)
                w = model.embed(ids)
            # Run one forward pass and check routing weights
            for module in model.modules():
                if isinstance(module, A_E.MoLLayer):
                    if module.router.last_weights is not None:
                        w = module.router.last_weights
                        print(f"Expert weights: {w.tolist()}")
                        # Healthy: [0.6, 0.4] or [0.7, 0.3]
                        # Collapsed: [0.99, 0.01] ← one expert dominates
        elif user_input.lower() == "resetmol":
            for module in model.modules():
                if isinstance(module, A_E.ThinkingRouter):
                    A_E.nn.init.normal_(module.router[0].weight, std=0.01)
                    A_E.nn.init.normal_(module.router[2].weight, std=0.01)
                    A_E.nn.init.zeros_(module.router[0].bias) if hasattr(module.router[0], 'bias') else None
            print("Router weights reset")
            torch.save(model.state_dict(), "model_best.pth")
        else:
            with torch.no_grad():
                think(user_input, model, iter=3)