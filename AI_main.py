
if __name__ == '__main__':
    
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from biggerbrain import think, initmodel # Import the specific class
    import training_utils as t_u
    
    import time
    import ai_extras as A_E
    # Also force compile errors to be visible:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.verbose = True
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:",device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    #-----Vars
    filename = os.path.join(BASE_DIR, "DATA", "pretrain.txt")
    trainfilename = os.path.join(BASE_DIR, "DATA", "wiki.txt")
    train1filename = os.path.join(BASE_DIR, "DATA", "train1.txt")
    train2filename = os.path.join(BASE_DIR, "DATA", "combined.txt")
    bin = os.path.join(BASE_DIR, "DATA", "training_data.bin")
    lr = 0.0001
    train_lr = 0.00005
    subsetfraction = 0.05
    epochs = 100
    batchsize = 48
    chunksize= 256
    maxbatches = 100
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
        elif user_input.lower() == "view":
            model.forward
        elif user_input.lower() == "debug mode" or user_input.lower() == "debugmode":
            model.debugprints = True
            print(f"Debug mode enabled on model : {model}")
        elif user_input.lower() == "train":
            if 'train_ds' not in locals():
                train_ds = A_E.StreamDataset(bin_file=bin, seq_len=chunksize)
                print(f"Dataset loaded: {len(train_ds)} samples")
            model.trainingloop(train_ds, epochs=epochs, lr=train_lr, batchsize=batchsize, subset_fraction=subsetfraction, accumulation_steps=3)#._orig_mod

        elif user_input.lower() == "pretrain":#if inputs 'train'
        
            batches = t_u.make_batches_fast(filename=filename, chunk_size=chunksize, 
                                 batch_size=batchsize, max_batches=maxbatches)
        
            model.trainingloop(batches, epochs=epochs, lr=lr, batchsize=batchsize)#TODO: implement profiling.
        elif user_input.lower() == "profile":
        
            from torch.profiler import profile, ProfilerActivity
            test_input = torch.zeros(32, 256, dtype=torch.long).to(device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with torch.no_grad():
                    model.forward(test_input, iter=3, is_training=False)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
        elif user_input.lower() == "dataset copy":
        
            t_u.build_dataset(filename)
        elif user_input.lower() == "load":
        
            model.load_state_dict(torch.load("C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\best_model.pt", weights_only=True))
            print("Weights loaded!")
        elif user_input.lower() == "save":
        
            torch.save(model.state_dict(), "C:\\Users\\chand\\OneDrive\\Documents\\pytorchplayground\\AI\\manual_save.pt")
            print("Saved!")
        elif user_input.lower() == "filesize":
        
            size = os.path.getsize(filename) / 1024 / 1024
            print(f"Training file: {size:.1f} MB")
        elif user_input.lower() == "makefiles":
            t_u.build_pretrain("pretrain.txt")
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
            
        else:
        
            with torch.no_grad():
                think(user_input, model, iter=3)