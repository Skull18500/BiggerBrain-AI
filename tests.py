import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.encode(":"))     # single colon
print(enc.encode("::"))    # double colon
print(enc.encode(" :"))    # space colon
print(enc.encode(":::"))   # triple colon