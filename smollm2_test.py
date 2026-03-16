from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompts = [
    "The glass fell off the table and hit the floor.",
    "Every morning John checked the weather before leaving. On Tuesday he forgot his umbrella. That afternoon,",
    "One plus one equals two. Two plus two equals four. Three plus three equals six. Four plus four equals",
    "She put the key in the lock, turned it, and pushed. The door did not open. She tried again. Still nothing. She looked down and realized",
    "The village had not seen rain in three months. The crops were dying and the river had slowed to a trickle. The farmers looked at the sky and"
]

checkpoint = "HuggingFaceTB/SmolLM2-135M"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
model      = AutoModelForCausalLM.from_pretrained(
                 checkpoint, torch_dtype=torch.bfloat16
             ).to("cpu")  # CPU to not kill your training

for prompt in prompts:
    inputs  = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=40,
                             temperature=0.7, do_sample=True,
                             top_k=40)
    result  = tokenizer.decode(outputs[0][inputs.shape[1]:])
    print(f"\nPROMPT: {prompt}")
    print(f"SmolLM2: {result}")