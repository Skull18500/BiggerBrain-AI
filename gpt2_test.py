from transformers import pipeline

gpt2 = pipeline("text-generation", model="gpt2", device=-1)

prompts = [
    "The glass fell off the table and hit the floor.",
    "Every morning John checked the weather before leaving. On Tuesday he forgot his umbrella. That afternoon,",
    "One plus one equals two. Two plus two equals four. Three plus three equals six. Four plus four equals",
    "She put the key in the lock, turned it, and pushed. The door did not open. She tried again. Still nothing. She looked down and realized",
    "The village had not seen rain in three months. The crops were dying and the river had slowed to a trickle. The farmers looked at the sky and"
]

for prompt in prompts:
    result = gpt2(
        prompt,
        max_new_tokens=40,
        temperature=0.7,
        do_sample=True,
        top_k=40
    )
    print(f"\nPROMPT: {prompt}")
    print(f"GPT-2:  {result[0]['generated_text'][len(prompt):]}")
    print("-" * 60)