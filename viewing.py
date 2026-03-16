import torch
import numpy as np
import plotly.graph_objects as go
import tiktoken
import umap
from biggerbrain import biggerbrain   # your model file

enc    = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ───────────────────────────────────────────────────────────
model = biggerbrain(device).to(device)
model.load_state_dict(torch.load("model_best.pth", map_location=device))
model.eval()

def visualize_thinking(prompt: str, model, iter: int = 3):
    model.eval()
    
    formatted = f"user: {prompt}\nassistant:"
    input_ids = torch.tensor([enc.encode(formatted,
                    allowed_special={'<|endoftext|>'})]).to(model.device)
    token_strs = [enc.decode([t]) for t in 
                  enc.encode(formatted, allowed_special={'<|endoftext|>'})]

    with torch.no_grad():
        # Only generate 1 token — we just want to see the thinking states
        logits, states = model.forward_chat(input_ids, outlength=1, iter=iter)

    # states structure from your new code per word:
    #   [0]         = pre-loop state (after pre-block)
    #   [1..iter]   = after each thinking iteration
    #   [iter+1]    = after post-block
    # Total per word = iter + 2

    num_tokens  = len(token_strs)
    states_per_word = iter + 2  # pre + iters + post

    # We only generated 1 word, so take first (iter+2) states
    word_states = states[:states_per_word]  # list of [1, seq, 768] tensors

    # Stack into [num_stages, seq, 768]
    stacked = np.vstack([s[0].numpy() for s in word_states])
    # stacked shape: [(iter+2) * seq_len, 768]

    print(f"Fitting UMAP on {stacked.shape[0]} vectors...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15,
                        min_dist=0.1, random_state=42)
    coords  = reducer.fit_transform(stacked)

    # Reshape to [num_stages, seq_len, 3]
    seq_len    = word_states[0].shape[1]
    num_stages = len(word_states)
    coords     = coords.reshape(num_stages, seq_len, 3)

    # Labels for each stage
    stage_labels = ["Pre-loop"] + \
                   [f"Iter {i+1}" for i in range(iter)] + \
                   ["Post-block"]
    colors = ["#4488ff", "#ff8844", "#44ff88", "#ff44aa",
              "#ffff44", "#44ffff"][:num_stages]

    fig = go.Figure()

    for i in range(num_stages):
        x, y, z = coords[i,:,0], coords[i,:,1], coords[i,:,2]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            marker=dict(size=6, color=colors[i], opacity=0.9),
            text=token_strs[:seq_len],
            textposition="top center",
            textfont=dict(size=8),
            name=stage_labels[i]
        ))

        # Lines showing token movement between stages
        if i > 0:
            for t in range(min(seq_len, len(token_strs))):
                fig.add_trace(go.Scatter3d(
                    x=[coords[i-1,t,0], coords[i,t,0]],
                    y=[coords[i-1,t,1], coords[i,t,1]],
                    z=[coords[i-1,t,2], coords[i,t,2]],
                    mode="lines",
                    line=dict(color=colors[i], width=1),
                    opacity=0.3,
                    showlegend=False
                ))

    fig.update_layout(
        title=f"Token trajectories — '{prompt}'",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
            bgcolor="rgb(10,10,20)"
        ),
        paper_bgcolor="rgb(10,10,20)",
        font_color="white",
    )

    fig.write_html("thought_viz.html")
    fig.show()
    print("Saved to thought_viz.html")
    
visualize_thinking("The man ran. The man jumped. The man ", model, iter=3)