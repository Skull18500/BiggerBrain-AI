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
model.load_state_dict(torch.load("model_best.pt", map_location=device))
model.eval()

def visualize_thinking(prompt: str, iter: int = 3):
        # Tokenize
        tokens    = enc.encode(f"user: {prompt}\nassistant:", 
                           allowed_special={"<|endoftext|>"})
        token_strs = [enc.decode([t]) for t in tokens]  # for labels
        input_ids  = torch.tensor([tokens]).to(device)

        with torch.no_grad():
            states = model.forward_with_hooks(input_ids, iter=iter)

        # states: list of (iter+1) tensors [1, seq, 768]
        # Flatten to [num_tokens * num_iters, 768] for UMAP
        num_iters  = len(states)     # iter+1
        num_tokens = len(tokens)

        all_vecs   = np.vstack([s[0].numpy() for s in states])
        # Shape: [num_tokens * num_iters, 768]

        #── Project to 3D ────────────────────────────────────────────────────
        print("Fitting UMAP (this takes ~10 seconds)...")
        reducer = umap.UMAP(n_components=3, n_neighbors=15, 
                        min_dist=0.1, random_state=42)
        coords  = reducer.fit_transform(all_vecs)
        # coords: [num_tokens * num_iters, 3]

        # Reshape to [num_iters, num_tokens, 3]
        coords = coords.reshape(num_iters, num_tokens, 3)

        # ── Build Plotly 3D figure ───────────────────────────────────────────
        fig    = go.Figure()
        colors = ["#4488ff", "#ff8844", "#44ff88", "#ff44aa"]  # one per iter
        labels = [f"Pre-loop", "Iter 1", "Iter 2", "Iter 3"][:num_iters]

        for i in range(num_iters):
            x, y, z = coords[i,:,0], coords[i,:,1], coords[i,:,2]

            # Points — one per token at this iteration
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers+text",
                marker=dict(size=6, color=colors[i], opacity=0.9),
                text=token_strs,
                textposition="top center",
                textfont=dict(size=8),
                name=labels[i]
            ))

            # Lines connecting same token across iterations (shows movement)
            if i > 0:
                for t in range(num_tokens):
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
            title=f"Token trajectories across thinking iterations<br>"
                f"<sub>'{prompt}'</sub>",
            scene=dict(
                xaxis_title="UMAP-1",
                yaxis_title="UMAP-2",
                zaxis_title="UMAP-3",
                bgcolor="rgb(10,10,20)"
            ),
                paper_bgcolor="rgb(10,10,20)",
                font_color="white",
                legend=dict(bgcolor="rgba(0,0,0,0.5)")
        )

        fig.write_html("thought_viz.html")
        fig.show()
        print("Saved to thought_viz.html")


biggerbrain.train
        
visualize_thinking("What is the capital of France?", iter=3)