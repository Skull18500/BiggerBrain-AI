import torch
import numpy as np
import plotly.graph_objects as go
import tiktoken
import umap
import webbrowser
import os
from biggerbrain import biggerbrain   # your model file

enc    = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ───────────────────────────────────────────────────────────
model = biggerbrain(device).to(device)
model.load_state_dict(torch.load("model_best.pth", map_location=device))
model.eval()


 
enc = tiktoken.get_encoding("gpt2")
 
# ── Color palette — one per stage, high contrast on dark background ──────────
STAGE_COLORS = [
    "#4a9eff",   # Pre-loop    — bright blue
    "#ff7f44",   # Iter 1      — orange
    "#44ff88",   # Iter 2      — green
    "#ff44aa",   # Iter 3      — pink/magenta
    "#ffee44",   # Post-block  — yellow
    "#44ffee",   # Extra iter  — cyan (if iter > 3)
]
 
 
def visualize_thinking(
    prompt:   str,
    model,
    iter:     int  = 3,
    out_path: str  = "thought_viz.html",
    open_browser: bool = True
):
    """
    Generate a 3D visualization of token thought trajectories.
 
    Args:
        prompt:       Text prompt to visualize
        model:        BiggerBrain model instance (eval mode, on device)
        iter:         Number of thinking iterations to run
        out_path:     Where to save the HTML file
        open_browser: Auto-open in browser after saving
    """
    model.eval()
 
    # ── Tokenize ─────────────────────────────────────────────────────────────
    formatted  = f"user: {prompt}\nassistant:"
    token_ids  = enc.encode(formatted, allowed_special={"<|endoftext|>"})
    token_strs = []
    for t in token_ids:
        raw = enc.decode([t])
        # Clean up whitespace for display
        raw = raw.replace("\n", "↵").replace(" ", "·")
        token_strs.append(raw if raw.strip() else "·")
 
    input_ids = torch.tensor([token_ids]).to(model.device)
 
    # ── Run forward pass and collect states ──────────────────────────────────
    with torch.no_grad():
        # Generate 1 token — we just want the internal states
        _, states = model.forward_chat(input_ids, outlength=1, iter=iter)
 
    # States structure per word (from your forward_chat):
    #   [0]         = pre-loop  (after pre-block)
    #   [1 .. iter] = after each thinking iteration
    #   [iter+1]    = after post-block
    # Total = iter + 2
    states_per_word = iter + 2
    word_states     = states[:states_per_word]   # only first word
 
    if len(word_states) == 0:
        print("No states captured — check forward_chat returns states correctly.")
        return
 
    seq_len    = word_states[0].shape[1]
    num_stages = len(word_states)
 
    # Clip token_strs to actual sequence length
    token_strs = token_strs[:seq_len]
 
    # ── Stack all states for UMAP ─────────────────────────────────────────────
    # Shape: [num_stages * seq_len, hidden_dim]
    all_vecs = np.vstack([s[0].float().numpy() for s in word_states])
 
    print(f"Fitting UMAP on {all_vecs.shape[0]} vectors "
          f"({num_stages} stages × {seq_len} tokens)...")
 
    reducer = umap.UMAP(
        n_components = 3,
        n_neighbors  = min(15, all_vecs.shape[0] - 1),
        min_dist     = 0.1,
        random_state = 42,
        metric       = "cosine"   # cosine distance works well for embeddings
    )
    coords = reducer.fit_transform(all_vecs)
    # Reshape to [num_stages, seq_len, 3]
    coords = coords.reshape(num_stages, seq_len, 3)
 
    # ── Stage labels ──────────────────────────────────────────────────────────
    stage_labels = (
        ["Pre-loop"] +
        [f"Iter {i+1}" for i in range(iter)] +
        ["Post-block"]
    )[:num_stages]
 
    colors = STAGE_COLORS[:num_stages]
 
    # ── Build Plotly figure ───────────────────────────────────────────────────
    fig = go.Figure()
 
    for i in range(num_stages):
        x = coords[i, :, 0]
        y = coords[i, :, 1]
        z = coords[i, :, 2]
 
        # ── Token dots + labels ───────────────────────────────────────────────
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            marker=dict(
                size=7,
                color=colors[i],
                opacity=0.95,
                line=dict(
                    color="black",   # dark outline around each dot
                    width=1
                )
            ),
            text=token_strs,
            textposition="top center",
            textfont=dict(
                size=9,
                color="white",
                family="Arial Black"  # bold font = readable without background
            ),
            name=stage_labels[i],
            hovertemplate=(
                f"<b>{stage_labels[i]}</b><br>"
                "Token: %{text}<br>"
                "x: %{x:.2f}<br>"
                "y: %{y:.2f}<br>"
                "z: %{z:.2f}"
                "<extra></extra>"
            )
        ))
 
        # ── Lines connecting same token across stages ─────────────────────────
        if i > 0:
            for t in range(seq_len):
                fig.add_trace(go.Scatter3d(
                    x=[coords[i-1, t, 0], coords[i, t, 0]],
                    y=[coords[i-1, t, 1], coords[i, t, 1]],
                    z=[coords[i-1, t, 2], coords[i, t, 2]],
                    mode="lines",
                    line=dict(
                        color=colors[i],
                        width=2
                    ),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo="skip"
                ))
 
    # ── Layout — full dark theme ──────────────────────────────────────────────
    axis_style = dict(
    backgroundcolor = "rgb(5, 5, 15)",
    gridcolor       = "rgb(35, 35, 60)",
    showbackground  = True,
    zerolinecolor   = "rgb(60, 60, 100)",
    tickfont        = dict(color="rgba(200,200,255,0.7)", size=9),
    )
 
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Token Thought Trajectories</b><br>"
                f"<span style='font-size:13px; color:#aaaaff'>"
                f"Prompt: \"{prompt}\" — {iter} thinking iterations</span>"
            ),
            font=dict(size=18, color="white"),
            x=0.5,
            xanchor="center"
        ),
        scene=dict(
            xaxis=dict(**axis_style, title=dict(text="UMAP-1",
            font=dict(color="rgba(200,200,255,0.9)", size=11))),
            yaxis=dict(**axis_style, title=dict(text="UMAP-2",
            font=dict(color="rgba(200,200,255,0.9)", size=11))),
            zaxis=dict(**axis_style, title=dict(text="UMAP-3",
            font=dict(color="rgba(200,200,255,0.9)", size=11))),
            bgcolor="rgb(8, 8, 18)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)   # good default viewing angle
            )
        ),
        paper_bgcolor="rgb(8, 8, 18)",
        plot_bgcolor ="rgb(8, 8, 18)",
        font=dict(color="white", family="Arial"),
        legend=dict(
            bgcolor    = "rgba(20,20,40,0.85)",
            bordercolor= "rgba(100,100,180,0.4)",
            borderwidth= 1,
            font       = dict(size=12, color="white"),
            itemsizing = "constant",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top"
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        # Annotation explaining controls
        annotations=[dict(
            text="🖱 Drag to rotate  •  Scroll to zoom  •  Click legend to toggle stages",
            xref="paper", yref="paper",
            x=0.5, y=0.01,
            xanchor="center", yanchor="bottom",
            font=dict(size=10, color="rgba(150,150,200,0.7)"),
            showarrow=False
        )]
    )
 
    # ── Save and open ─────────────────────────────────────────────────────────
    fig.write_html(out_path)
    print(f"\nSaved to: {os.path.abspath(out_path)}")
    print(f"Stages:   {stage_labels}")
    print(f"Tokens:   {seq_len}")
 
    if open_browser:
        webbrowser.open(f"file:///{os.path.abspath(out_path)}")
 
    return fig
 
 
# ── Quick test runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(
        r"C:\Users\chand\OneDrive\Documents\pytorchplayground\AI"
    )
    from biggerbrain import biggerbrain
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = biggerbrain(device).to(device)
 
    checkpoint = "model_best.pth"
    if os.path.exists(checkpoint):
        model.load_state_dict(
            torch.load(checkpoint, map_location=device)
        )
        print(f"Loaded weights from {checkpoint}")
    else:
        print("No checkpoint found — using random weights")
 
    model.eval()
 
    # Test prompts — try a few to see different patterns
    prompts = [
        "The man ran and",
        "The glass fell off the table and",
        "Once upon a time there was a",
    ]
 
    for i, prompt in enumerate(prompts):
        out_file = f"thought_viz_{i}.html"
        print(f"\n{'='*50}")
        print(f"Visualizing: '{prompt}'")
        visualize_thinking(prompt, model, iter=3,
                          out_path=out_file,
                          open_browser=(i == 0))  # only auto-open first one
    
visualize_thinking("The man ran. The man jumped. The man ", model, iter=3)