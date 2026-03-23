import re, math, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
from collections import Counter

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cpu")  # HF Spaces free tier uses CPU

# ── Load artifacts ────────────────────────────────────────────────────────────
print("Loading artifacts...")

df = pd.read_csv("tickets_clean.csv")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("tfidf_dicts.pkl", "rb") as f:
    tfidf_dicts = pickle.load(f)

idf        = torch.load("idf.pt",        map_location=device)
tfidf_norm = torch.load("tfidf_norm.pt", map_location=device)
glove_norm = torch.load("glove_norm.pt", map_location=device)

VOCAB_SIZE = len(vocab)
GLOVE_DIM  = glove_norm.shape[1]

print(f"Loaded! Dataset: {df.shape}, Vocab: {VOCAB_SIZE}, GloVe: {glove_norm.shape}")

# ── Stopwords ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and",
    "or","but","with","this","that","was","are","be","been","have",
    "has","had","i","we","my","your","you","he","she","they","its",
    "do","did","so","if","as","by","from","about","just",
    "also","more","than",
    "me","us","him","her","them","our","their","what","which","who",
    "how","when","where","there","here","am","being",
    "all","any","some","one","two","into"
}

# ── Tokenizer ─────────────────────────────────────────────────────────────────
def tokenize(text):
    text   = str(text).lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS]

def get_ngrams(tokens, n):
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def tokenize_with_ngrams(text):
    tokens   = tokenize(text)
    bigrams  = get_ngrams(tokens, 2)
    trigrams = get_ngrams(tokens, 3)
    return tokens + bigrams + trigrams

# ── TF-IDF query vector ───────────────────────────────────────────────────────
def tfidf_query_vector(text):
    tokens = tokenize_with_ngrams(text)
    tf     = Counter(tokens)
    max_tf = max(tf.values(), default=1)
    vec    = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for tok, count in tf.items():
        if tok in vocab:
            vec[vocab[tok]] = (count / max_tf) * idf[vocab[tok]].item()
    return vec

# ── GloVe query vector ────────────────────────────────────────────────────────
def text_to_glove_vector(text):
    """
    For query encoding — uses uniform weights since no tfidf_row available.
    Looks up each token in glove_norm directly via vocab.
    """
    tokens = tokenize(text)
    if not tokens:
        return torch.zeros(GLOVE_DIM)

    vecs = []
    for tok in tokens:
        if tok in vocab:
            idx = vocab[tok]
            if idx < glove_norm.shape[0]:
                vecs.append(glove_norm[idx])

    if not vecs:
        return torch.zeros(GLOVE_DIM)

    return torch.stack(vecs).mean(0)

# ── Hybrid search ─────────────────────────────────────────────────────────────
def hybrid_search(query, alpha=0.5, top_k=5):
    # TF-IDF similarity
    q_tfidf   = tfidf_query_vector(query)
    q_tfidf   = q_tfidf / (q_tfidf.norm() + 1e-9)
    sim_tfidf = tfidf_norm @ q_tfidf

    # GloVe similarity
    q_glove   = text_to_glove_vector(query)
    q_glove   = q_glove / (q_glove.norm() + 1e-9)
    sim_glove = glove_norm @ q_glove

    # Hybrid score
    score = alpha * sim_tfidf + (1 - alpha) * sim_glove
    topk  = score.topk(top_k)

    results              = df.iloc[topk.indices.tolist()].copy()
    results["score"]     = topk.values.detach().numpy()
    results["sim_tfidf"] = sim_tfidf[topk.indices].detach().numpy()
    results["sim_glove"] = sim_glove[topk.indices].detach().numpy()
    return results[[
        "Ticket Description","Ticket Type",
        "Ticket Priority","Ticket Channel",
        "score","sim_tfidf","sim_glove"
    ]]

# ── Gradio function ───────────────────────────────────────────────────────────
def compare_search(query, alpha):
    if not query.strip():
        return "Please enter a ticket description.", "", ""

    r_tfidf  = hybrid_search(query, alpha=1.0, top_k=3)
    r_glove  = hybrid_search(query, alpha=0.0, top_k=3)
    r_hybrid = hybrid_search(query, alpha=alpha, top_k=3)

    predicted_type = r_hybrid["Ticket Type"].mode()[0]

    # TF-IDF output
    tfidf_out = "### TF-IDF Results (Keyword)\n\n"
    for i, (_, row) in enumerate(r_tfidf.iterrows(), 1):
        tfidf_out += f"**{i}. {row['Ticket Type']}** "
        tfidf_out += f"[{row['Ticket Priority']}] "
        tfidf_out += f"Score: `{row['sim_tfidf']:.4f}`\n\n"
        tfidf_out += f"> {str(row['Ticket Description'])[:150]}...\n\n"

    # GloVe output
    glove_out = "### GloVe Results (Semantic)\n\n"
    for i, (_, row) in enumerate(r_glove.iterrows(), 1):
        glove_out += f"**{i}. {row['Ticket Type']}** "
        glove_out += f"[{row['Ticket Priority']}] "
        glove_out += f"Score: `{row['sim_glove']:.4f}`\n\n"
        glove_out += f"> {str(row['Ticket Description'])[:150]}...\n\n"

    # Hybrid output
    hybrid_out  = f"### Predicted Ticket Type: `{predicted_type}`\n\n"
    hybrid_out += f"### Top 3 Similar Past Tickets (α={alpha:.2f})\n\n"
    for i, (_, row) in enumerate(r_hybrid.iterrows(), 1):
        hybrid_out += f"**{i}. {row['Ticket Type']}** "
        hybrid_out += f"[{row['Ticket Priority']} | {row['Ticket Channel']}] "
        hybrid_out += f"Score: `{row['score']:.4f}`\n\n"
        hybrid_out += f"> {str(row['Ticket Description'])[:200]}...\n\n"

    return tfidf_out, glove_out, hybrid_out

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="HSRIS — Hybrid Support Ticket Search") as demo:
    gr.Markdown("# HSRIS — Hybrid Semantic Retrieval & Intelligence System")
    gr.Markdown("Adjust **α** to blend keyword (TF-IDF) and semantic (GloVe) search.")

    query_input = gr.Textbox(
        lines=3,
        placeholder="Describe your issue here...",
        label="Ticket Description"
    )
    alpha_slider = gr.Slider(
        minimum=0.0, maximum=1.0,
        value=0.5, step=0.05,
        label="α  —  0.0 = GloVe only  |  1.0 = TF-IDF only"
    )
    search_btn = gr.Button("Search", variant="primary")

    with gr.Row():
        tfidf_out = gr.Markdown(label="TF-IDF")
        glove_out = gr.Markdown(label="GloVe")

    hybrid_out = gr.Markdown(label="Hybrid Result")

    gr.Examples(
        examples=[
            ["my payment failed and I was charged twice", 0.5],
            ["laptop screen keeps going dark randomly",   0.3],
            ["cannot log into my account",               0.7],
            ["I want my money back",                     0.0],
            ["software keeps crashing at startup",       0.5],
        ],
        inputs=[query_input, alpha_slider]
    )

    search_btn.click(
        fn=compare_search,
        inputs=[query_input, alpha_slider],
        outputs=[tfidf_out, glove_out, hybrid_out]
    )

demo.launch()