import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import util, SentenceTransformer

st.set_page_config(page_title="NutriBot AI", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

st.title("NutriBot AI - Nutrition Research Assistant")
st.write("Ask questions about nutrition and get accurate answers from research documents.")  # plain text

@st.cache_resource
def load_model_and_embeddings():
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

    # Data
    df = pd.read_csv("text_chunks_and_embeddings_df.csv")

    # Parse embeddings from scientific-notation strings like: "[ 6.7424e-02 ... ]"
    def parse_emb(s):
        if isinstance(s, str):
            return np.fromstring(s.strip("[]"), sep=" ")
        return np.asarray(s, dtype=float)

    df["embedding"] = df["embedding"].apply(parse_emb)

    # Torch tensor
    emb = torch.tensor(np.stack(df["embedding"].to_list()), dtype=torch.float32, device=device)
    return embedding_model, df, emb, device

def retrieve_relevant_resources(query, emb_tensor, model, top_k=2):
    q = model.encode(query, convert_to_tensor=True)
    scores = util.dot_score(q, emb_tensor)[0]
    vals, idx = torch.topk(scores, k=top_k)
    return vals.tolist(), idx.tolist()

def build_prompt(query, ctx_items):
    ctx_text = "- " + "\n- ".join([it["sentence_chunk"] for it in ctx_items])
    return (
         "Based on the following context items, please answer the query.\n"
        "Extract relevant passages from the context before answering the query. "
        "Do not include your reasoning, only return the final answer. "
        "Use the retrieved information from the RAG system to create a user-friendly, precise answer (around 30 words).\n\n"
        f"Context items:\n{ctx_text}\n\nQuery: {query}\n\nAnswer:"
    )

# Load
try:
    with st.spinner("Loading models and embeddings..."):
        embedding_model, chunks_df, embeddings, DEVICE = load_model_and_embeddings()
    st.write("Models loaded successfully!")  # no status icon
except Exception as e:
    # Use write to avoid red error icon; comment next line if you prefer st.error
    st.write(f"Error loading models: {e}")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=None):
        st.text(msg["content"])  # strict plain text

# Input
if user_query := st.chat_input("Ask about nutrition..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=None):
        st.text(user_query)

    # Generate assistant reply
    with st.chat_message("assistant", avatar=None):
        with st.spinner("Thinking..."):
            try:
                scores, indices = retrieve_relevant_resources(user_query, embeddings, embedding_model, top_k=5)
                ctx_items = [chunks_df.iloc[i].to_dict() for i in indices]

                # Make a plain-text answer (no emojis, no markdown)
                if ctx_items:
                    top_text = ctx_items[0]["sentence_chunk"]
                    ans = (
                        f"Retrieved {len(ctx_items)} relevant passages from the document.\n\n"
                        f"Top relevant passage (Score: {scores[0]:.4f}):\n\n"
                        f"{top_text}"
                    )
                else:
                    ans = "No relevant passages found."

                st.text(ans)  # strict plain text output
            except Exception as e:
                ans = f"Error: {e}"
                st.text(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})

# Optional: remove sidebar entirely by deleting this block
with st.sidebar:
    st.header("About")
    st.write("RAG-based AI assistant for nutrition research.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
