import streamlit as st
import sys
import pickle
import numpy as np
import faiss
import torch
import os

st.set_page_config(page_title="EVA — Enterprise Knowledge Assistant", page_icon="🧭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .eva-title { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #F4F6FA; }
#MainMenu, footer, header { visibility: hidden; }
.eva-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #3D5A80 55%, #5C7FA8 100%);
    padding: 32px 36px; border-radius: 20px; margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(27, 42, 74, 0.18);
}
.eva-title { color: white; font-size: 30px; font-weight: 700; margin: 0; }
.eva-subtitle { color: #D7E1F0; font-size: 15px; margin-top: 6px; }
.welcome-card {
    background: white; border-radius: 20px; padding: 36px; text-align: center;
    border: 1px solid #E8ECF3; box-shadow: 0 4px 16px rgba(27,42,74,0.05); margin-bottom: 20px;
}
.welcome-avatar {
    width: 64px; height: 64px; background: linear-gradient(135deg, #E8A33D, #C97B2E);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: 30px; margin: 0 auto 14px auto; box-shadow: 0 6px 18px rgba(232,163,61,0.35);
}
.welcome-heading { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: #1B2A4A; margin-bottom: 6px; }
.welcome-text { color: #6B7280; font-size: 14px; max-width: 460px; margin: 0 auto; }
.chat-row { display: flex; align-items: flex-start; gap: 10px; margin: 14px 0; }
.chat-row.user-row { flex-direction: row-reverse; }
.avatar-circle { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
.avatar-eva { background: linear-gradient(135deg, #E8A33D, #C97B2E); color: white; }
.avatar-user { background: #1B2A4A; color: white; }
.chat-bubble-user {
    background: #1B2A4A; color: white; padding: 12px 18px; border-radius: 16px 16px 4px 16px;
    max-width: 70%; font-size: 15px; box-shadow: 0 2px 8px rgba(27,42,74,0.15);
}
.chat-bubble-eva {
    background: white; color: #2A2E35; padding: 14px 20px; border-radius: 16px 16px 16px 4px;
    max-width: 75%; border: 1px solid #E8ECF3; font-size: 15px; line-height: 1.6;
    box-shadow: 0 2px 10px rgba(27,42,74,0.06);
}
.domain-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700;
    letter-spacing: 0.4px; text-transform: uppercase; margin-bottom: 10px;
}
.badge-HR { background: #E8F5E9; color: #2E7D32; }
.badge-Legal { background: #EDE7F6; color: #5E35B1; }
.badge-Finance { background: #FFF3E0; color: #E65100; }
.badge-IT { background: #E3F2FD; color: #1565C0; }
.sidebar-stat { background: white; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid #E8ECF3; }
.sidebar-stat-num { font-size: 22px; font-weight: 800; color: #1B2A4A; font-family: 'Space Grotesk', sans-serif; }
.sidebar-stat-label { font-size: 12px; color: #6B7280; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading EVA — this takes a minute on first run...")
def load_eva_system():
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    from transformers import (
        DistilBertTokenizer, DistilBertForSequenceClassification,
        AutoTokenizer, AutoModelForQuestionAnswering,
    )

    ASSETS_DIR = snapshot_download(repo_id="irthayag/eva-assistant-assets", repo_type="dataset")
    sys.path.append(f"{ASSETS_DIR}/src")

    device = "cpu"

    index = faiss.read_index(f"{ASSETS_DIR}/embeddings/faiss_index.bin")
    with open(f"{ASSETS_DIR}/embeddings/all_chunks_metadata.pkl", "rb") as f:
        all_chunks = pickle.load(f)

    domain_indices = {}
    domain_folder = f"{ASSETS_DIR}/embeddings/domain_indices"
    for domain in ["HR", "Legal", "Finance", "IT"]:
        domain_indices[domain] = faiss.read_index(f"{domain_folder}/{domain}_index.bin")
    with open(f"{domain_folder}/domain_chunks_map.pkl", "rb") as f:
        domain_chunks_map = pickle.load(f)

    embedding_model = SentenceTransformer(f"{ASSETS_DIR}/models/embedding_model", device=device)
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    router_path = f"{ASSETS_DIR}/models/distilbert_router"
    tokenizer = DistilBertTokenizer.from_pretrained(router_path)
    router_model = DistilBertForSequenceClassification.from_pretrained(router_path)
    router_model.eval()
    with open(f"{router_path}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    qa_model.eval()

    from rag_pipeline import generate_with_groq
    from router_utils import classify_query
    from hybrid_retrieval import retrieve_hybrid
    from extractive_qa import extract_exact_answer
    from eva_chatbot import ask_eva

    return {
        "index": index, "all_chunks": all_chunks, "domain_indices": domain_indices,
        "domain_chunks_map": domain_chunks_map, "embedding_model": embedding_model,
        "groq_client": groq_client, "router_model": router_model, "tokenizer": tokenizer,
        "label_encoder": label_encoder, "qa_model": qa_model, "qa_tokenizer": qa_tokenizer,
        "generate_with_groq": generate_with_groq, "classify_query": classify_query,
        "retrieve_hybrid": retrieve_hybrid, "extract_exact_answer": extract_exact_answer,
        "ask_eva": ask_eva,
    }

sys_components = load_eva_system()

for key, default in [("conversation_history", []), ("display_messages", []), ("pending_question", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### 🧭 EVA")
    st.caption("Enterprise Virtual Assistant — Demo")
    st.markdown("---")
    st.markdown('<div class="sidebar-stat"><div class="sidebar-stat-num">243,847</div><div class="sidebar-stat-label">Knowledge chunks indexed</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat"><div class="sidebar-stat-num">4</div><div class="sidebar-stat-label">Domains — HR · Legal · Finance · IT</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.display_messages = []
        st.rerun()

st.markdown("""
<div class="eva-header">
    <p class="eva-title">🧭 EVA</p>
    <p class="eva-subtitle">Enterprise Virtual Assistant — ask about HR, Legal, Finance, or IT</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.display_messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-avatar">🧭</div>
        <div class="welcome-heading">Hi, I'm EVA</div>
        <div class="welcome-text">Ask me anything about HR policies, legal contracts, finance reports, or IT support.</div>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "What is the maternity leave policy?",
        "Is there a non-compete clause in the contracts?",
        "How do I fix a VPN connection issue?",
        "What was the company's revenue?",
    ]
    cols = st.columns(4)
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_question = s
                st.rerun()

for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        st.markdown(f'''
        <div class="chat-row user-row">
            <div class="avatar-circle avatar-user">🙂</div>
            <div class="chat-bubble-user">{msg["content"]}</div>
        </div>''', unsafe_allow_html=True)
    else:
        badge_html = ""
        if msg.get("domain"):
            badge_html = f'<span class="domain-badge badge-{msg["domain"]}">{msg["domain"]}</span><br>'
        st.markdown(f'''
        <div class="chat-row">
            <div class="avatar-circle avatar-eva">🧭</div>
            <div class="chat-bubble-eva">{badge_html}{msg["content"]}</div>
        </div>''', unsafe_allow_html=True)
        if msg.get("exact_quote"):
            with st.expander("📄 Verified source excerpt"):
                st.markdown(f"*\"{msg['exact_quote']}\"*")
                st.caption(f"Source: {msg.get('source', 'N/A')}")

user_input = st.chat_input("Ask EVA about HR, Legal, Finance, or IT...")
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.spinner("EVA is thinking..."):
        result = sys_components["ask_eva"](
            user_input, st.session_state.conversation_history,
            sys_components["embedding_model"], sys_components["index"], sys_components["all_chunks"],
            sys_components["groq_client"], sys_components["router_model"], sys_components["tokenizer"],
            sys_components["label_encoder"], sys_components["domain_indices"], sys_components["domain_chunks_map"],
            sys_components["qa_model"], sys_components["qa_tokenizer"],
            sys_components["classify_query"], sys_components["retrieve_hybrid"],
            sys_components["generate_with_groq"], sys_components["extract_exact_answer"],
        )
    st.session_state.display_messages.append({
        "role": "eva", "content": result["answer"],
        "domain": result.get("domain"), "exact_quote": result.get("exact_quote"),
        "source": result.get("source"),
    })
    st.rerun()
