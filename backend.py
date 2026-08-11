from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys, pickle, numpy as np, faiss, torch, os
from huggingface_hub import snapshot_download

app = FastAPI(title="EVA API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Downloading EVA assets from HuggingFace Hub...")
ASSETS_DIR = snapshot_download(
    repo_id="irthayag/eva-assistant-assets",
    repo_type="dataset",
    allow_patterns=["embeddings/domain_indices/*", "embeddings/faiss_index.bin",
                     "embeddings/all_chunks_metadata.pkl", "models/embedding_model/*",
                     "models/distilbert_router/*", "src/*"]
)
print(f"Assets downloaded to: {ASSETS_DIR}")

sys.path.append(f"{ASSETS_DIR}/src")

from sentence_transformers import SentenceTransformer
from groq import Groq
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

device = 'cpu'

index = faiss.read_index(f"{ASSETS_DIR}/embeddings/faiss_index.bin")
with open(f"{ASSETS_DIR}/embeddings/all_chunks_metadata.pkl", 'rb') as f:
    all_chunks = pickle.load(f)

domain_indices = {}
domain_faiss_folder = f"{ASSETS_DIR}/embeddings/domain_indices"
for domain in ['HR', 'Legal', 'Finance', 'IT']:
    domain_indices[domain] = faiss.read_index(f"{domain_faiss_folder}/{domain}_index.bin")
with open(f"{domain_faiss_folder}/domain_chunks_map.pkl", 'rb') as f:
    domain_chunks_map = pickle.load(f)

embedding_model = SentenceTransformer(f"{ASSETS_DIR}/models/embedding_model", device=device)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

router_path = f"{ASSETS_DIR}/models/distilbert_router"
tokenizer = DistilBertTokenizer.from_pretrained(router_path)
router_model = DistilBertForSequenceClassification.from_pretrained(router_path)
router_model.eval()
with open(f"{router_path}/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

from rag_pipeline import generate_with_groq
from router_utils import classify_query
from hybrid_retrieval import retrieve_hybrid

print("EVA (lightweight) system loaded successfully.")

conversations = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    domain: str | None = None

def is_small_talk(query):
    prompt = f"""Is this small talk/greeting (like "hi", "hello", "thanks")? Answer ONLY "yes" or "no".
Message: {query}"""
    return "yes" in generate_with_groq(prompt, groq_client).strip().lower()

@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": index.ntotal}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in conversations:
        conversations[req.session_id] = []
    history = conversations[req.session_id]

    if is_small_talk(req.message):
        answer = generate_with_groq(
            f"You are EVA, a friendly enterprise assistant for HR, Legal, Finance, IT. Respond briefly to: {req.message}",
            groq_client
        )
        history.append({'question': req.message, 'answer': answer})
        return ChatResponse(answer=answer, domain=None)

    domain, _ = classify_query(req.message, router_model, tokenizer, label_encoder)

    history_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in history[-3:]])
    rewrite_prompt = f"""Given this conversation history:\n{history_text}\n\nRewrite the NEW question to be a clear, standalone search query, resolving references to earlier context. Keep it short.\n\nNew question: {req.message}\n\nRewritten:"""
    rewritten = generate_with_groq(rewrite_prompt, groq_client).strip()

    results = retrieve_hybrid(rewritten, domain, embedding_model, domain_indices, domain_chunks_map, index, all_chunks, top_k=5)
    best_distance = results[0][0]

    if best_distance > 0.95:
        answer = "I couldn't find information about this in my knowledge base. Could you rephrase, or ask about HR, Legal, Finance, or IT topics?"
        history.append({'question': req.message, 'answer': answer})
        return ChatResponse(answer=answer, domain=domain)

    context_text = "\n\n".join([f"[Source: {c['domain']} - {c['title']}]\n{c['text']}" for dist, c in results])
    prompt = f"""You are EVA, an enterprise knowledge assistant. Answer using ONLY the context below. Directly explain what it says.

Context:
{context_text}

Question: {req.message}

Answer:"""
    answer = generate_with_groq(prompt, groq_client)
    history.append({'question': req.message, 'answer': answer})
    return ChatResponse(answer=answer, domain=domain)

@app.post("/api/reset")
def reset(req: ChatRequest):
    conversations[req.session_id] = []
    return {"status": "reset"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
