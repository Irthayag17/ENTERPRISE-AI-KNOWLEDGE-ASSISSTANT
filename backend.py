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
ASSETS_DIR = snapshot_download(repo_id="irthayag/eva-assistant-assets", repo_type="dataset")
print(f"Assets downloaded to: {ASSETS_DIR}")

sys.path.append(f"{ASSETS_DIR}/src")

from sentence_transformers import SentenceTransformer
from groq import Groq
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model.eval()

from rag_pipeline import generate_with_groq
from router_utils import classify_query
from hybrid_retrieval import retrieve_hybrid
from extractive_qa import extract_exact_answer
from eva_chatbot import ask_eva

print("EVA system loaded successfully.")

conversations = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    domain: str | None = None
    exact_quote: str | None = None
    source: str | None = None

@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": index.ntotal}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in conversations:
        conversations[req.session_id] = []

    result = ask_eva(
        req.message, conversations[req.session_id],
        embedding_model, index, all_chunks, groq_client,
        router_model, tokenizer, label_encoder, domain_indices, domain_chunks_map,
        qa_model, qa_tokenizer, classify_query, retrieve_hybrid, generate_with_groq, extract_exact_answer
    )
    return ChatResponse(
        answer=result['answer'],
        domain=result.get('domain'),
        exact_quote=result.get('exact_quote'),
        source=result.get('source')
    )

@app.post("/api/reset")
def reset(req: ChatRequest):
    conversations[req.session_id] = []
    return {"status": "reset"}

# Serve the React frontend (must be mounted AFTER all /api routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
