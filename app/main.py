from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(
    title="SmartChat AI Gateway",
    description="Gateway de atendimento inteligente com RAG e Roteamento de Intenções para a Blip",
    version="1.0.0"
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

faq_documents = [
    Document(
        page_content="Os planos da API variam conforme o volume mensal de mensagens. O plano Standard inclui até 10 mil interações por mês.",
        metadata={"intent": "duvida_comercial_rag"},
    ),
    Document(
        page_content="O plano Standard permite até 10 mil interações por mês. Limites de requisições por minuto podem variar conforme o plano contratado.",
        metadata={"intent": "duvida_limite_api_rag"},
    ),
    Document(
        page_content="Como exemplo, o suporte humano funciona de segunda a sexta-feira, das 9h às 18h, no horário de Brasília, exceto feriados.",
        metadata={"intent": "duvida_suporte_rag"},
    ),
]

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vector_store = Chroma.from_documents(
    documents=faq_documents,
    embedding=embedding_function,
    collection_name="smartchat_faq",
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    intent: str
    response: str
    transferred_to_human: bool

@app.get("/", response_class=HTMLResponse)
def health_check():
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text(encoding="utf-8"))

@app.post("/chat", response_model=ChatResponse)
def process_chat(payload: ChatRequest):
    msg_lower = payload.message.lower()

    if any(word in msg_lower for word in ["atendente", "humano", "especialista", "falar com pessoa", "reclamação grave"]):
        return {
            "intent": "transbordamento_humano",
            "response": "Compreendo. Estou transferindo seu atendimento para um de nossos especialistas humanos agora mesmo. Um momento.",
            "transferred_to_human": True
        }

    matches = vector_store.similarity_search_with_relevance_scores(payload.message, k=1)
    if matches and matches[0][1] >= 0.3:
        document, _score = matches[0]
        return {
            "intent": document.metadata["intent"],
            "response": document.page_content,
            "transferred_to_human": False,
        }

    return {
        "intent": "duvida_geral_llm",
        "response": f"Olá! Recebi sua mensagem: '{payload.message}'. Como assistente virtual inteligente, estou aqui para otimizar seu atendimento.",
        "transferred_to_human": False,
    }