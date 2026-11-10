from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="SmartChat AI Gateway",
    description="Gateway de atendimento inteligente com RAG e Roteamento de Intenções para a Blip",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    intent: str
    response: str
    transferred_to_human: bool

@app.get("/")
def health_check():
    return {"status": "online", "service": "SmartChat AI Gateway running successfully"}

@app.post("/chat", response_model=ChatResponse)
def process_chat(payload: ChatRequest):
    msg_lower = payload.message.lower()

    if any(word in msg_lower for word in ["atendente", "humano", "falar com pessoa", "reclamação grave"]):
        return {
            "intent": "transbordamento_humano",
            "response": "Compreendo. Estou transferindo seu atendimento para um de nossos especialistas humanos agora mesmo. Um momento.",
            "transferred_to_human": True
        }

    elif any(word in msg_lower for word in ["preço", "plano", "custo", "assinatura"]):
        return {
            "intent": "duvida_comercial_rag",
            "response": "[Base Vetorial RAG] Nossos planos de API variam de acordo com o volume de mensagens mensais. O plano Standard cobre até 10 mil interações.",
            "transferred_to_human": False
        }

    else:
        return {
            "intent": "duvida_geral_llm",
            "response": f"Olá! Recebi sua mensagem: '{payload.message}'. Como assistente virtual inteligente, estou aqui para otimizar seu atendimento.",
            "transferred_to_human": False
        }