# SmartChat AI Gateway (RAG + Intent Router)

Microsserviço de atendimento conversacional inteligente construído em Python com **FastAPI**, simulando a arquitetura de roteamento de mensagens, busca semântica em base vetorial (**RAG**) e transbordo humano para automação de comunicação corporativa.

## 🚀 Tecnologias Utilizadas
- **Python 3.10** & **FastAPI**
- **Docker** & **Docker Compose**
- **LangChain** / **ChromaDB** (Busca semântica & Embeddings)
- **OpenAPI / Swagger** (Documentação interativa)

## 📌 Funcionalidades
1. **Classificação e Roteamento de Intenções:** Identificação automatizada do objetivo do cliente.
2. **RAG (Retrieval-Augmented Generation):** Consulta em base vetorial técnica para eliminar alucinações.
3. **Transbordo Humano:** Redirecionamento automático de chamados críticos ou complexos.

## 🏃 Como Executar
```bash
docker compose up --build
```

Acesse a documentação interativa em: `http://localhost:8000/docs`
