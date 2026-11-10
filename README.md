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

Para habilitar respostas flexíveis com IA, copie `.env.example` para `.env` e
preencha `OPENAI_API_KEY` com uma chave válida. O modelo padrão
é `gpt-4o-mini`; ele pode ser alterado em `OPENAI_MODEL`. Sem a chave, o serviço
continua disponível com as respostas da FAQ local e informa essa limitação para
perguntas abertas.

```bash
cp .env.example .env
# Edite .env e substitua OPENAI_API_KEY pela sua chave
docker compose up --build
```

Acesse a documentação interativa em: `http://localhost:8000/docs`
