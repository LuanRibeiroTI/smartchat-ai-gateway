# SmartChat AI Gateway

## Build and run

- Start the service with `docker compose up --build`; the API listens on port `8000`.
- Open the interactive API documentation at `http://localhost:8000/docs`.
- No test runner or lint commands are configured in this repository.

## Architecture

- `app/main.py` defines the FastAPI application, request/response models, and both routes. The root route serves `app/static/index.html`; the browser sends `POST /chat` with `user_id` and `message` and consumes `intent`, `response`, and `transferred_to_human`.
- The chat pipeline checks human-transfer keywords first, then searches the in-memory Chroma FAQ vector store, and finally returns a generic response when no FAQ match reaches the relevance threshold. The embedding model and vector store are initialized when the app module loads, so startup may download the Hugging Face model.
- FAQ content and its intent labels are defined together in `faq_documents` in `app/main.py`; the vector store is built from that list at startup.

## Repository conventions

- Keep chat responses aligned with the `ChatResponse` schema and the fields consumed by the browser UI.
- Preserve the ordering of human-transfer detection before FAQ retrieval so escalation requests are not answered by a similar FAQ.
- Each FAQ document's `metadata["intent"]` is the intent returned for a retrieval hit; update the metadata alongside its content when adding or changing FAQ entries.
