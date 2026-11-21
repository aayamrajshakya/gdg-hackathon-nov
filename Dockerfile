# SentinelDesk console for Google Cloud Run (CPU). Generation uses Gemma on the Gemini API,
# so the container only needs EmbeddingGemma (CPU) + the prebuilt artifacts.
#   uv run python scripts/build_all.py && uv run python scripts/export_embedder.py
#   gcloud run deploy sentineldesk --source . --region us-central1 --memory 4Gi --cpu 2 \
#       --set-secrets GEMINI_API_KEY=gemini-api-key:latest --allow-unauthenticated
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SD_EMBED_MODEL=/app/artifacts/embeddinggemma \
    SD_LLM_BACKEND=gemini \
    HF_HUB_OFFLINE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY sentineldesk/ sentineldesk/
COPY scripts/ scripts/
COPY app.py .
COPY .streamlit/ .streamlit/
COPY artifacts/kb.parquet artifacts/kb_emb.npy artifacts/triage_model.joblib artifacts/
COPY artifacts/embeddinggemma/ artifacts/embeddinggemma/

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]
