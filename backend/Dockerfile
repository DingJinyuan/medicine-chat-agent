FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model and the triage classifier's base model into the
# image so the first request doesn't pay a cold-start download against
# Hugging Face Hub. fastembed's cache_dir is explicit and outside /tmp --
# some platforms (Render included) mount a fresh /tmp at container runtime,
# which would otherwise hide whatever got cached here during the build.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/.fastembed_cache')"
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)"

COPY app ./app
COPY evals ./evals
COPY finetuning/artifacts ./finetuning/artifacts
COPY data ./data

EXPOSE 8000
ENV PYTHONUNBUFFERED=1
# Models are already baked into the image; skip the Hub version-check HEAD
# request on every startup.
ENV HF_HUB_OFFLINE=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
