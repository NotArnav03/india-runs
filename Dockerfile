# Self-contained sandbox image (submission spec §10.5 "docker run" option).
# Builds the ranker + Streamlit demo; runs CPU-only.
#
#   docker build -t redrob-ranker .
#   docker run -p 8501:8501 redrob-ranker          # Streamlit sandbox UI
#   # or reproduce a submission from a mounted candidates file:
#   docker run -v "$PWD:/data" redrob-ranker \
#       python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps kept minimal; sentence-transformers/torch wheels are CPU-only.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-fetch the SBERT model at build time so the running container is offline.
RUN python prepare.py || echo "model prefetch skipped (no network at build); will fetch on first run"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
