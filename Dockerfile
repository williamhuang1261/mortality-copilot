# Builds the FastAPI service only (pipeline/api.py). The R pipeline and the
# RAG stack are not needed to serve the API: it reads the already-committed
# artifacts/cases.json and artifacts/model_card.json, exactly as
# pipeline/agent.py's CLI does.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ pipeline/
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "pipeline.api:app", "--host", "0.0.0.0", "--port", "8000"]
