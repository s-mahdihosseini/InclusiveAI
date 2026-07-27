FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scenarios/rubric.json scenarios/scenarios.json ./scenarios/
COPY scenarios/extractions/ ./scenarios/extractions/
COPY models/expertise/static/Counterfactual.dta models/expertise/static/occupation_titles.csv ./models/expertise/static/

WORKDIR /app/backend

# HF Spaces uses 7860; Render/Cloud Run inject $PORT
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
