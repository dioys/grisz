FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY engine engine
COPY agents agents
COPY api api
COPY worker worker
COPY static static
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
