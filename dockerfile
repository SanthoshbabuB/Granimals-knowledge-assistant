
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY ui ./ui
COPY pyproject.toml .

RUN mkdir -p data/documents \
    data/assets \
    data/processed

EXPOSE 8501
EXPOSE 8000

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]