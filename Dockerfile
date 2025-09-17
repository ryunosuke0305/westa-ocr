# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

RUN mkdir -p /data/uploads /data/jobs /data/responses /data/keys
VOLUME ["/data/uploads", "/data/jobs", "/data/responses", "/data/keys"]

WORKDIR /app/backend

EXPOSE 5000

CMD ["python", "app.py"]
