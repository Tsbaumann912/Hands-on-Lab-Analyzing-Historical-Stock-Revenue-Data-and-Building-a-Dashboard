FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD gunicorn wsgi:server --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120
