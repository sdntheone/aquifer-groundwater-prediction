# Dockerfile — Groundwater Prediction Service
# Build:  docker build -t groundwater-app .
# Run:    docker run -p 5000:5000 groundwater-app

FROM python:3.11-slim

WORKDIR /app

# system deps needed by pandas/numpy wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model and generate prediction artifacts at build time so the
# image is self-contained (no training needed at container start).
RUN python train.py && python predict.py

ENV PORT=5000
EXPOSE 5000

# gunicorn for production-grade serving (vs the Flask dev server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
