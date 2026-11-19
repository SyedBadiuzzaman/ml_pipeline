FROM python:3.12-slim

# XGBoost requires the OpenMP runtime library (libgomp1), which slim images omit.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs Docker containers as non-root user UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Install runtime Python dependencies first so Docker caches this layer.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files needed to serve predictions.
COPY --chown=user app/ app/
COPY --chown=user pipeline/__init__.py pipeline/process_split.py pipeline/
COPY --chown=user model/ model/
COPY --chown=user data/synthetic_occupancy.csv data/

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
