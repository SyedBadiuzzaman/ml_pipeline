FROM python:3.12-slim

# Copy the fast uv binary for quick, reliable dependency installation.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# XGBoost requires the OpenMP runtime library (libgomp1), which slim images omit.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install runtime Python dependencies into system site-packages using uv.
COPY requirements.txt /tmp/requirements.txt
RUN UV_HTTP_TIMEOUT=120 uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Hugging Face Spaces runs Docker containers as non-root user UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Copy only the files needed to serve predictions.
COPY --chown=user app/ app/
COPY --chown=user pipeline/__init__.py pipeline/process_split.py pipeline/
COPY --chown=user model/ model/
COPY --chown=user data/synthetic_occupancy.csv data/

EXPOSE 7860
# Use Render/cloud $PORT if provided, otherwise default to 7860 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
