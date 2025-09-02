FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy only the API folder
COPY sentiment-sphere/api/ .

# Install Python dependencies without cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import sys; print('Python', sys.version)" && \
    pip cache purge || true

# Create data directories
RUN mkdir -p data/search_cache data/analyses

# Expose port (Render/Railway will provide $PORT at runtime)
EXPOSE 8000

# Run the API (bind to provided $PORT if set)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
