FROM python:3.11-slim

WORKDIR /app

# Copy only the API folder
COPY sentiment-sphere/api/ .

# Install Python dependencies
RUN pip install -r requirements.txt

# Create data directories
RUN mkdir -p data/search_cache data/analyses

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "main.py"]