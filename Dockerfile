FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 5000

# Run API server
CMD ["python", "src/api.py"]
