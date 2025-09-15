# Use Python slim image for smaller size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1


# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Remove build dependencies to reduce size
RUN apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /root/.cache/pip

# Expose fastapi port
EXPOSE 8000

# Run Streamlit
CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]