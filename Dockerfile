# AI-XAUUSD Trading System Dockerfile

FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Create directories for models and data
RUN mkdir -p models config data logs

# Expose port for web interface (if implemented)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "quick_demo.py"]

# Labels
LABEL maintainer="JonusNattapong / Zombitx64" \
      description="AI-Driven XAUUSD Trading System with Ensemble RL" \
      version="1.0.0" \
      repository="https://github.com/JonusNattapong/AI-XAUUSD-Trading" \
      license="MIT"