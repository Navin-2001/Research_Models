FROM python:3.10-slim

# Install system dependencies (FFmpeg for audio processing, libGL for OpenCV/DeepFace)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/uploads

# Copy application files
COPY . .

# Start the FastAPI web server using Render's $PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
