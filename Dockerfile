FROM python:3.11-slim

WORKDIR /app

# System dependency required by LightGBM at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch from the CPU-only wheel index FIRST — avoids pip pulling
# in the much larger default CUDA-enabled build. pip will see torch is
# already satisfied when it processes requirements.txt next, as long as
# the version there is compatible (torch==2.2.0).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.0
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects $PORT at runtime — do not hardcode 8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
