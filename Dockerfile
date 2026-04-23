FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code.
COPY . .

# Streamlit default port.
EXPOSE 8501

# Keep cache in a writable, persistent-friendly folder inside container.
ENV FASTF1_CACHE_DIR=/app/cache

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
