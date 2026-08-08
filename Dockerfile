FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag_engine.py app.py ./

# Suppress tokenizers parallelism warning at the OS level
ENV TOKENIZERS_PARALLELISM=false

# Streamlit default port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
