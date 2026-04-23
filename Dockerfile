FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir streamlit plotly scikit-learn joblib pandas numpy==1.26.4 scipy \
    && pip cache purge

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.fileWatcherType=none"]
