FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.9.0+cpu torchvision==0.24.0+cpu torchaudio==2.9.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
