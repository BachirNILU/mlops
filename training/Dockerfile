FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY training/requirements.txt /app/training/requirements.txt

RUN pip install --no-cache-dir -r /app/training/requirements.txt

COPY training/ /app/training/

CMD ["python", "training/train.py"]
