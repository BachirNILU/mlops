# The Turbine Assistant API (script 15) in a container. Mock mode by default.
FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LAB_MODE=mock
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lab ./lab
COPY scripts ./scripts
COPY eval ./eval
RUN python scripts/00_make_data.py
# run as a non-root user
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "scripts/15_api.py"]
