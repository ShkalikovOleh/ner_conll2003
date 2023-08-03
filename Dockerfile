FROM python:3.11.4-slim-bookworm

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY weights /app/weights
WORKDIR /app
RUN python3 weights/download_weights.py

COPY api /app/api

EXPOSE 8000
CMD [ "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000" ]


