FROM python:3.11.4-slim-bookworm

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app
WORKDIR /app
RUN python3 weights/download_weights.py

EXPOSE 8000
CMD [ "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000" ]


