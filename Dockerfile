# Stage 1: Build the application
FROM python:3.8 AS builder

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

# Copy and install dependencies
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install timm==0.9.2

COPY . .

EXPOSE 8080