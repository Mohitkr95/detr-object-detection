# Stage 1: Build the application
FROM python:3.8 AS builder

# Set Python to not create .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

# Install uv using pip
RUN pip install uv

# Copy and install dependencies
COPY docker-requirements.txt .
RUN uv pip install --system --no-cache-dir -r docker-requirements.txt
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system timm==0.9.2

COPY . .

EXPOSE 8080