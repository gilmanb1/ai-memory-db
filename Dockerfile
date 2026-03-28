# Multi-stage build: frontend static export + Python backend

# Stage 1: Build frontend
FROM node:22-slim AS frontend
WORKDIR /app/dashboard/frontend
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json ./
RUN npm ci --production=false
COPY dashboard/frontend/ ./
# Configure for static export
RUN echo 'export default { output: "export", images: { unoptimized: true } }' > next.config.ts
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

# Stage 2: Python backend + demo data
FROM python:3.13-slim AS backend
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir fastapi uvicorn duckdb numpy

# Copy project
COPY memory/ ./memory/
COPY dashboard/backend/ ./dashboard/backend/
COPY demo/ ./demo/
COPY test_corpus.py ./
COPY test_memory.py ./

# Copy built frontend
COPY --from=frontend /app/dashboard/frontend/out ./dashboard/frontend/out

# Generate demo database
RUN python demo/seed_demo_db.py

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run demo server
CMD ["python", "demo/demo_server.py"]
