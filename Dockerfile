# -----------------------------
# Base Image
# -----------------------------
FROM python:3.12-slim

# -----------------------------
# Environment Variables
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# Set Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python Dependencies
# -----------------------------
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy Application Code
# -----------------------------
COPY backend ./backend
COPY docs ./docs

# -----------------------------
# Expose Port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Run the Application
# -----------------------------
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]