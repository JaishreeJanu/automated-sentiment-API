# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (needed for some ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only the requirements first (optimizes Docker caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the model and source code
# We only need the processed model and the API code for serving
COPY models/ /app/models/
COPY src/ /app/src/

# 7. Expose the port FastAPI will run on
EXPOSE 8000

# 8. Command to run the API
# We use 0.0.0.0 so it's accessible outside the container
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]