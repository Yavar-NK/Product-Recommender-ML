# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first to leverage Docker cache layers
COPY requirements.txt .

# Install Python dependencies using a high-speed mirror to bypass network drops
RUN pip install --no-cache-dir --upgrade pip -i https://mirror-pypi.runflare.com/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://mirror-pypi.runflare.com/simple

# Copy the rest of the application code and pretrained models
COPY src/ ./src/
COPY models/ ./models/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]