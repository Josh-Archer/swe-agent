# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./src /app/src

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for the default model
ENV OLLAMA_MODEL="llama2"

# Default durable feedback storage (mount a volume at /app/data in production)
ENV FEEDBACK_STORAGE_BACKEND="sqlite"
ENV FEEDBACK_STORAGE_PATH="/app/data/feedback.db"
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# Run the application when the container launches
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]