# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by whisper
# Install ffmpeg for audio processing and rust for the tokenizer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust compiler
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# Add rust to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This will now install the correct 'openai-whisper'
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Expose the port the app runs on
EXPOSE 5066

# Define the command to run your app
CMD ["python", "src/app.py"]