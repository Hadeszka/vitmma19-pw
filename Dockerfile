# Demo Dockerfile: minimal setup for the course template.
# IMPORTANT: This Dockerfile is a simple example for demonstration purposes
# and must be adapted to your project topic (dependencies, system packages,
# runtime, GPU support, volumes, entrypoint behaviour etc.).

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and notebooks
COPY src/ src/
COPY notebook/ notebook/
COPY run.sh run.sh
RUN sed -i 's/\r$//' /app/run.sh

# Create a directory for data (to be mounted)
RUN chmod +x /app/run.sh || true

# Set the entrypoint to run the training script by default
# You can override this with `docker run ... python src/04-inference.py` etc.
CMD ["bash", "/app/run.sh"]
