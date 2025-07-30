FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install curl and uv
RUN apt-get update && apt-get install -y curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the container
COPY . /app

# Create virtual environment using uv
RUN uv venv /app/.venv

# Set virtual environment path
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies using uv inside the venv
RUN uv pip install -r requirements.txt

# Expose default Streamlit port
EXPOSE 8506

# Start the app with venv's Python
CMD ["streamlit", "run", "app.py", "--server.port=8506", "--server.address=0.0.0.0"]
