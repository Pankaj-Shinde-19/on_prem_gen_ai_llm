# Use official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required Python packages, excluding pywin32 if present
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y pywin32

# Install Streamlit for running the frontend UI
RUN pip install streamlit

# Expose port 8501 for Streamlit (default port)
EXPOSE 8501

# Run watcher.py first, then app.py, and finally chatbot_ui.py using streamlit
CMD ["bash", "-c", "python handlers/watcher.py && python api/app.py && streamlit run frontend/chatbot_ui.py"]
