# Use official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit for running the frontend UI
RUN pip install streamlit

# Copy the rest of the project into the container
COPY . .

# Expose port 8501 for Streamlit (default port)
EXPOSE 8501

# Run watcher.py first, then app.py, and finally chatbot_ui.py using Streamlit
CMD ["bash", "-c", "python handlers/watcher.py && python api/app.py && streamlit run frontend/chatbot_ui.py"]
