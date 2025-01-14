# Use Python as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit
RUN pip install streamlit

# Set the command to run the sequence of scripts
CMD ["sh", "-c", "python flask_llm_integration/preprocess.py && python flask_llm_integration/ollama_llama_library.py && streamlit run frontend/chatbot_ui.py"]
