# On Premise Generative AI (on_prem_gen_ai)

## Description
**On Premise Generative AI (on_prem_gen_ai)** is a secure and efficient Retrieval-Augmented Generation (RAG) system designed for enterprise use. The key differentiator of this project is its capability to run all required Large Language Models (LLMs) and libraries locally, ensuring maximum **data security** and **control over sensitive information**. 

This project includes:
- A **local deployment of LLMs** and libraries to process user queries.
- A **custom chatbot UI** for seamless interaction.
- An **automated PDF handling module** for processing documents in real time.

By running the application entirely offline, your data remains within your system, ensuring compliance with data security policies.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Key Features](#key-features)
4. [Project Hierarchy](#project-hierarchy)

---

## Installation
To get started, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Pankaj-Shinde-19/on_prem_gen_ai.git
   cd on_prem_gen_ai
2. Install dependencies:
   ```plaintext
   pip install -r requirements.txt
3. Install the Llama library using the following GitHub repository: [Ollama Python](https://github.com/ollama/ollama-python)
## Usage
Follow the steps below to use the application:

1. **Run the Watcher Script:**
   This monitors the pdfs directory for new uploads.
   ```plaintext
   python handlers/watcher.py
2. **Start the Flask API**:
   Ensure the backend is up and running.
    ```plaintext
    python api/app.py
3. **Run the Chatbot UI:**
   Launch the user interface for interactions.
    ```plaintext
    python frontend/chatbot_ui.py
4. **PDF Uploads:**
   Upload PDF files to the pdfs directory.
   The pdf_filehandler.py module processes them asynchronously.


## Key Features

1. **Data Security:**
   The entire application, including LLMs and libraries, runs locally, eliminating the need for an internet connection and ensuring sensitive data stays secure.

2. **Customizable Deployment:**
   Built for on-premise setups, making it ideal for enterprises with strict security policies.

3. **Retrieval-Augmented Generation (RAG):**
   Combines local document retrieval with advanced LLMs to provide accurate and context-aware responses.

4. **Asynchronous File Handling:**
    Automates the processing of PDF files for seamless integration into the chatbot.

5. **User-Friendly Interface:**
   Intuitive chatbot UI for easy interaction with the system.
## ChatBot UI
![image](https://github.com/user-attachments/assets/b553db5d-5bd1-436d-b8ed-4bf4605c1af3)

## Project Hierarchy
Here's an overview of the project's structure:
```plaintext
on_prem_gen_ai/
├── api/
│   ├── __init__.py
│   └── app.py
├── frontend/
│   ├── __init__.py
│   ├── chatbot_ui.py
│   ├── logo.png
│   └── style.css
├── handlers/
│   ├── __init__.py
│   ├── pdf_filehandler.py
│   └── watcher.py
├── pdfs/          # Directory for uploading PDF files
├── .gitignore
├── Dockerfile
├── requirements.txt

