# Import necessary libraries
from flask import Flask, request, jsonify
import time
import pickle
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import chat, ChatResponse
import os
from flask_cors import CORS
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Disable symlink warnings from the Hugging Face library
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Flask app and enable Cross-Origin Resource Sharing (CORS)
app = Flask(__name__)
CORS(app)

# Initialize embedding and ranking models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device='cpu')

# Initialize Qdrant client for document vector database interaction
qdrant_client = QdrantClient(host="localhost", port=6333)


# Set up a deque to maintain conversation history (FIFO, max length 2)
conversation_history = deque(maxlen=2)

# Initialize a thread pool for asynchronous processing
executor = ThreadPoolExecutor(max_workers=4)

# ========================== Utility Functions ==========================

def get_query_embedding(query):
    """
    Generate embeddings for a given query using the embedding model.

    Args:
        query (str): Input query text.

    Returns:
        ndarray: Embedding vector for the query.
    """
    return embedding_model.encode([query])[0]

def retrieve_documents(query, top_k=5):
    """
    Retrieve top-k relevant documents for a query from Qdrant and rank them.

    Args:
        query (str): User query text.
        top_k (int): Number of top documents to retrieve.

    Returns:
        list: Ranked documents.
        dict: Timing breakdown for each step.
    """
    timings = {}
    start_retrieval = time.time()

    # Step 1: Generate embedding for the query
    query_vector = get_query_embedding(query)
    timings["embedding_time"] = time.time() - start_retrieval

    # Step 2: Perform vector search in Qdrant
    start_search = time.time()
    results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=top_k
    )
    timings["qdrant_search_time"] = time.time() - start_search

    # Step 3: Extract and rank retrieved documents
    candidate_docs = [result.payload["text"] for result in results]
    if candidate_docs:
        start_ranking = time.time()
        pairs = [[query, doc] for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)
        timings["ranking_time"] = time.time() - start_ranking

        # Rank results by score
        ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        timings["total_retrieval_time"] = sum(timings.values())
        return [doc for doc, score in ranked_results], timings

    # If no documents are found, return empty results
    timings["total_retrieval_time"] = sum(timings.values())
    return [], timings

def generate_response(context, query, history):
    """
    Generate a response using the LLM based on context and conversation history.

    Args:
        context (str): Relevant document text.
        query (str): User query text.
        history (deque): Conversation history.

    Returns:
        str: Generated response text.
        float: Response generation time.
    """
    # Prepare conversation history as text
    history_text = "\n".join([f"Q: {h['query']}\nA: {h['response']}" for h in history])

    # Prepare input for the LLM
    input_text = (
        f"Conversation History:\n{history_text}\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        f"Please provide a concise and accurate response."
    )

    # Generate response using the chat function
    start_chat = time.time()
    response: ChatResponse = chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": input_text}]
        # , options={"num_predict": 300, "stream": True}  # Options for token prediction and streaming response
    )

    # Uncomment below to print the token count for generated response
    # print(f"Number of tokens generated: {response.eval_count}")
    generation_time = time.time() - start_chat
    return response.message.content.strip(), generation_time

def async_generate_response(context, query, history):
    """
    Generate a response asynchronously using a thread pool.

    Args:
        context (str): Relevant document text.
        query (str): User query text.
        history (deque): Conversation history.

    Returns:
        Future: Future object for the response generation.
    """
    return executor.submit(generate_response, context, query, history)

# ========================== Flask API Routes ==========================

@app.route("/ask", methods=["POST"])
def ask():
    """
    Handle user queries and generate responses using the LLM.

    Returns:
        JSON response containing the generated answer and timing breakdowns.
    """
    data = request.get_json()  # Get input data
    query = data.get("query", "")

    if query:
        overall_start = time.time()  # Start overall timing

        # Step 1: Retrieve relevant documents
        retrieved_docs, retrieval_timings = retrieve_documents(query, top_k=3)

        # Step 2: Generate context from retrieved documents
        context = " ".join(retrieved_docs)[:1000]  # Limit context size to 1000 characters
        response_future = async_generate_response(context, query, conversation_history)

        # Add query to conversation history (response pending)
        conversation_history.append({"query": query, "response": "Pending..."})

        # Wait for the response from LLM
        response, response_time = response_future.result()
        conversation_history[-1]["response"] = response  # Update history with response

        overall_end = time.time()  # End overall timing

        # Compile timing breakdowns
        timings = {
            "embedding_time": retrieval_timings.get("embedding_time", 0),
            "qdrant_search_time": retrieval_timings.get("qdrant_search_time", 0),
            "ranking_time": retrieval_timings.get("ranking_time", 0),
            "total_retrieval_time": retrieval_timings.get("total_retrieval_time", 0),
            "llm_response_time": response_time,
            "total_api_processing_time": overall_end - overall_start,
        }

        # Return response and timings
        return jsonify({"response": response, "timings": timings})
    else:
        # Return error if no query is provided
        return jsonify({"error": "No query provided"}), 400

# ========================== Main Entry Point ==========================

if __name__ == "__main__":
    # Run the Flask app with threading enabled
    app.run(debug=True, threaded=True)
