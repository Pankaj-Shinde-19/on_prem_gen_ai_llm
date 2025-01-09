from flask import Flask, request, jsonify
import time  # Import time module to measure execution time
import pickle
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from ollama import chat, ChatResponse
import os
from flask_cors import CORS

# Disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define file paths for loading preprocessed data
embedding_files_path = r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\embedding_files\file_name"
corpus_file_path = os.path.join(embedding_files_path, "corpus.txt")
chunks_file_path = os.path.join(embedding_files_path, "chunks.pkl")

# Load preprocessed data
with open(corpus_file_path, "r") as f:
    corpus = f.read()

with open(chunks_file_path, "rb") as f:
    chunks = pickle.load(f)

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device='cpu')

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Retrieval function
def retrieve_documents(query, top_k=5):
    timings = {}
    start_retrieval = time.time()
    query_vector = embedding_model.encode([query])[0]
    timings["embedding_time"] = time.time() - start_retrieval

    start_search = time.time()
    results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=top_k
    )
    timings["qdrant_search_time"] = time.time() - start_search

    candidate_docs = [result.payload["text"] for result in results]
    if candidate_docs:
        start_ranking = time.time()
        pairs = [[query, doc] for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)
        timings["ranking_time"] = time.time() - start_ranking

        ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        timings["total_retrieval_time"] = sum(timings.values())
        return [doc for doc, score in ranked_results], timings
    timings["total_retrieval_time"] = sum(timings.values())
    return [], timings

# Response generation function
def generate_response(context, query):
    input_text = f"Context: {context}\nQuestion: {query}\nPlease provide a concise and accurate response."
    start_chat = time.time()
    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': input_text,
        }
    ])
    generation_time = time.time() - start_chat
    return response.message.content.strip(), generation_time

# Flask route for API
@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles user queries and measures response time.

    Returns:
        JSON response containing the generated answer, timing breakdowns, and time taken.
    """
    data = request.get_json()
    query = data.get("query", "")

    if query:
        overall_start = time.time()  # Start overall timing

        # Step 1: Retrieve documents
        retrieved_docs, retrieval_timings = retrieve_documents(query)

        # Step 2: Generate context and response
        context = " ".join(retrieved_docs)  # Combine context
        response, response_time = generate_response(context, query)

        overall_end = time.time()  # End overall timing

        timings = {
            "embedding_time": retrieval_timings.get("embedding_time", 0),
            "qdrant_search_time": retrieval_timings.get("qdrant_search_time", 0),
            "ranking_time": retrieval_timings.get("ranking_time", 0),
            "total_retrieval_time": retrieval_timings.get("total_retrieval_time", 0),
            "llm_response_time": response_time,
            "total_api_processing_time": overall_end - overall_start,
        }

        return jsonify({
            "response": response,
            "timings": timings
        })
    else:
        return jsonify({"error": "No query provided"}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
