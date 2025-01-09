import pdfplumber  # For extracting text from PDF files
from qdrant_client import QdrantClient  # Qdrant client for vector database operations
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import for embeddings
from sentence_transformers import SentenceTransformer  # For generating text embeddings
import pickle  # For saving and loading serialized data
import os  # For file and directory operations

# Initialize the sentence embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from multiple PDF files
def extract_text_from_pdfs(pdf_files):
    """
    Reads text from a list of PDF files and combines it into a single string.

    Args:
        pdf_files (list): List of file paths to PDF documents.

    Returns:
        str: Combined text extracted from all pages of the PDFs.
    """
    extracted_text = []
    for file in pdf_files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted_text.append(page.extract_text())
    return " ".join(extracted_text)

# List of PDF file paths to process (updated paths)
pdf_files = [
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Accelerate Testing Efficiency with testRigor.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Azure Optimization & Acceleration Framework.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Hyper Intelligent Automation Using WorkFusion.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Hyperpersonalization Using Adobe AEP.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Conversational AI with Google Dialogflow.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Platform Migration and Modernization to Cloud with Cast.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Platform Observability with Splunk.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Scalable Data Insights using Databricks.pdf",
    r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\pdfs\Agivant - Well-Architected Framework for AWS Migration.pdf"
]

# Extract text from the specified PDF files
corpus = extract_text_from_pdfs(pdf_files)

# Define the file path for embedding files
embedding_files_path = r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\embedding_files\file_name"

# Ensure the directory exists
os.makedirs(embedding_files_path, exist_ok=True)

# Save the extracted corpus to a text file for reference
corpus_file_path = os.path.join(embedding_files_path, "corpus.txt")
with open(corpus_file_path, "w") as f:
    f.write(corpus)

# Generate embeddings for chunks of the corpus
chunks = corpus.split(". ")  # Split corpus into sentences or chunks based on periods
embeddings = embedding_model.encode(chunks)  # Generate embeddings for each chunk

# Save the chunks and their embeddings for later use
chunks_file_path = os.path.join(embedding_files_path, "chunks.pkl")
embeddings_file_path = os.path.join(embedding_files_path, "embeddings.pkl")

with open(chunks_file_path, "wb") as f:
    pickle.dump(chunks, f)

with open(embeddings_file_path, "wb") as f:
    pickle.dump(embeddings, f)

# Initialize the Qdrant client for vector database operations
# Uncomment the line below for a file-based database
# qdrant_client = QdrantClient(path="qdrant.db")

# Use Qdrant in server mode (ensure Qdrant is running on localhost at port 6333)
qdrant_client = QdrantClient(host="localhost", port=6333)

from qdrant_client.http.models import PointStruct, VectorParams  # For defining collection structure

# Name of the collection to store document embeddings
collection_name = "documents"

# Check if the collection exists; create it if it doesn't
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_model.get_sentence_embedding_dimension(),  # Dimensionality of the embeddings
            distance="Cosine"  # Metric used for similarity search
        ),
    )

# Prepare points (data to be inserted into the collection)
points = [
    PointStruct(id=i, vector=emb, payload={"text": chunk})  # Embedding and associated text chunk
    for i, (emb, chunk) in enumerate(zip(embeddings, chunks))
]

# Insert the points into the Qdrant collection
qdrant_client.upsert(collection_name=collection_name, points=points)

# Script complete: PDF text extracted, embeddings generated, and data stored in Qdrant
