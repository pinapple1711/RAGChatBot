import os
from pypdf import PdfReader
from chromadb.utils import embedding_functions
import chromadb

# Step 1: Extract text from PDF
PDF_PATH = r"C:\Users\mezna\Documents\ChatUI\OperatingSystemConcepts9th2012.12.pdf"
CHROMA_PATH = r"C:\Users\mezna\Documents\ChatUI"

reader = PdfReader(PDF_PATH)
all_text = []
for page in reader.pages:
    text = page.extract_text()
    if text:
        all_text.append(text)

# Step 2: Chunk text (simple split by characters)
chunk_size = 1000  # characters per chunk
chunks = []
for text in all_text:
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)

# Step 3: Store chunks in ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedder = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="OS_Concepts", embedding_function=embedder)

# Add chunks to collection (with unique IDs)
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        metadatas=[{"page": idx}],
        ids=[f"osconcepts-{idx}"]
    )

print(f"Added {len(chunks)} chunks to ChromaDB collection 'OS_Concepts'.") 