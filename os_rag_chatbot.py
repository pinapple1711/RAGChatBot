import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = r"os_concepts_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="OS_Concepts")

user_query = input("How can I help you about Operating Systems?\n\n")

results = collection.query(
    query_texts=[user_query],
    n_results=4
)

print(results['documents'])

client = OpenAI(
    base_url="https://router.huggingface.co/fireworks-ai/inference/v1",
    api_key=os.environ["HF_TOKEN"],
)

system_prompt = f"""
You are a helpful assistant. You answer questions about Operating System Concepts.
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
--------------------
The data:
{results['documents']}
"""

response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-r1-0528", 
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_query}    
    ]
)

print("\n\n---------------------\n\n")
print(response.choices[0].message.content) 