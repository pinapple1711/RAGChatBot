from flask import Flask, render_template, request, session, redirect, url_for
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')

CHROMA_PATH = r"C:\Users\mezna\Documents\ChatUI"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="OS_Concepts")

client = OpenAI(
    base_url="https://router.huggingface.co/fireworks-ai/inference/v1",
    api_key=os.environ["HF_TOKEN"],
)

def clean_agent_reply(reply):
    # Remove any 'think' statements or similar reasoning blocks
    import re
    # Remove <think>...</think> or <think>...</think> style blocks
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)
    # Remove lines starting with 'Hmm,' or similar
    reply = re.sub(r'^Hmm.*$', '', reply, flags=re.MULTILINE)
    return reply.strip()

@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
    chat_history = session['chat_history']

    if request.method == 'POST':
        user_input = request.form['user_input']
        chat_history.append({'role': 'user', 'content': user_input})

        # RAG retrieval
        results = collection.query(query_texts=[user_input], n_results=4)
        context = results['documents']

        system_prompt = f"""
You are a helpful assistant. You answer questions about Operating System Concepts.
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
--------------------
The data:
{context}
"""
        response = client.chat.completions.create(
            model="accounts/fireworks/models/deepseek-r1-0528",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        agent_reply = response.choices[0].message.content
        agent_reply = clean_agent_reply(agent_reply)
        chat_history.append({'role': 'agent', 'content': agent_reply})
        session['chat_history'] = chat_history
        return redirect(url_for('chat'))

    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True, port=8080) 