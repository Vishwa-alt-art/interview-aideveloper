import os
import chromadb
from dotenv import load_dotenv
from groq import Groq

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_or_create_collection(name="resume_collection")

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(request: Question):
    question = request.question.strip()
    query_embedding = embedding_model.encode(question)
    k = min(3, collection.count())
    results = collection.query(

    query_embeddings=[query_embedding.tolist()],
    n_results=k,
    include = ['documents', 'metadatas', 'distances']
)

    retrieved_chunks = results['documents'][0]

    context = '\n\n'.join(retrieved_chunks)

    from groq import Groq

    client = Groq()

    chat_completion = client.chat.completions.create( model= "openai/gpt-oss-120b"
      messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": "You are a helpful assistant.Answer the user's question"
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            Context: {context}
            Question: {question}
        }
      ],

      # The language model which will generate the completion.
        
    )

    # Print the completion returned by the LLM.
    answer = chat_completion.choices[0].message.content

    return {
        'question': question,
        'answer': answer
    }

    


