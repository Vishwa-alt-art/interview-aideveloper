import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
import PyPDF2
import uuid

import uvicorn

from groq import Groq
from pydantic import BaseModel

from fastapi import FastAPI

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

#load the env variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR/'.env')
api_key = os.getenv('GROQ_API_KEY')

if not api_key:
    raise ValueError('Groq api key is missing !')

#load the model
groq_client = Groq(api_key=api_key)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#chromadb
client = chromadb.PersistentClient(path=str(BASE_DIR/'chroma_db'))
collection = client.get_or_create_collection(name="resume_collection")

#fastapi
app = FastAPI()

class Question(BaseModel):
    question: str

def extracttext_frompdf(pdf_file):
    pdf_file = Path(pdf_file)

    page_texts = []
    with open(pdf_file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                page_texts.append({'text':page_text, 'page_number':page_num +1})
    if not page_texts:
        raise ValueError("No extractable text found in the PDF")

    #split the page_text into individual texts.
    text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500,
          chunk_overlap=100,
    
       )
    chunks = []
    metadata = []

    for page in page_texts:
        page_chunks = text_splitter.split_text(page['text'])
        for chunk in page_chunks:
            chunks.append(chunk)
            metadata.append({'source': pdf_file.name, 'page_number':page['page_number']})
    return chunks, metadata

#convert the chunks into embeddings using embedding vector.
def chunks_to_vectors(pdf_file):
    chunks, metadata = extracttext_frompdf(pdf_file)
    embeddings = embedding_model.encode(chunks)

    ids = [str(uuid.uuid4() for _ in chunks)]

    #store in chromadb

    collection.add(ids=ids, embeddings=embeddings,docements=chunks,metadata=metadata)

    return {"total_chunks":len(chunks), "message":"PDF successfully processed and stored."}

#injecting the pdf through API
@app.post('/ingest')
def ingest_pdf():
    pdf_file = BASE_DIR / "company_policy.pdf"
    result = extracttext_frompdf(pdf_file)
    return result

#from user question convert into embeddings.
def retrieve_context(question):
    query_embedding = embedding_model.encode(question)

    k = min(3,collection.count())
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k, include=['documents','metadatas','distances'])

    retrieved_chunks = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    distance = results['distances'][0]

    context = '\n'.join(retrieved_chunks)

    return context, retrieved_metadata, distance

#send the context + Question to Groq

def generate_answer(question, context):
    
    completion = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
      {
        "role": "system",
        "content": ("You are a helping assistant answer the user question")
      },
      {
          'role': 'user',
          'content':f'''Context:{context} Question:{question} '''
      }
    ],
    temperature=0.2,
    )

    answer = completion.choices[0].message.content
    return answer

#question answering API
@app.post('/ask')
def ask(request: Question):

    question = request.question.strip()

    context, metadata, distance = retrieve_context(question)
    answer = generate_answer(question, context)
    return {
        'question': question,
        'answer': answer,
    }

if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1',port=8000, reload=True)



        
