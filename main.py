from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn
from openai import OpenAI
import faiss
import numpy as np
from typing import List, Dict
import os
from dotenv import load_dotenv
import json
import asyncio
import sqlite3
import pickle
from functools import lru_cache
import hashlib

# Load environment variables from a .env file
load_dotenv()

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up OpenAI API key
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

# Database setup
conn = sqlite3.connect('embeddings.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS embeddings
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              text TEXT UNIQUE,
              embedding BLOB)''')
c.execute('''CREATE TABLE IF NOT EXISTS files
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              file_hash TEXT UNIQUE,
              processed BOOLEAN)''')
conn.commit()

# Global variables for vector database
index: faiss.IndexFlatL2 = None
documents: List[str] = []
@lru_cache(maxsize=1000)
def get_embedding_cached(text: str) -> List[float]:
    c.execute("SELECT embedding FROM embeddings WHERE text=?", (text,))
    result = c.fetchone()
    if result:
        return pickle.loads(result[0])
    embedding = get_embedding(text)
    c.execute("INSERT INTO embeddings (text, embedding) VALUES (?, ?)",
              (text, pickle.dumps(embedding)))
    conn.commit()
    return embedding

def get_file_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def is_file_processed(file_hash: str) -> bool:
    c.execute("SELECT processed FROM files WHERE file_hash=?", (file_hash,))
    result = c.fetchone()
    return result and result[0]

def mark_file_as_processed(file_hash: str):
    c.execute("INSERT OR REPLACE INTO files (file_hash, processed) VALUES (?, ?)",
              (file_hash, True))
    conn.commit()

async def process_file_content(content: str):
    global index, documents

    try:
        file_hash = get_file_hash(content)
        if is_file_processed(file_hash):
            yield json.dumps({"progress": 100, "status": "File already processed!"})
            return

        yield json.dumps({"progress": 20, "status": "File content received..."})
        await asyncio.sleep(0.1)

        documents = [doc.strip() for doc in content.split('.') if doc.strip()]
        yield json.dumps({"progress": 40, "status": "Content split into chunks..."})
        await asyncio.sleep(0.1)

        embeddings = []
        batch_size = 50  # Adjust based on your needs and API limits
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_embeddings = [get_embedding_cached(doc) for doc in batch]
            embeddings.extend(batch_embeddings)
            progress = 40 + int((i / len(documents)) * 50)
            yield json.dumps({"progress": progress, "status": "Creating embeddings..."})
            await asyncio.sleep(0.1)

        vector_dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(vector_dimension)
        index.add(np.array(embeddings).astype('float32'))
        yield json.dumps({"progress": 100, "status": "FAISS index built!"})

        # Save index and documents for persistence
        faiss.write_index(index, 'faiss_index.bin')
        with open('documents.json', 'w') as f:
            json.dump(documents, f)

        mark_file_as_processed(file_hash)

    except Exception as e:
        yield json.dumps({"error": str(e)})

@app.on_event("startup")
async def startup_event():
    global index, documents
    if os.path.exists('faiss_index.bin') and os.path.exists('documents.json'):
        index = faiss.read_index('faiss_index.bin')
        with open('documents.json', 'r') as f:
            documents = json.load(f)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    try:
        content = await file.read()
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        return JSONResponse(content={"error": "File encoding is not UTF-8. Please upload a UTF-8 encoded file."}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Error reading file: {str(e)}"}, status_code=500)

    file_hash = get_file_hash(text_content)
    if is_file_processed(file_hash):
        return JSONResponse(content={"message": "File already processed", "status": "File already processed!"}, status_code=200)

    async def event_stream():
        async for data in process_file_content(text_content):
            yield f"data: {data}\n\n"
        yield "data: {\"complete\": true}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global index, documents
    
    if index is None or not documents:
        raise HTTPException(status_code=400, detail="Please upload a file first")
    
    try:
        # Get embedding for the question
        question_embedding = get_embedding(question)
        
        # Search for similar chunks
        k = 3  # Number of similar chunks to retrieve
        distances, indices = index.search(np.array([question_embedding]).astype('float32'), k)
        
        # Retrieve the most similar chunks
        similar_chunks = [documents[i] for i in indices[0]]
        
        # Prepare the prompt for OpenAI
        context = "\n".join(similar_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Use an appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        # Extract the answer from the response
        answer = response.choices[0].message.content.strip()
        
        return {"answer": answer, "context": similar_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    conn.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)