# Disney RAG Demo

This project demonstrates a Retrieval-Augmented Generation (RAG) system using FastAPI, OpenAI embeddings, and FAISS for efficient similarity search.

## Features

- Upload and process text files
- Generate embeddings for text chunks
- Store embeddings in a SQLite database
- Build a FAISS index for fast similarity search
- Ask questions about the uploaded content

## Prerequisites

- Python 3.10+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/armantark/RAG-Demo.git
   cd RAG-Demo
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Environment Setup

1. Create a `.env` file in the root directory of the project.

2. Add the following content to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Running the Application

1. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

2. Open a web browser and navigate to `http://localhost:8000`.

## Usage

1. Upload a text file using the file input on the web interface.
2. Wait for the file to be processed and the embeddings to be created.
3. Once processing is complete, you can ask questions about the uploaded content using the question input field.

## Project Structure

- `main.py`: FastAPI application and main logic
- `static/index.html`: Web interface
- `embeddings.db`: SQLite database for storing embeddings
- `faiss_index.bin`: FAISS index file for similarity search
- `documents.json`: Processed text chunks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.