# 📄 RAG PDF Retriever

A Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, and Mistral AI. Upload a PDF document, create vector embeddings, and ask questions about the document using semantic search and LLM-powered responses.

---

## 🚀 Features

* Upload PDF documents through a simple web interface.
* Extract and split PDF content into manageable chunks.
* Generate embeddings using Hugging Face sentence-transformer models.
* Store embeddings in Chroma vector database.
* Retrieve relevant document sections using Maximum Marginal Relevance (MMR) search.
* Generate context-aware answers using Mistral AI.
* Maintain conversation history during the session.
* User-friendly Streamlit interface.

---

## 🏗️ Architecture

```text
PDF Upload
     │
     ▼
PyPDFLoader
     │
     ▼
Text Chunking
(RecursiveCharacterTextSplitter)
     │
     ▼
HuggingFace Embeddings
     │
     ▼
Chroma Vector Database
     │
     ▼
Retriever (MMR Search)
     │
     ▼
Mistral AI LLM
     │
     ▼
Answer Generation
```

---

## 📦 Tech Stack

* Streamlit
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Mistral AI
* Python
* PyPDFLoader

---

## 📋 Prerequisites

Before running the application, ensure you have:

* Python 3.10+
* A Mistral AI API Key

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ankitaganguly09-hue/RAG_PDF_Retriever.git

cd RAG_PDF_Retriever
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📂 Project Structure

```text
rag-pdf-retriever/
│
├── app.py
├── .env
├── requirements.txt
├── chroma_db/
├── README.md
└── assets/
```

---

## 🧠 How It Works

### PDF Processing

1. User uploads a PDF file.
2. PDF is temporarily stored.
3. Content is extracted using `PyPDFLoader`.
4. Text is divided into chunks:

   * Chunk Size: 1000
   * Chunk Overlap: 100

### Embedding Generation

Each text chunk is converted into vector embeddings using:

```python
HuggingFaceEmbeddings()
```

### Vector Storage

Embeddings are stored in:

```python
Chroma
```

with persistence enabled via:

```python
persist_directory="chroma_db"
```

### Retrieval

Relevant chunks are retrieved using MMR search:

```python
search_type="mmr"
```

Configuration:

```python
{
    "k": 4,
    "fetch_k": 10,
    "lambda_mult": 0.5
}
```

### Response Generation

Retrieved context is passed to Mistral AI along with the user's question:

```python
ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.7
)
```

The model generates a contextual answer based on the retrieved document content.

---

## 💬 Example Usage

### Upload PDF

Click **Upload a PDF** and select a document.

### Process PDF

Click **Process PDF** to:

* Extract text
* Create embeddings
* Store vectors

### Ask Questions

Example questions:

```text
What is the main topic of the document?

Summarize chapter 3.

What are the key findings?

Who is the author?

Explain the methodology section.
```

---

## 🔄 Session Management

The application stores:

### Vector Store

```python
st.session_state.vectorstore
```

### Chat History

```python
st.session_state.chat_history
```

This allows users to continue asking questions during the current session.

---

## 📈 Future Improvements

* Multi-PDF support
* Conversational memory with LangChain chains
* Source citation display
* Metadata filtering
* Streaming responses
* User authentication
* Docker deployment
* Hybrid search (keyword + vector)
* Persistent chat history
* PDF page references in answers

---

## 🐞 Troubleshooting

### Mistral API Key Error

Ensure your `.env` file contains:

```env
MISTRAL_API_KEY=your_api_key
```

### Empty Responses

Check:

* PDF was processed successfully.
* Vector database was created.
* API key is valid.

### Slow Processing

Large PDFs may take longer due to:

* Text extraction
* Embedding generation
* Vector storage

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Built using LangChain, ChromaDB, Hugging Face, Mistral AI, and Streamlit to demonstrate Retrieval-Augmented Generation (RAG) workflows for PDF question answering.
