from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

#Load data from PDF file
data = PyPDFLoader("GENAI_LLM_RESUME_QUESTIONS_ANSWERS.pdf")
docs = data.load()

#Splitting the data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = text_splitter.split_documents(docs)

#Creating embeddings for the chunks
embeddings = HuggingFaceEmbeddings()

#Creating a vector store to store the embeddings
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="chroma_db"
)


