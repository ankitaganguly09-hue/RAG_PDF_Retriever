from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#Create embeddings for the chunks
embeddings = HuggingFaceEmbeddings()

#Loading the vector store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

#Retriever to retrieve relevant chunks from the vector store
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

#LLM to generate answers based on the retrieved chunks
llm = ChatMistralAI(model="mistral-small-2506", temperature=0.7)

#Prompt template to format the retrieved chunks and the question
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

print("Welcome to the RAG PDF Retriever! Ask any question related to the content of the PDF, and I'll do my best to provide an answer based on the retrieved information. \n Press 0 to exit.")
while True:
    query = input("Enter your question: ")
    if query == "0":
        print("Exiting the RAG PDF Retriever. Goodbye!")
        break
    
    #Retrieving relevant chunks from the vector store
    relevant_chunks = retriever.invoke(query)
    
    #Formatting the retrieved chunks and the question using the prompt template
    formatted_prompt = prompt_template.format(context="\n".join([chunk.page_content for chunk in relevant_chunks]), question=query)
    
    #Generating an answer using the LLM
    answer = llm.invoke(formatted_prompt)
    
    print(f"AI: {answer.content}\n")