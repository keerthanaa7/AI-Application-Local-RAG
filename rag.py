from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load your custom data (Create a 'data.txt' file with some facts)
# Example data.txt: "The secret code for the Sunnyvale office is 998877."
loader = TextLoader("data.txt")
documents = loader.load()

# 2. Split text into chunks (LLMs have a "context window" limit)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Create Embeddings & Store in Vector DB
# This turns text into numbers (vectors) so we can search them mathematically
embeddings = OllamaEmbeddings(model="llama3") # or whichever model you have
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)

# 4. Setup the Local LLM (Ollama)
llm = OllamaLLM(model="llama3")

# 5. Create the RAG Chain
# This tells the system: "When I ask a question, look in the Vector DB first!"
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 6. Ask a question about your private data
query = "What is the secret code for the Sunnyvale office?"
response = qa_chain.invoke(query)

print(f"Answer: {response}")