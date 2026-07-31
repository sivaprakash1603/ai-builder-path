import os
import glob
from langchain_core.tools import Tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize the vector store once when the module loads
_vectorstore = None

def init_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    print("Initializing RAG Vector Store...")
    # Load all markdown files from the data directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    docs = []
    for filepath in glob.glob(os.path.join(data_dir, "*.md")):
        loader = TextLoader(filepath)
        docs.extend(loader.load())

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Embed and store in FAISS (in-memory for this lab)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _vectorstore = FAISS.from_documents(splits, embeddings)
    print(f"Vector store initialized with {len(splits)} chunks.")
    return _vectorstore

def search_hr_policies(query: str) -> str:
    """Search HR and Compliance policies for the given query."""
    vs = init_vectorstore()
    # Retrieve top 3 relevant chunks
    docs = vs.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant policies found."
        
    results = []
    for d in docs:
        results.append(d.page_content)
        
    return "\n\n---\n\n".join(results)

# Create the LangChain tool
rag_tool = Tool(
    name="Internal_Knowledge_Search",
    description="Use this tool to search for and retrieve answers from Presidio's internal documents, including HR policies, compliance documents, marketing feedback, and internal hiring metrics. Input should be a specific search query.",
    func=search_hr_policies
)
