# build_index.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Load PDF
pdf_path = "the_alchemist.pdf"
print("Loading PDF...")
loader = PyPDFLoader(pdf_path, mode="single")
documents = loader.load()

#Split into chunks
print("Splitting into chunks...")
def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

chunks = split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

#Embeddings
print("Generating embeddings using all-MiniLM-L6-v2...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Create FAISS vector store
print("Creating FAISS vector store...")
vector_store = FAISS.from_documents(chunks, embedding_model)

#Save FAISS index
save_path = "faiss_index"
vector_store.save_local(save_path)
print(f"Vector store saved successfully at: {save_path}")