# query_rag.py
import os
import getpass
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

#Load environment variables
def configure():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Google API key: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key

GOOGLE_API_KEY = configure()

#Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#Setup Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    convert_system_message_to_human=True
)

#Create prompt template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided document context.
    If the context is insufficient, say "I don’t know" — do not make up answers.

    Context:
    {context}

    Question:
    {question}
    """,
    input_variables=["context", "question"]
)

#Ask a question
question = input("\n Ask your question: ")
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

#Generate answer
final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = llm.invoke(final_prompt)

# print("\n🧩 Retrieved context snippets:")
# for i, doc in enumerate(retrieved_docs, 1):
#     print(f"\n--- Document {i} ---\n{doc.page_content[:400]}...")

print("\n Answer:")
print(answer.content)