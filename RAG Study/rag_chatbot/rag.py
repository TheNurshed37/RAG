# rag.py
import os
import getpass
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

def configure():
    """Load or ask for Google API key."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Google API key: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key

def setup_rag(vector_store: FAISS):
    """Set up retriever, LLM, and prompt."""
    configure()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2,
        convert_system_message_to_human=True
    )

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

    return retriever, llm, prompt

def answer_query(question: str, retriever, llm, prompt):
    """Generate answer to a user question."""
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)
    return answer.content
