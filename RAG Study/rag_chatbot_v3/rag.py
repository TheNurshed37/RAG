import os
import getpass
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

def answer_question(question: str):
    """Retrieve from FAISS and generate answer using Gemini."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = getpass.getpass("Enter your Google API key: ")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2,
        streaming = True,
        convert_system_message_to_human=True
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided document context.
        If the context is insufficient, say "I don’t know".

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)
    return answer.content


# def answer_question_stream(question: str):
#     """Yield LLM tokens as they are generated for streaming."""
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     if not GOOGLE_API_KEY:
#         GOOGLE_API_KEY = getpass.getpass("Enter your Google API key: ")
#         os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-pro",
#         temperature=0.2,
#         streaming=True,
#         convert_system_message_to_human=True
#     )

#     prompt = PromptTemplate(
#         template="""
#         You are a helpful assistant.
#         Answer ONLY from the provided document context.
#         If the context is insufficient, say "I don’t know".

#         Context:
#         {context}

#         Question:
#         {question}
#         """,
#         input_variables=["context", "question"]
#     )

#     retrieved_docs = retriever.invoke(question)
#     context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#     final_prompt = prompt.invoke({"context": context_text, "question": question})

#     for chunk in llm.stream(final_prompt):
#         yield chunk.text 
        


'''
def stream_answer(question: str):
    """Generator that streams LLM answer chunk by chunk."""
    retriever = get_retriever()
    if not retriever:
        yield "I don’t know (No document found)."
        return

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = build_prompt(context_text, question)
    llm = get_llm()

    # Stream chunks from LLM
    for chunk in llm.stream(final_prompt):
        yield chunk.content  # must yield only string

'''

'''
# ADD THIS FUNCTION FOR STREAMING
def answer_question_stream(question: str):
    """Streaming version - returns generator for streaming response"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = getpass.getpass("Enter your Google API key: ")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0.2,
        convert_system_message_to_human=True,
        streaming=True  # Enable streaming
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided document context.
        If the context is insufficient, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = prompt.invoke({"context": context_text, "question": question})
    
    # Stream the response
    stream = llm.stream(final_prompt)
    return stream

    '''