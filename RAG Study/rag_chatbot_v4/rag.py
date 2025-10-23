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
    try:
        # Check if FAISS index exists
        if not os.path.exists("faiss_index") or not os.listdir("faiss_index"):
            print("FAISS index not found or empty")
            return "I don't know (No documents have been uploaded yet)."

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            GOOGLE_API_KEY = getpass.getpass("Enter your Google API key: ")
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load vector store with error handling
        try:
            vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return "I don't know (Error loading documents)."

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})  # Increased k

        # Retrieve documents
        retrieved_docs = retriever.invoke(question)
        print(f"🔍 Retrieved {len(retrieved_docs)} documents for question: '{question}'")
        
        # Debug: print retrieved content
        for i, doc in enumerate(retrieved_docs):
            print(f"Doc {i+1} (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content[:200]}...")

        if not retrieved_docs:
            print("No documents retrieved")
            return "I don't know (No relevant information found in the documents)."

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        # Check if context is meaningful
        if len(context_text.strip()) < 10:
            print("Retrieved context is too short or empty")
            return "I don't know (The documents don't contain enough text to answer your question)."

        print(f"Context length: {len(context_text)} characters")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,  # Increased temperature for more creative responses
            streaming=True,
            convert_system_message_to_human=True
        )

        #prompt template
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

        final_prompt = prompt.invoke({"context": context_text, "question": question})
        
        # Debug: print the final prompt
        print(f"🤖 Sending prompt to LLM...")
        
        answer = llm.invoke(final_prompt)
        
        print(f"✅ LLM response received: {answer.content[:100]}...")
        return answer.content

    except Exception as e:
        print(f"❌ Error in answer_question: {e}")
        return f"I don't know (Error: {str(e)})"
 