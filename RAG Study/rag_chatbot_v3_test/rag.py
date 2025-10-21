# rag.py
import os
import getpass
import threading
import queue
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from vector_store import load_vector_store, SAVE_PATH

load_dotenv()

# sentinel objects for queue
_END = "<STREAM_END>"
_ERROR = "<STREAM_ERROR>"


class QueueCallbackHandler(BaseCallbackHandler):
    """
    Callback that pushes tokens into a queue. Used for streaming to clients.
    """
    def __init__(self, q: queue.Queue):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs):
        # put token as soon as it's received
        try:
            self.q.put_nowait(token)
        except Exception:
            pass

    def on_llm_end(self, response, **kwargs):
        # signal finished
        try:
            self.q.put_nowait(_END)
        except Exception:
            pass

    def on_llm_error(self, error, **kwargs):
        # signal error
        try:
            self.q.put_nowait((_ERROR, str(error)))
        except Exception:
            pass


def _ensure_google_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Google API key: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


def _build_prompt(context: str, question: str) -> str:
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
        input_variables=["context", "question"],
    )
    return prompt.invoke({"context": context, "question": question})


def stream_answer_generator(question: str, save_path: str = SAVE_PATH, max_retrieval_k: int = 5):
    """
    Generator that yields tokens produced by the LLM as they arrive.
    This function starts the LLM invocation in a background thread and yields tokens from a queue.
    """
    _ensure_google_api_key()

    # load vector store and prepare retriever
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": max_retrieval_k})

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = _build_prompt(context_text, question)

    q = queue.Queue(maxsize=1024)
    callback = QueueCallbackHandler(q)

    def _run_llm():
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.2,
                streaming=True,
                callbacks=[callback],
                convert_system_message_to_human=True,
            )
            # the blocking call; callbacks will push tokens into queue
            llm.invoke(final_prompt)
            # ensure end is signaled by callback.on_llm_end; if not, do it here
            try:
                q.put_nowait(_END)
            except Exception:
                pass
        except Exception as e:
            try:
                q.put_nowait((_ERROR, str(e)))
            except Exception:
                pass

    thread = threading.Thread(target=_run_llm, daemon=True)
    thread.start()

    # yield tokens as they arrive
    while True:
        item = q.get()
        if item == _END:
            break
        if isinstance(item, tuple) and item and item[0] == _ERROR:
            # error occurred in LLM thread
            yield f"\n[ERROR] {item[1]}\n"
            break
        # item is a token string
        yield item

    # optionally yield a termination marker
    yield "\n\n[STREAM_END]\n"
