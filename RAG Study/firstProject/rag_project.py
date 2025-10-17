# %% [markdown]
# Rag system to generate answer of queries by user

# %%
#Install libraries


#Indexing (Document Ingestion)

#Indexing (Text Splitting)

#Indexing (Embedding Generation)

#Indexing( Storing in Vector Store))

#Retrieval
#Augmentation
#Generation


#Building a Chain

# %% [markdown]
# Load Data

# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Load PDF
loader = PyPDFLoader(
    "/home/nurshed/Desktop/python/project/RAG Study/firstProject/the_alchemist.pdf",
    mode="single",
)
documents = loader.load()

# %%
print(len(documents))
print(documents[0].metadata)

# %% [markdown]
# Split the Document

# %%
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

chunks = split_documents(documents)
print(chunks[0])

# %%
len(chunks)

# %%
print(chunks[100])

# %% [markdown]
# Embedding

# %%
from langchain.embeddings import HuggingFaceEmbeddings

# Wrap your model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# %%
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model,
)

print("✅ FAISS vector store created with IDs for each chunk!")


# %%
vector_store.index_to_docstore_id

# %%
vector_store.get_by_ids(['57b42fc0-bafb-4375-a077-04f69a9400fe'])

# %% [markdown]
# Retriver

# %%
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# %%
retriever

# %%
retriever.invoke('who is crystal merchant')

# %% [markdown]
# Augmentation

# %%
import getpass
import os
from dotenv import load_dotenv

def configure():
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key = getpass.getpass("Enter your Google API key: ")
        os.environ["GOOGLE_API_KEY"] = api_key

    return api_key

GOOGLE_API_KEY = configure()
print("✅ Google API key loaded successfully.")


# %%
# ----------------------------
# Imports
# ----------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ----------------------------
# LLM Setup: Gemini 2.5 Pro
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.2,
    convert_system_message_to_human=True
)

# %%
# ----------------------------
# Prompt Template
# ----------------------------
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided document context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

# %%
# ----------------------------
# Retrieve + Prepare Context
# ----------------------------
question = "is the topic of Pyramid in this document? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

# %%
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
context_text

# %%
# ----------------------------
# Final Prompt 
# ----------------------------
final_prompt = prompt.invoke({"context": context_text, "question": question})
final_prompt

# %% [markdown]
# Generation

# %%
# Generation
answer = llm.invoke(final_prompt)

print(answer.content)





# %% [markdown]
# Chain

# %%
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# %%
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

# %%
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# %%
parallel_chain.invoke('who is santiago')

# %%
parser = StrOutputParser()

# %%
main_chain = parallel_chain | prompt | llm | parser

# %%
main_chain.invoke('what is the learning from the story?')
