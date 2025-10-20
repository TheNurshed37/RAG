# main.py
from vector_store import build_vector_store, load_vector_store
from rag import setup_rag, answer_query

def main():
    print("RAG System")
    print("1. Build or update FAISS index")
    print("2. Query existing index")
    choice = input("Choose an option (1 or 2): ").strip()

    if choice == "1":
        print("Building FAISS index...")
        build_vector_store()
    elif choice == "2":
        print("Loading RAG components...")
        vector_store = load_vector_store()
        retriever, llm, prompt = setup_rag(vector_store)

        while True:
            question = input("\nAsk a question (or type 'exit' to quit): ").strip()
            if question.lower() == "exit":
                print("Exiting.")
                break

            answer = answer_query(question, retriever, llm, prompt)
            print("\nAnswer:")
            print(answer)
    else:
        print("Invalid option. Please choose 1 or 2.")

if __name__ == "__main__":
    main()
