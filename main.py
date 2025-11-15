import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def build_vector_store():
    print("Loading speech.txt ...")
    loader = TextLoader("speech.txt")
    documents = loader.load()

    print("Splitting text ...")
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    print("Creating embeddings ...")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building Chroma vector store ...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory="db"
    )

    vectordb.persist()
    print("Vector DB created.")
    return vectordb


def load_vector_store():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory="db",
        embedding_function=embedder
    )
    return vectordb


def start_qa_system():

    if not os.path.exists("db"):
        vectordb = build_vector_store()
    else:
        vectordb = load_vector_store()

    retriever = vectordb.as_retriever()

    print("Loading Ollama Mistral ...")
    llm = Ollama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    print("\n📌 AmbedkarGPT Q&A System (Type 'exit' to quit)\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = qa_chain.invoke(query)

        print("\nAnswer:\n", result["result"])
        print("\nSources:")
        for src in result["source_documents"]:
            print("-", src.page_content)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    start_qa_system()
