from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import config


def load_documents():
    loader = PyPDFLoader(str(config.DATA_DIR / "attention_is_all_you_need.pdf"))
    docs = loader.load()
    print(f"Loaded {len(docs)} page(s) from the PDF.")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks


def embed_and_store(chunks):
    embedder = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.PERSIST_DIR),
    )
    print(f"Stored {len(chunks)} chunks in the vector database at {config.PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)
