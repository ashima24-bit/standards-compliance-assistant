from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
import config
import llm_provider

_embedder = None
_vectorstore = None
_all_chunks_cache = None
_bm25_index = None


def get_vectorstore():
    global _embedder, _vectorstore
    if _vectorstore is None:
        _embedder = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
        _vectorstore = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=_embedder,
            persist_directory=str(config.PERSIST_DIR),
        )
    return _vectorstore


def get_all_chunks_and_bm25():
    global _all_chunks_cache, _bm25_index
    if _all_chunks_cache is None:
        store = get_vectorstore()
        raw = store.get(include=["documents", "metadatas"])
        texts = raw["documents"]
        metas = raw["metadatas"]
        _all_chunks_cache = list(zip(texts, metas))
        tokenized = [t.lower().split() for t in texts]
        _bm25_index = BM25Okapi(tokenized)
    return _all_chunks_cache, _bm25_index


def hybrid_retrieve(question, k=None):
    k = k or config.RETRIEVE_K
    vectorstore = get_vectorstore()
    chunks, bm25 = get_all_chunks_and_bm25()

    semantic_results = vectorstore.similarity_search(question, k=k)
    semantic_texts = {doc.page_content for doc in semantic_results}

    bm25_scores = bm25.get_scores(question.lower().split())
    ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_texts = {chunks[i][0] for i in ranked_idx}

    combined_texts = semantic_texts | bm25_texts
    results = [
        {"text": text, "metadata": meta}
        for text, meta in chunks
        if text in combined_texts
    ]
    return results


_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict relevance grader. Given a QUESTION and a "
               "CHUNK of text, answer with exactly one word: 'yes' if the "
               "chunk helps answer the question, or 'no' if it does not."),
    ("human", "QUESTION: {question}\n\nCHUNK:\n{chunk}"),
])


def grade_relevance(question, chunks):
    if not chunks:
        return []
    llm = llm_provider.get_llm(fast=True)

    numbered_chunks = "\n\n".join(
        f"[{i}] {c['text'][:300]}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"QUESTION: {question}\n\n"
        f"Below are numbered text chunks. Reply with ONLY the numbers "
        f"(comma-separated) of chunks relevant to answering the question. "
        f"If none are relevant, reply 'none'.\n\n{numbered_chunks}"
    )
    response = llm.invoke(prompt).content.strip().lower()

    if response == "none":
        return []
    try:
        relevant_indices = [int(x.strip()) for x in response.split(",") if x.strip().isdigit()]
        return [chunks[i] for i in relevant_indices if i < len(chunks)]
    except (ValueError, IndexError):
        return chunks

def retrieve_and_grade(question):
    candidates = hybrid_retrieve(question)
    graded = grade_relevance(question, candidates)

    if len(graded) < 2:
        wider = hybrid_retrieve(question, k=config.RETRIEVE_K * 2)
        graded = grade_relevance(question, wider)

    return graded


def hybrid_retrieve_from_store(vectorstore, question, k=None):
    k = k or config.RETRIEVE_K
    raw = vectorstore.get(include=["documents", "metadatas"])
    texts = raw["documents"]
    metas = raw["metadatas"]

    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    semantic_results = vectorstore.similarity_search(question, k=k)
    semantic_texts = {doc.page_content for doc in semantic_results}

    bm25_scores = bm25.get_scores(question.lower().split())
    ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_texts = {texts[i] for i in ranked_idx}

    combined_texts = semantic_texts | bm25_texts
    results = [
        {"text": t, "metadata": m}
        for t, m in zip(texts, metas)
        if t in combined_texts
    ]
    return results


def retrieve_and_grade_from_store(vectorstore, question):
    candidates = hybrid_retrieve_from_store(vectorstore, question)
    graded = grade_relevance(question, candidates)

    if len(graded) < 2:
        wider = hybrid_retrieve_from_store(vectorstore, question, k=config.RETRIEVE_K * 2)
        graded = grade_relevance(question, wider)

    return graded

