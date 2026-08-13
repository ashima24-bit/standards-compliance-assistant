from langchain_ollama import ChatOllama
import config
import retrieval
import groundedness
import llm_provider


def build_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"(page {c['metadata'].get('page') + 1})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question):
    print("Step 1: Retrieving candidates (hybrid search)...")
    chunks = retrieval.retrieve_and_grade(question)
    print(f"Step 1 done: {len(chunks)} relevant chunk(s) found.")

    if not chunks:
        return {
            "question": question,
            "answer": "Not covered in the provided document.",
            "status": "Unverified / Not Found",
            "sources": [],
        }

    print("Step 2: Generating answer...")
    context = build_context(chunks)
    prompt = (
    "Answer the QUESTION using only the CONTEXT below. Do not mention page "
    "numbers in your answer - page citations are shown separately. If the "
    "context is insufficient, say so.\n\n"
    f"CONTEXT:\n{context}\n\nQUESTION: {question}"
)
    llm = llm_provider.get_llm()
    raw_answer = llm.invoke(prompt).content
    print("Step 2 done.")

    print("Step 3: Checking groundedness...")
    check = groundedness.check_groundedness(raw_answer, chunks)
    print("Step 3 done.")

    status = "Verified" if check["verdict"] == "supported" else "Unverified"
    final_answer = raw_answer
    if status == "Unverified":
        final_answer += f"\n\n(Note: could not be fully verified - {check['explanation']})"

    return {
        "question": question,
        "answer": final_answer,
        "status": status,
        "sources": chunks,
    }


def answer_question_from_store(vectorstore, question):
    chunks = retrieval.retrieve_and_grade_from_store(vectorstore, question)

    if not chunks:
        return {
            "question": question,
            "answer": "Not covered in the uploaded document.",
            "status": "Unverified / Not Found",
            "sources": [],
        }

    context = build_context(chunks)
    prompt = (
    "Answer the QUESTION using only the CONTEXT below. Do not mention page "
    "numbers in your answer - page citations are shown separately. If the "
    "context is insufficient, say so.\n\n"
    f"CONTEXT:\n{context}\n\nQUESTION: {question}"
)
    llm = llm_provider.get_llm()
    raw_answer = llm.invoke(prompt).content

    check = groundedness.check_groundedness(raw_answer, chunks)
    status = "Verified" if check["verdict"] == "supported" else "Unverified"

    final_answer = raw_answer
    if status == "Unverified":
        final_answer += f"\n\n(Note: could not be fully verified - {check['explanation']})"

    return {
        "question": question,
        "answer": final_answer,
        "status": status,
        "sources": chunks,
    }


if __name__ == "__main__":
    result = answer_question("What is multi-head attention?")
    print("\nAnswer:", result["answer"])
    print("Status:", result["status"])
    print("Sources:")
    for c in result["sources"]:
        print(f"  - page {c['metadata'].get('page') + 1}")
        