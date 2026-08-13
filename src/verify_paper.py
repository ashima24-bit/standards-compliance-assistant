from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import config
import retrieval
import groundedness
import extract_claims
import llm_provider


_SUGGESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "The CLAIM below could not be verified against the SOURCE TEXT. "
     "In one short sentence, state what the source text actually says "
     "and how the claim should be corrected. If the source text has no "
     "relevant information at all, say so plainly."),
    ("human", "CLAIM: {claim}\n\nSOURCE TEXT:\n{source}"),
])


def suggest_correction(claim, chunks):
    llm = llm_provider.get_llm(fast=True)
    chain = _SUGGESTION_PROMPT | llm | StrOutputParser()
    combined_source = "\n\n---\n\n".join(c["text"] for c in chunks)
    return chain.invoke({"claim": claim, "source": combined_source}).strip()


def verify_paper_against_builtin(paper_text):
    claims = extract_claims.extract_claims(paper_text)
    results = []

    for claim in claims:
        chunks = retrieval.retrieve_and_grade(claim)
        results.append(_evaluate_claim(claim, chunks))

    return {"total_claims": len(claims), "results": results}


def verify_paper_against_store(vectorstore, paper_text):
    claims = extract_claims.extract_claims(paper_text)
    results = []

    for claim in claims:
        chunks = retrieval.retrieve_and_grade_from_store(vectorstore, claim)
        results.append(_evaluate_claim(claim, chunks))

    return {"total_claims": len(claims), "results": results}


def _evaluate_claim(claim, chunks):
    if not chunks:
        return {
            "claim": claim,
            "verdict": "Unsupported",
            "explanation": "No matching content found in the reference document(s).",
            "suggestion": None,
            "sources": [],
        }

    check = groundedness.check_groundedness(claim, chunks)

    if check["verdict"] == "supported":
        return {
            "claim": claim,
            "verdict": "Verified",
            "explanation": check["explanation"],
            "suggestion": None,
            "sources": chunks,
        }
    else:
        suggestion = suggest_correction(claim, chunks)
        return {
            "claim": claim,
            "verdict": "Outdated / Unsupported",
            "explanation": check["explanation"],
            "suggestion": suggestion,
            "sources": chunks,
        }


if __name__ == "__main__":
    sample_paper = (
        "The Transformer model uses a dimension of 256 for its embeddings. "
        "It uses 4 attention heads in the base model. Training was done "
        "using the Adam optimizer."
    )
    report = verify_paper_against_builtin(sample_paper)
    print(f"Extracted {report['total_claims']} claim(s).\n")
    for r in report["results"]:
        print(f"Claim: {r['claim']}")
        print(f"Verdict: {r['verdict']}")
        print(f"Explanation: {r['explanation']}")
        if r["suggestion"]:
            print(f"Suggested correction: {r['suggestion']}")
        print()
        