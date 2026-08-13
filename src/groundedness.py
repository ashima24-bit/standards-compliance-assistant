from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import config
import llm_provider


_GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fact-checker. Compare the ANSWER against the SOURCE TEXT. "
"Judge based on MEANING, not exact wording. The answer may combine "
"multiple true facts from different parts of the source text into one "
"sentence - this counts as supported, as long as each individual fact "
"in the answer is actually present somewhere in the source. Only mark "
"'unsupported' if the answer contains a claim that is not present in "
"the source at all, or that contradicts it. Respond with exactly one "
"word first: 'supported' or 'unsupported'. Then, on a new line, give "
"one short sentence explaining why."),
    ("human", "ANSWER:\n{answer}\n\nSOURCE TEXT:\n{source}"),
])


def check_groundedness(answer, chunks):
    llm = llm_provider.get_llm(fast=True)
    chain = _GROUNDEDNESS_PROMPT | llm | StrOutputParser()

    combined_source = "\n\n---\n\n".join(c["text"] for c in chunks)
    if not combined_source.strip():
        return {"verdict": "unsupported", "explanation": "No source text was retrieved."}

    result = chain.invoke({"answer": answer, "source": combined_source}).strip()
    lines = result.split("\n", 1)
    verdict = "supported" if lines[0].strip().lower().startswith("s") else "unsupported"
    explanation = lines[1].strip() if len(lines) > 1 else ""
    return {"verdict": verdict, "explanation": explanation}
