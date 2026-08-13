from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import config
import llm_provider


_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract factual claims from a text. Find sentences that state a "
     "specific fact, number, or technical detail that could be checked "
     "against a reference source. List each claim on its own line, with "
     "no numbering or bullets. If there are no checkable claims, respond "
     "with exactly: none"),
    ("human", "TEXT:\n{text}"),
])


def extract_claims(paper_text):
    llm = llm_provider.get_llm(fast=True)
    chain = _EXTRACT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"text": paper_text}).strip()

    if result.lower() == "none":
        return []

    claims = [line.strip() for line in result.split("\n") if line.strip()]
    return claims
