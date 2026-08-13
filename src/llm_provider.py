from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import config


def get_llm(fast=False):
    if config.USE_GROQ:
        return ChatGroq(
            model=config.GROQ_MODEL,
            temperature=0,
            api_key=config.GROQ_API_KEY,
        )
    else:
        model = config.FAST_MODEL if fast else config.LLM_MODEL
        return ChatOllama(model=model, temperature=0)
    