from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "papers"

LLM_MODEL = "llama3.1"
FAST_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "nomic-embed-text"

USE_GROQ = True
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
RETRIEVE_K = 8

