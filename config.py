"""Configuration management for Sage RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Paths
BASE_DIR = Path(__file__).parent

# Sage (Technical Support) Paths
DATA_PATH = BASE_DIR / "Data_Doc_main.txt"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "curriculum.faiss"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"

# Scout (FAQ Support) Paths
FAQ_DATA_PATH = BASE_DIR / "FAQ_Doc.txt"
FAQ_EMBEDDINGS_DIR = BASE_DIR / "embeddings_faq"
FAQ_FAISS_INDEX_PATH = FAQ_EMBEDDINGS_DIR / "faq.faiss"
FAQ_METADATA_PATH = FAQ_EMBEDDINGS_DIR / "faq_metadata.json"

# Ensure embeddings directories exist
EMBEDDINGS_DIR.mkdir(exist_ok=True)
FAQ_EMBEDDINGS_DIR.mkdir(exist_ok=True)

# RAG Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's latest efficient model
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension
TOP_K_RESULTS = 3  # Number of chunks to retrieve per query

# OpenAI API Configuration
MAX_RETRIES = 3
TIMEOUT = 30  # seconds
