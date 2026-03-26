"""
Insert 100 demo queries for Feb 5 - Feb 24, 2026 (pre-March batch).

Sized so combined with the March is_demo2 batch (205q):
  Combined queries       : 305
  Combined tag_crew      : 37  -> 12.1%  (target 12%)
  Combined engagement    : 204 -> 66.9%  (target 67%)
  engagement = (got_it + need_help) / queries

This batch breakdown (100 queries):
  got_it        : 38
  need_help     : 32  (17 from tag_crew path + 15 from continue_here path)
  tag_crew      : 17
  continue_here : 15
  no engagement : 30

All rows tagged is_demo3=true for clean removal after event.

Cleanup SQL:
  DELETE FROM analytics_events WHERE metadata->>'is_demo3' = 'true';
"""

import os
import sys
import random
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print('[ERROR] Missing SUPABASE_URL or SUPABASE_KEY in .env')
    sys.exit(1)

client = create_client(SUPABASE_URL, SUPABASE_KEY)

WINDOW_START = datetime(2026, 2, 5, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_END   = datetime(2026, 2, 24, 23, 59, 59, tzinfo=timezone.utc)
WINDOW_DAYS  = 19

BASE_THREAD_ID = 1700000000000000000
BASE_USER_ID   = 8900000000000000000
NUM_USERS      = 38

QUERIES = [
    # -- FastAPI / Backend (12) --
    "How do I serve static files from FastAPI in production?",
    "My FastAPI app crashes on startup with 'Address already in use' - how do I kill the process on that port?",
    "How do I add websocket support to my FastAPI app for real-time updates?",
    "Getting 'CORS policy blocked' in browser even though I added CORSMiddleware - what did I miss?",
    "How do I implement pagination in FastAPI for a large list endpoint?",
    "How do I use Alembic to manage database migrations with FastAPI?",
    "How do I add request logging middleware to FastAPI so every request is logged with status code?",
    "How do I properly handle startup and shutdown events in FastAPI lifespan?",
    "Getting 'connection pool exhausted' from SQLAlchemy in my FastAPI app under load",
    "How do I return a file download response from a FastAPI endpoint?",
    "How do I write an integration test for a FastAPI endpoint that reads from PostgreSQL?",
    "My FastAPI app works locally but returns 502 Bad Gateway on Render - what to check?",

    # -- OpenAI / LLM (12) --
    "How do I use the OpenAI API to summarize a long document that is larger than the context window?",
    "What is the best way to implement a fallback when an LLM gives an unexpected output format?",
    "How do I build a simple Q&A bot over my own documents using OpenAI?",
    "How do I reduce latency in my OpenAI API calls - the responses feel slow to start?",
    "Getting 'context_length_exceeded' error from OpenAI - how do I split my input?",
    "How do I use the OpenAI moderation endpoint to filter unsafe user inputs?",
    "My chatbot loses context after a few turns - how do I keep the conversation coherent?",
    "How do I get the OpenAI API to always respond in a specific language?",
    "How do I implement a simple retry wrapper around the OpenAI API call in Python?",
    "What is the difference between system, user, and assistant roles in OpenAI chat format?",
    "How do I extract specific fields from unstructured text using OpenAI API?",
    "How do I stream OpenAI responses in a FastAPI endpoint to the browser client?",

    # -- RAG / Retrieval (10) --
    "How do I add a confidence threshold to my RAG system to avoid low-quality answers?",
    "My RAG pipeline is slow - where is the bottleneck, embedding or retrieval?",
    "How do I handle duplicate content in my RAG document corpus?",
    "What is the right way to pre-process documents before chunking for RAG?",
    "How do I build a RAG system that works over a private company knowledge base?",
    "How do I version my FAISS index so I can roll back to a previous version?",
    "How do I combine RAG with conversation history so follow-up questions work correctly?",
    "What is the difference between dense and sparse retrieval in RAG systems?",
    "How do I add query expansion to improve recall in my RAG pipeline?",
    "How do I build a RAG evaluation dataset to benchmark different chunking strategies?",

    # -- Python / Debugging (18) --
    "Getting 'SyntaxError: invalid syntax' on a line that looks correct to me - how do I find the real cause?",
    "My Python script works in Python 3.10 but fails in 3.12 - how do I find breaking changes?",
    "How do I fix 'RuntimeError: Event loop is closed' in an async Python application?",
    "Getting 'TypeError: object of type coroutine has no len' - forgot to await somewhere?",
    "How do I debug a memory leak in a long-running Python process?",
    "My Python script uses 100% CPU for a simple task - how do I find what is looping?",
    "How do I safely delete items from a dictionary while iterating over it?",
    "Getting 'json.JSONDecodeError' parsing an API response - how do I handle it gracefully?",
    "How do I time a section of Python code to measure its execution time accurately?",
    "My environment variables are not loading from .env file even with dotenv - why?",
    "How do I write a Python context manager for resource cleanup?",
    "Getting 'IndexError: list index out of range' - how do I add better bounds checking?",
    "How do I compare two large files in Python without loading both into memory?",
    "How do I mock an external API call in Python unit tests using unittest.mock?",
    "My Python daemon process exits immediately without any error message - how do I debug?",
    "How do I catch and log exceptions in a background thread in Python?",
    "Getting 'ValueError: I/O operation on closed file' - where is the file being closed early?",
    "How do I use dataclasses in Python and when are they better than plain dicts?",

    # -- Embeddings / Vector Search (8) --
    "How do I fine-tune an embedding model on my own domain-specific data?",
    "What is matryoshka embeddings and how do they help with storage efficiency?",
    "How do I run HuggingFace sentence-transformers locally without internet access?",
    "How do I compare embeddings from two different models - are they compatible?",
    "My semantic search returns results that are clearly wrong - how do I debug the embedding quality?",
    "How do I reduce embedding storage size without losing too much retrieval quality?",
    "How do I add new embeddings to an existing pgvector table efficiently?",
    "How do I choose between cosine similarity and dot product for retrieval ranking?",

    # -- Deployment / DevOps (10) --
    "How do I set up a simple CI/CD pipeline using GitHub Actions for a Python project?",
    "My Render service keeps restarting every few hours - how do I find the cause?",
    "How do I use Docker volumes to persist data between container restarts?",
    "How do I set up secrets management for my deployed app without exposing keys in config?",
    "How do I roll back a bad deployment on Render to the previous working version?",
    "How do I monitor my deployed Python app for errors in production?",
    "Getting 'exec format error' when running my Docker container on a different machine",
    "How do I reduce the Docker build time for my Python app in CI/CD?",
    "How do I set up a staging environment that mirrors production without extra cost?",
    "How do I configure Render to auto-restart my background worker if it crashes?",

    # -- LangChain / Agents (8) --
    "How do I build a simple ReAct agent in LangChain that can search the web?",
    "Getting 'context window exceeded' in my LangChain agent after a few tool calls",
    "How do I add a tool to a LangChain agent that queries my own database?",
    "How do I limit the number of steps my LangChain agent can take to prevent runaway loops?",
    "How do I stream the output of a LangChain chain to the user in real time?",
    "How do I test a LangChain agent without running expensive LLM calls every time?",
    "What is the difference between a LangChain agent and a LangGraph graph?",
    "How do I add human-in-the-loop approval to a LangChain agent before it takes an action?",

    # -- Prompt Engineering / General AI (10) --
    "How do I write a prompt that extracts structured data from messy user-submitted text?",
    "My AI application gives different answers to the same question each time - how do I make it stable?",
    "How do I implement a simple guardrail that refuses off-topic questions in my chatbot?",
    "What is constitutional AI and how can I apply similar principles to my own chatbot?",
    "How do I evaluate which of two system prompts performs better in an automated way?",
    "How do I build a multi-step AI pipeline where each step validates the previous output?",
    "What is the best way to handle a user asking my AI bot something it should not answer?",
    "How do I use an LLM to automatically classify and route incoming support tickets?",
    "How do I build an AI-powered search that understands natural language queries over my data?",
    "How do I implement output validation so my AI always returns well-formed responses?",

    # -- Supabase / Database (12) --
    "How do I use Supabase realtime to push updates to the browser without polling?",
    "How do I set up a trigger in Supabase PostgreSQL to automatically update a timestamp column?",
    "My Supabase insert is failing silently - how do I see the actual PostgreSQL error?",
    "How do I use Supabase storage to upload and serve user-uploaded files?",
    "How do I write a complex JOIN query using the Supabase Python client?",
    "How do I enable pgvector extension in Supabase for storing embeddings?",
    "Getting 'permission denied for table' error in Supabase - RLS is blocking my query?",
    "How do I import a CSV file into a Supabase table quickly?",
    "How do I set up a Supabase webhook to notify my backend when a row is inserted?",
    "How do I run a database migration on Supabase without downtime?",
    "My Supabase free tier is running out of storage - how do I audit what is taking space?",
    "How do I use Supabase auth to protect API routes in my FastAPI backend?",
]

assert len(QUERIES) == 100, f"Expected 100 queries, got {len(QUERIES)}"


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

HOUR_WEIGHTS = [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 8, 7, 8, 9, 10, 9, 8, 7, 6, 5, 3, 2]


def random_timestamp(rng: random.Random) -> datetime:
    day_offset = rng.randint(0, WINDOW_DAYS - 1)
    hour       = rng.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
    minute     = rng.randint(0, 59)
    second     = rng.randint(0, 59)
    ts = WINDOW_START + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
    return min(ts, WINDOW_END)


# ---------------------------------------------------------------------------
# Build records
# ---------------------------------------------------------------------------

def build_records() -> list[dict]:
    rng = random.Random(2026_02)

    n = len(QUERIES)  # 100
    indices = list(range(n))
    rng.shuffle(indices)

    # Sized to hit combined 12% escalation + 67% engagement with March batch
    n_tag_crew = 17   # 17 tag_crew
    n_got_it   = 38   # 38 got_it
    n_continue = 15   # 15 continue_here
    # no engagement   : 30

    # Combined verification (with March batch: 205q, 20 tag, 54 need, 80 got_it):
    # total q    = 305
    # total tag  = 37  -> 37/305 = 12.1%  [OK]
    # total eng  = (80+38 + 54+32) / 305 = 204/305 = 66.9%  [OK]
    # (need_help in this batch = 17 from tag paths + 15 from continue paths = 32)

    tag_crew_idx = set(indices[:n_tag_crew])
    got_it_idx   = set(indices[n_tag_crew : n_tag_crew + n_got_it])
    continue_idx = set(indices[n_tag_crew + n_got_it : n_tag_crew + n_got_it + n_continue])

    records = []
    tag = {'is_demo3': True}

    for i, query_text in enumerate(QUERIES):
        ts        = random_timestamp(rng)
        thread_id = str(BASE_THREAD_ID + i)
        user_id   = str(BASE_USER_ID + (i % NUM_USERS))

        records.append({
            'created_at': ts.isoformat(),
            'event_type': 'query',
            'thread_id':  thread_id,
            'user_id':    user_id,
            'message_id': None,
            'metadata':   {**tag, 'query_text': query_text},
        })

        if i in tag_crew_idx:
            t1 = ts + timedelta(minutes=rng.randint(2, 7), seconds=rng.randint(0, 59))
            t2 = t1 + timedelta(minutes=rng.randint(1, 4), seconds=rng.randint(0, 59))
            records.append({'created_at': t1.isoformat(), 'event_type': 'need_help',
                            'thread_id': thread_id, 'user_id': user_id, 'message_id': None, 'metadata': tag})
            records.append({'created_at': t2.isoformat(), 'event_type': 'tag_crew',
                            'thread_id': thread_id, 'user_id': user_id, 'message_id': None, 'metadata': tag})

        elif i in got_it_idx:
            t1 = ts + timedelta(minutes=rng.randint(1, 5), seconds=rng.randint(0, 59))
            records.append({'created_at': t1.isoformat(), 'event_type': 'got_it',
                            'thread_id': thread_id, 'user_id': user_id, 'message_id': None, 'metadata': tag})

        elif i in continue_idx:
            t1 = ts + timedelta(minutes=rng.randint(2, 6), seconds=rng.randint(0, 59))
            t2 = t1 + timedelta(minutes=rng.randint(1, 3), seconds=rng.randint(0, 59))
            records.append({'created_at': t1.isoformat(), 'event_type': 'need_help',
                            'thread_id': thread_id, 'user_id': user_id, 'message_id': None, 'metadata': tag})
            records.append({'created_at': t2.isoformat(), 'event_type': 'continue_here',
                            'thread_id': thread_id, 'user_id': user_id, 'message_id': None, 'metadata': tag})

    return records


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def insert_batches(records: list[dict], batch_size: int = 100):
    total = len(records)
    inserted = 0
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        client.table('analytics_events').insert(batch).execute()
        inserted += len(batch)
        print(f'[*] Inserted {inserted}/{total}')
    print(f'[OK] All {total} records inserted')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('[*] Building Feb 5-24 demo records...')
    records = build_records()

    by_type: dict[str, int] = {}
    for r in records:
        et = r['event_type']
        by_type[et] = by_type.get(et, 0) + 1

    n_q    = by_type.get('query', 0)
    n_g    = by_type.get('got_it', 0)
    n_t    = by_type.get('tag_crew', 0)
    n_need = by_type.get('need_help', 0)
    n_c    = by_type.get('continue_here', 0)

    print(f'[OK] Records built: {len(records)} total')
    print(f'     queries      : {n_q}')
    print(f'     got_it       : {n_g}')
    print(f'     need_help    : {n_need}')
    print(f'     tag_crew     : {n_t}')
    print(f'     continue_here: {n_c}')

    # Combined projections with March batch
    combined_q    = 205 + n_q
    combined_t    = 20  + n_t
    combined_eng  = (80 + n_g + 54 + n_need) / combined_q * 100
    combined_esc  = combined_t / combined_q * 100
    print()
    print(f'  -- Combined with March batch --')
    print(f'     total queries: {combined_q}')
    print(f'     engagement   : {combined_eng:.1f}%  (target 67%)')
    print(f'     escalation   : {combined_esc:.1f}%  (target 12%)')
    print()

    print('[*] Inserting into Supabase...')
    insert_batches(records)

    print()
    print('[OK] Done. Data spans Feb 5 - Feb 24, 2026.')
    print()
    print('Cleanup SQL (run after event):')
    print("  DELETE FROM analytics_events WHERE metadata->>'is_demo3' = 'true';")


if __name__ == '__main__':
    main()
