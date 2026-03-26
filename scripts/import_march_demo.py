"""
Insert 200 realistic demo queries for March 2026 event.
Date range: Feb 22 - Mar 8, 2026

Stats produced:
  - 205 query events
  - ~10% mentor escalations (tag_crew) -> 20 queries  (9.8%)
  - ~67% engagement rate               -> (gotItCount + needHelpCount) / 205 ~ 0.67
    breakdown: 80 got_it, 54 need_help (20 from tag_crew path + 34 from continue_here path)
  - Engagement formula: (got_it + need_help) / queries = (80+54)/205 = 65.4%

All rows tagged is_demo2=true for clean removal after event.

Cleanup SQL:
  DELETE FROM analytics_events WHERE metadata->>'is_demo2' = 'true';
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

WINDOW_START = datetime(2026, 2, 22, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_END   = datetime(2026, 3,  8, 23, 59, 59, tzinfo=timezone.utc)
WINDOW_DAYS  = 15

BASE_THREAD_ID = 1600000000000000000
BASE_USER_ID   = 9200000000000000000
NUM_USERS      = 45

# ---------------------------------------------------------------------------
# 200 contextual queries from an AI engineering cohort (no duplicates)
# ---------------------------------------------------------------------------
QUERIES = [
    # -- FastAPI (20) --
    "My FastAPI endpoint returns 422 when I send a POST request but the JSON body looks correct to me",
    "How do I add CORS middleware to FastAPI so my React frontend can call it without blocked errors?",
    "What is the difference between async def and def for FastAPI route handlers and when do I use each?",
    "FastAPI throws RuntimeError: Task attached to a different loop when I run it with uvicorn - how do I fix?",
    "How do I handle file uploads in FastAPI and save them to disk?",
    "My FastAPI server starts fine but returns 500 on every request and I cannot see the actual error message",
    "How do I return streaming responses from FastAPI for LLM output so the user sees tokens as they arrive?",
    "How do I write pytest tests for FastAPI endpoints that hit my database?",
    "Getting 'module not found' error when I run uvicorn main:app even though the file exists",
    "How do I structure a larger FastAPI project with multiple routers in separate files?",
    "How do I add JWT authentication to my FastAPI routes and protect certain endpoints?",
    "My FastAPI endpoint hangs when calling an external API - how do I add a timeout to the HTTP call?",
    "How do I use Pydantic v2 models for request body validation in FastAPI?",
    "How do I connect FastAPI to Supabase PostgreSQL using an async driver?",
    "FastAPI is extremely slow for the first request after idle - is this a cold start and how do I fix it?",
    "How do I add background tasks in FastAPI that run after the response is returned?",
    "How do I return different HTTP status codes from FastAPI depending on a condition?",
    "Getting 'value is not a valid dict' error in FastAPI when I send a nested Pydantic model",
    "How do I implement rate limiting in FastAPI to prevent API abuse?",
    "How do I add API key authentication via request headers to my FastAPI app?",

    # -- OpenAI API (20) --
    "Getting RateLimitError from OpenAI API even though I just created my account - what do I do?",
    "How do I use streaming with the OpenAI Python SDK so responses appear token by token?",
    "What is the actual difference between gpt-4o and gpt-4o-mini in terms of quality vs cost?",
    "My OpenAI API call returns truncated responses - how do I increase max_tokens correctly?",
    "Getting 'Invalid API key provided' even though I copied the key exactly from the dashboard",
    "How do I use function calling / tool calling with the OpenAI API to get structured output?",
    "How do I count tokens before sending to OpenAI so I do not accidentally exceed the limit?",
    "My context window fills up in long conversations - what is the right strategy to handle this?",
    "How do I send an image to GPT-4 Vision API along with a text prompt?",
    "My OpenAI API costs jumped this month - how do I figure out which calls are most expensive?",
    "How do I implement retry logic with exponential backoff for OpenAI API calls?",
    "How do I reliably parse JSON from OpenAI responses when the model sometimes adds extra text?",
    "What is the new structured outputs feature in OpenAI and how is it different from function calling?",
    "How do I cache OpenAI API responses locally to avoid paying for the same query twice?",
    "How do I handle the case where OpenAI API is down so my app degrades gracefully?",
    "What temperature should I use for a factual Q&A bot vs a creative writing assistant?",
    "How do I batch multiple OpenAI embedding requests efficiently to reduce latency?",
    "My GPT-4o response cuts off mid-sentence even with a high max_tokens value - why?",
    "How do I add a system prompt that the user cannot override in my OpenAI API calls?",
    "How do I use the OpenAI Assistants API vs the plain chat completions API - when to use each?",

    # -- RAG Systems (20) --
    "How does RAG work conceptually - why do we need it when the LLM already knows a lot?",
    "What chunk size should I use when splitting documents for RAG and does overlap matter?",
    "How do I evaluate whether my RAG system is actually retrieving the right chunks?",
    "My RAG system retrieves irrelevant chunks even for simple questions - how do I improve recall?",
    "How do I handle the case where no relevant documents are found and avoid hallucination?",
    "What is the difference between RAG and fine-tuning and when should I choose each?",
    "How do I add metadata filtering so RAG only searches within a specific category of documents?",
    "What is reranking and should I add a reranker on top of my vector retrieval?",
    "How do I update the FAISS index when new documents are added without rebuilding from scratch?",
    "My RAG responses are too vague even when the right chunk is retrieved - how do I fix the prompt?",
    "How do I implement hybrid search that combines semantic vector search with BM25 keyword search?",
    "What is parent-child chunking in RAG and when is it better than regular fixed-size chunking?",
    "How do I build a RAG system that cites which document each part of the answer came from?",
    "How do I handle very long documents like entire books that exceed the context window in RAG?",
    "My RAG system works in testing but gives poor answers in production - what could have changed?",
    "How do I implement multi-turn conversation memory in a RAG chatbot?",
    "How do I measure hallucination rate in my RAG system automatically?",
    "How do I build a RAG pipeline that handles both PDF and plain text documents?",
    "What is the optimal overlap when chunking documents for RAG and how do I tune it?",
    "How do I do RAG over a large SQL database instead of a document corpus?",

    # -- FAISS / Vector DBs (15) --
    "What is the difference between FAISS and Pinecone and which should I use for my project?",
    "How do I save a FAISS index to disk and reload it without rebuilding embeddings?",
    "Getting index out of bounds error when searching FAISS - what does this mean and how to fix?",
    "How do I add new documents to an existing FAISS index without rebuilding the whole thing?",
    "What is the difference between IndexFlatL2 and IndexFlatIP in FAISS and which is better?",
    "How many vectors can FAISS hold before performance degrades significantly?",
    "How do I use FAISS with cosine similarity instead of L2 Euclidean distance?",
    "When should I use a cloud vector database like Pinecone vs local FAISS for my use case?",
    "How do I do batched similarity search in FAISS for multiple queries at once?",
    "My FAISS search results do not seem relevant - could the embedding model be the problem?",
    "How do I store and retrieve metadata alongside vectors in FAISS?",
    "How do I install faiss-cpu on Windows without getting a compilation error?",
    "What does the distance score from FAISS search actually represent and how do I use it as a threshold?",
    "How do I choose between IVF and flat indexes in FAISS for a 500k vector dataset?",
    "How do I normalize embeddings before adding to FAISS to get cosine similarity scores?",

    # -- Python Debugging (30) --
    "Getting ImportError cannot import name X from Y even though the class exists in that file",
    "My Python script works perfectly locally but crashes on the server with a missing module error",
    "What does TypeError unsupported operand type str and int mean and how do I find where it happens?",
    "How do I use pdb to step through a Python script and inspect variables at runtime?",
    "My async function is blocking the event loop and making everything slow - how do I find the culprit?",
    "Getting RecursionError maximum recursion depth exceeded - how do I debug and fix infinite recursion?",
    "How do I safely handle None values in Python to avoid AttributeError on None?",
    "Getting KeyError when accessing a dictionary - how do I handle missing keys gracefully?",
    "How do I read a Python traceback properly to find where the actual error originated?",
    "My for loop is modifying the list it is iterating over and skipping elements - is that the bug?",
    "Getting UnicodeDecodeError when reading a CSV file - how do I fix the encoding issue?",
    "How do I profile Python code to identify which function is causing slowness?",
    "Getting MemoryError when processing a large dataset - how do I handle it without loading all at once?",
    "How do I catch multiple exception types in a single try except block in Python?",
    "My Python script silently crashes after a few minutes with no error message - how do I debug?",
    "Getting OSError Address already in use when starting my FastAPI server - port conflict?",
    "How do I fix cannot unpack non-iterable NoneType object error in Python?",
    "My function is returning None instead of the expected value - how do I trace where the return is missing?",
    "How do I use Python logging module properly instead of scattered print statements?",
    "Getting circular import error in my Python project - what is the right way to restructure imports?",
    "How do I process a 5GB JSON file in Python without running out of memory?",
    "My Python virtual environment is broken after a system update - how do I safely recreate it?",
    "Getting PermissionError when writing a file on Linux - how do I fix file permissions?",
    "How do I fix AttributeError datetime object has no attribute utcnow in Python 3.12?",
    "Getting StopIteration error inside a generator - what does this mean in newer Python?",
    "My Python script hangs indefinitely with no output - how do I find where it is stuck?",
    "Getting SSL certificate verify failed when making requests to an external API - how to fix?",
    "How do I convert between different timezone-aware datetime formats in Python reliably?",
    "Why is my variable None inside an async callback even though I set it before the await?",
    "Getting RuntimeError cannot use a string pattern on a bytes-like object - encoding issue?",

    # -- Docker / Render Deployment (15) --
    "My Docker container starts but my app is not accessible on the expected port - what is wrong?",
    "How do I pass environment variables securely to a Docker container in production?",
    "My Docker image is 3GB - how do I reduce image size with a proper multi-stage build?",
    "Getting permission denied error inside a Docker container when writing to a mounted volume",
    "How do I set up Docker Compose for a FastAPI app plus a PostgreSQL database?",
    "My app works fine locally but fails after Docker build with a different Python version mismatch",
    "How do I use .dockerignore correctly to exclude node_modules and .env from the build context?",
    "How do I deploy my Python FastAPI app to Render as a web service?",
    "My Render deployment keeps failing during the pip install step with a dependency conflict",
    "How do I set environment variables on Render so my app can read them at runtime?",
    "What is the difference between a Render web service and a background worker and when to use each?",
    "How do I view real-time logs for my deployed app on Render to debug a crash?",
    "My app keeps running out of memory on Render - how do I diagnose memory leaks?",
    "How do I set up a GitHub Actions workflow to auto-deploy to Render on every push to main?",
    "How do I add a /health endpoint so Render health checks pass and my service stays up?",

    # -- LangChain (15) --
    "What is the practical difference between using LangChain vs calling the OpenAI API directly?",
    "How do I create a simple retrieval QA chain in LangChain for document question answering?",
    "Getting deprecation warnings throughout my LangChain code - how do I migrate to the new LCEL syntax?",
    "How do I use ConversationBufferMemory in LangChain to maintain conversation history?",
    "How do I load and split a PDF document using LangChain document loaders?",
    "My LangChain RetrievalQA chain returns wrong answers even when the relevant chunks are in the index",
    "How do I add a custom system prompt to my LangChain RAG chain?",
    "What is LCEL and should I rewrite my old LangChain chains to use it?",
    "How do I use LangChain callbacks to log every LLM call with its input and output?",
    "How do I switch my LangChain chain from OpenAI to a locally running Ollama model?",
    "How do I create a LangChain agent that can use multiple tools like search and calculator?",
    "Getting OutputParserException in LangChain - the LLM is not returning the expected format",
    "How do I use LangChain with Supabase pgvector as the vector store instead of FAISS?",
    "How do I handle context window overflow in LangChain when documents are too long?",
    "How do I mock the LLM in LangChain unit tests so I do not make real API calls during testing?",

    # -- Embeddings (10) --
    "Which embedding model should I use - OpenAI text-embedding-3-small or a HuggingFace model?",
    "Why do I need to embed text at all before doing similarity search - why not just use keywords?",
    "My embedding similarity scores are all very high regardless of relevance - what is wrong?",
    "How do I batch embed thousands of text chunks efficiently without hitting rate limits?",
    "How do I compare two sentences for semantic similarity and set a meaningful threshold?",
    "What is the right way to normalize embeddings before cosine similarity comparison?",
    "How do I check if two document chunks are semantically duplicate using embeddings?",
    "How do I store text embeddings in PostgreSQL using the pgvector extension?",
    "What is the difference between symmetric and asymmetric embedding models for retrieval?",
    "How do I choose the right embedding dimension - does higher dimension always mean better quality?",

    # -- Prompt Engineering (10) --
    "How do I write a prompt that reliably gets JSON output from the LLM without extra text?",
    "My LLM keeps ignoring the system prompt constraints - how do I enforce them more strictly?",
    "What is chain-of-thought prompting and when does it actually improve accuracy?",
    "How do I write effective few-shot examples in my prompt to guide the model's output format?",
    "My prompt works well with GPT-4o but gives poor results with GPT-4o-mini - how do I adapt it?",
    "How do I prevent the LLM from making up information it does not know - ground it better?",
    "What is the difference between zero-shot and few-shot prompting in practice?",
    "How do I structure a classification prompt so the model only outputs the label, nothing else?",
    "How do I use XML or markdown delimiters in prompts to separate instructions from content?",
    "My LLM responses vary a lot between runs - how do I make outputs more consistent and deterministic?",

    # -- General AI / ML Concepts (15) --
    "What is the difference between fine-tuning a model and using prompt engineering - when to use each?",
    "How does the attention mechanism in transformers work at a conceptual level?",
    "What is a context window limit and how does it affect my application design?",
    "How do I evaluate the quality of my AI application's responses in an automated way?",
    "What is hallucination in LLMs and what are the main strategies to reduce it?",
    "When should I use an AI agent vs a simple prompt chain vs a single LLM call?",
    "How do I implement token streaming in my web app for a better user experience?",
    "What is the practical difference between temperature and top_p sampling parameters?",
    "How do I build an AI application that stays within a fixed monthly API cost budget?",
    "What is the difference between semantic search and BM25 keyword search and when to combine?",
    "How do I add persistent conversation memory to my AI chatbot across sessions?",
    "What is grounding in AI systems and how does RAG provide factual grounding?",
    "How do I handle multi-language queries in my AI application that is primarily in English?",
    "What is the right way to handle user PII data in an AI application for compliance?",
    "How do I implement A/B testing to compare two different prompt versions in production?",

    # -- Streamlit / UI (10) --
    "How do I build a simple streaming chat interface using Streamlit and the OpenAI API?",
    "How do I deploy my Streamlit app publicly so others can access it without running locally?",
    "Getting StreamlitAPIException when I use st.session_state - what am I doing wrong?",
    "How do I add simple password authentication to my Streamlit demo app?",
    "How do I stream LLM token responses in Streamlit so the user sees text appear in real time?",
    "What is the difference between st.cache_resource and st.cache_data in Streamlit?",
    "My Streamlit app re-runs the entire script on every button click and resets state - how to fix?",
    "How do I add file upload and parse the uploaded CSV in Streamlit?",
    "How do I display a pandas DataFrame with custom formatting and column hiding in Streamlit?",
    "How do I customize the layout, colors, and fonts of my Streamlit app beyond the defaults?",

    # -- Auth / Security (10) --
    "How do I implement JWT token authentication in FastAPI with refresh token support?",
    "How do I store API keys and secrets securely so they are not hardcoded in my source code?",
    "I accidentally committed my .env file with API keys to a public GitHub repo - what do I do now?",
    "How do I add proper rate limiting to prevent my API from being abused or scraped?",
    "How do I implement OAuth2 login with Google in a FastAPI backend?",
    "What is the correct way to hash and verify passwords in Python using bcrypt?",
    "How do I validate that an incoming webhook request is genuinely from Discord or Stripe?",
    "How do I implement role-based access control so only admins can hit certain endpoints?",
    "How do I prevent prompt injection attacks in my AI application that processes user input?",
    "How do I audit log every user action in my application for compliance and debugging?",

    # -- Supabase / Database (15) --
    "How do I query Supabase from Python and handle the 1000 row default limit correctly?",
    "How do I insert a batch of 500 rows into Supabase without hitting request size limits?",
    "How do I filter Supabase rows by a nested JSON field inside a metadata column?",
    "Getting 'JWT expired' error from Supabase - how do I refresh the service role key?",
    "How do I use Supabase Row Level Security to restrict data access per user?",
    "How do I run raw SQL queries against my Supabase PostgreSQL database from Python?",
    "How do I set up Supabase realtime subscriptions to listen for new row inserts?",
    "What is the difference between the Supabase anon key and service role key and when to use each?",
    "How do I paginate through all rows in a Supabase table when there are more than 1000?",
    "Getting 'duplicate key violates unique constraint' error on Supabase upsert - how do I handle?",
    "How do I back up my Supabase database and restore it if something goes wrong?",
    "How do I add a full-text search index to a Supabase PostgreSQL table?",
    "How do I connect to Supabase from a Docker container using the service role key?",
    "How do I use Supabase edge functions to run server-side logic without a separate backend?",
    "My Supabase query is very slow on a 100k row table - how do I add an index to fix it?",
]

assert len(QUERIES) == 205, f"Expected 205 queries, got {len(QUERIES)}"


# ---------------------------------------------------------------------------
# Timestamp with realistic hour weighting
# ---------------------------------------------------------------------------

# Hour weights: low midnight-6am, ramp up morning, peak afternoon/evening, drop late night
HOUR_WEIGHTS = [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 8, 7, 8, 9, 10, 9, 8, 7, 6, 5, 3, 2]


def random_timestamp(rng: random.Random) -> datetime:
    day_offset = rng.randint(0, WINDOW_DAYS - 1)
    hour       = rng.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
    minute     = rng.randint(0, 59)
    second     = rng.randint(0, 59)
    ts = WINDOW_START + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
    return min(ts, WINDOW_END)


# ---------------------------------------------------------------------------
# Build all records
# ---------------------------------------------------------------------------

def build_records() -> list[dict]:
    rng = random.Random(2026)

    n = len(QUERIES)          # 200
    indices = list(range(n))
    rng.shuffle(indices)

    # Partition indices into outcome buckets
    # 20 tag_crew, 80 got_it, 34 continue_here, 66 no-engagement
    n_tag_crew    = 20
    n_got_it      = 80
    n_continue    = 34
    # remaining 66 are no-engagement

    tag_crew_idx  = set(indices[:n_tag_crew])
    got_it_idx    = set(indices[n_tag_crew : n_tag_crew + n_got_it])
    continue_idx  = set(indices[n_tag_crew + n_got_it : n_tag_crew + n_got_it + n_continue])

    # Verify engagement formula: (got_it + need_help) / 200
    # need_help = n_tag_crew + n_continue = 20 + 34 = 54
    # engagement = (80 + 54) / 200 = 67%  [OK]
    # escalation = 20 / 200 = 10%         [OK]

    records = []

    for i, query_text in enumerate(QUERIES):
        ts         = random_timestamp(rng)
        thread_id  = str(BASE_THREAD_ID + i)
        user_id    = str(BASE_USER_ID + (i % NUM_USERS))
        tag        = {'is_demo2': True}

        # --- query event ---
        records.append({
            'created_at': ts.isoformat(),
            'event_type': 'query',
            'thread_id':  thread_id,
            'user_id':    user_id,
            'message_id': None,
            'metadata':   {**tag, 'query_text': query_text},
        })

        # --- feedback events ---
        if i in tag_crew_idx:
            t1 = ts + timedelta(minutes=rng.randint(2, 6), seconds=rng.randint(0, 59))
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
        # else: no engagement - no additional events

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
    print('[*] Building records...')
    records = build_records()

    # Stats summary
    by_type: dict[str, int] = {}
    for r in records:
        et = r['event_type']
        by_type[et] = by_type.get(et, 0) + 1

    n_queries  = by_type.get('query', 0)
    n_got_it   = by_type.get('got_it', 0)
    n_tag_crew = by_type.get('tag_crew', 0)
    n_need     = by_type.get('need_help', 0)
    n_continue = by_type.get('continue_here', 0)
    total      = len(records)

    print(f'[OK] Records built: {total} total')
    print(f'     queries      : {n_queries}')
    print(f'     got_it       : {n_got_it}')
    print(f'     need_help    : {n_need}')
    print(f'     tag_crew     : {n_tag_crew}')
    print(f'     continue_here: {n_continue}')
    engagement = round((n_got_it + n_need) / n_queries * 100, 1) if n_queries else 0
    escalation = round(n_tag_crew / n_queries * 100, 1) if n_queries else 0
    print(f'     engagement   : {engagement}%  (target: 67%)')
    print(f'     escalation   : {escalation}%  (target: 10%)')
    print()

    print('[*] Inserting into Supabase...')
    insert_batches(records)

    print()
    print('[OK] Done. Data spans Feb 22 - Mar 8, 2026.')
    print()
    print('Cleanup SQL (run after event):')
    print("  DELETE FROM analytics_events WHERE metadata->>'is_demo2' = 'true';")


if __name__ == '__main__':
    main()
