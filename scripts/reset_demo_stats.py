"""
Clean slate demo: delete ALL data in 30-day window, insert exactly 1047 fresh queries.

Targets:
  total queries : 1047
  engagement    : 67%  -> (got_it + need_help) = 701
  escalation    : 12%  -> tag_crew = 126

Breakdown:
  got_it        : 470
  need_help     : 231  (126 from tag_crew path + 105 from continue_here path)
  tag_crew      : 126
  continue_here : 105
  no engagement : 346

All demo rows tagged is_demo5=true.

Cleanup SQL:
  DELETE FROM analytics_events WHERE metadata->>'is_demo5' = 'true';
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

WINDOW_START     = datetime(2026, 2, 5,  0,  0, 0, tzinfo=timezone.utc)
WINDOW_END       = datetime(2026, 3, 8, 23, 59, 59, tzinfo=timezone.utc)
WINDOW_SPAN_DAYS = 32

BASE_THREAD_ID = 1900000000000000000
BASE_USER_ID   = 8700000000000000000
NUM_USERS      = 52

# Exact targets
TOTAL_QUERIES = 1047
N_TAG_CREW    = 126   # 12.03%
N_GOT_IT      = 470
N_CONTINUE    = 105
N_NEED_HELP   = N_TAG_CREW + N_CONTINUE   # 231
N_NO_ENG      = TOTAL_QUERIES - N_TAG_CREW - N_GOT_IT - N_CONTINUE  # 346

# Quick sanity
assert N_NO_ENG >= 0, "Engagement counts exceed total queries"
_eng = N_GOT_IT + N_NEED_HELP
print(f'[*] Planned: {TOTAL_QUERIES} queries | '
      f'engagement {_eng/TOTAL_QUERIES*100:.1f}% | '
      f'escalation {N_TAG_CREW/TOTAL_QUERIES*100:.1f}%')

# ---------------------------------------------------------------------------
# Query pool (305 unique; will be cycled to reach 1047)
# ---------------------------------------------------------------------------
QUERIES_POOL = [
    # -- FastAPI (25) --
    "My FastAPI endpoint returns 422 when I send a POST request but the JSON body looks correct to me",
    "How do I add CORS middleware to FastAPI so my React frontend can call it without blocked errors?",
    "What is the difference between async def and def for FastAPI route handlers?",
    "FastAPI throws RuntimeError: Task attached to a different loop with uvicorn - how do I fix?",
    "How do I handle file uploads in FastAPI and save them to disk?",
    "My FastAPI server starts fine but returns 500 on every request and I cannot see the error",
    "How do I return streaming responses from FastAPI for LLM output?",
    "How do I write pytest tests for FastAPI endpoints that hit my database?",
    "Getting module not found error when I run uvicorn main:app even though the file exists",
    "How do I structure a larger FastAPI project with multiple routers in separate files?",
    "How do I add JWT authentication to my FastAPI routes?",
    "My FastAPI endpoint hangs when calling an external API - how do I add a timeout?",
    "How do I use Pydantic v2 models for request body validation in FastAPI?",
    "How do I connect FastAPI to Supabase PostgreSQL using an async driver?",
    "FastAPI is very slow for the first request after idle - is this a cold start issue?",
    "How do I add background tasks in FastAPI that run after the response is returned?",
    "How do I return different HTTP status codes from FastAPI depending on a condition?",
    "Getting value is not a valid dict error in FastAPI with a nested Pydantic model",
    "How do I implement rate limiting in FastAPI to prevent API abuse?",
    "How do I add API key authentication via request headers to my FastAPI app?",
    "How do I serve static files from FastAPI in production?",
    "Getting CORS policy blocked in browser even though I added CORSMiddleware - what did I miss?",
    "How do I implement pagination in a FastAPI endpoint for a large list response?",
    "How do I use Alembic to manage database migrations with FastAPI and SQLAlchemy?",
    "How do I add request logging middleware to FastAPI so every request is logged with status code?",
    # -- OpenAI API (25) --
    "Getting RateLimitError from OpenAI API even though I just created my account - what do I do?",
    "How do I use streaming with the OpenAI Python SDK so responses appear token by token?",
    "What is the actual difference between gpt-4o and gpt-4o-mini in terms of quality vs cost?",
    "My OpenAI API call returns truncated responses - how do I increase max_tokens correctly?",
    "Getting Invalid API key provided even though I copied the key exactly from the dashboard",
    "How do I use function calling with the OpenAI API to get structured output?",
    "How do I count tokens before sending to OpenAI so I do not accidentally exceed the limit?",
    "My context window fills up in long conversations - what is the right strategy?",
    "How do I send an image to GPT-4 Vision API along with a text prompt?",
    "My OpenAI API costs jumped this month - how do I figure out which calls are most expensive?",
    "How do I implement retry logic with exponential backoff for OpenAI API calls?",
    "How do I reliably parse JSON from OpenAI responses when the model adds extra text?",
    "What is the structured outputs feature in OpenAI and how is it different from function calling?",
    "How do I cache OpenAI API responses locally to avoid paying for the same query twice?",
    "How do I handle the case where OpenAI API is down so my app degrades gracefully?",
    "What temperature should I use for a factual Q&A bot vs a creative writing assistant?",
    "How do I batch multiple OpenAI embedding requests efficiently to reduce latency?",
    "My GPT-4o response cuts off mid-sentence even with a high max_tokens value - why?",
    "How do I add a system prompt that the user cannot override in my OpenAI API calls?",
    "How do I use the OpenAI Assistants API vs the plain chat completions API?",
    "How do I use the OpenAI API to summarize a long document larger than the context window?",
    "How do I reduce latency in my OpenAI API calls - the responses feel slow to start?",
    "Getting context_length_exceeded error from OpenAI - how do I split my input?",
    "My chatbot loses context after a few turns - how do I keep the conversation coherent?",
    "How do I extract specific fields from unstructured text using the OpenAI API?",
    # -- RAG Systems (25) --
    "How does RAG work conceptually - why do we need it when the LLM already knows a lot?",
    "What chunk size should I use when splitting documents for RAG and does overlap matter?",
    "How do I evaluate whether my RAG system is actually retrieving the right chunks?",
    "My RAG system retrieves irrelevant chunks even for simple questions - how do I improve recall?",
    "How do I handle the case where no relevant documents are found in RAG to avoid hallucination?",
    "What is the difference between RAG and fine-tuning and when should I choose each?",
    "How do I add metadata filtering so RAG only searches within a specific document category?",
    "What is reranking and should I add a reranker on top of my vector retrieval?",
    "How do I update the FAISS index when new documents are added without full rebuild?",
    "My RAG responses are too vague even when the right chunk is retrieved - how do I fix the prompt?",
    "How do I implement hybrid search that combines semantic vector search with BM25?",
    "What is parent-child chunking in RAG and when is it better than fixed-size chunking?",
    "How do I build a RAG system that cites which document each part of the answer came from?",
    "How do I handle very long documents that exceed the context window in RAG?",
    "My RAG system works in testing but gives poor answers in production - what changed?",
    "How do I implement multi-turn conversation memory in a RAG chatbot?",
    "How do I measure hallucination rate in my RAG system automatically?",
    "How do I build a RAG pipeline that handles both PDF and plain text documents?",
    "How do I add a confidence threshold to my RAG system to avoid low-quality answers?",
    "My RAG pipeline is slow - where is the bottleneck, embedding or retrieval?",
    "How do I handle duplicate content in my RAG document corpus?",
    "What is the right way to pre-process documents before chunking for RAG?",
    "How do I combine RAG with conversation history so follow-up questions work?",
    "What is the difference between dense and sparse retrieval in RAG systems?",
    "How do I build a RAG evaluation dataset to benchmark different chunking strategies?",
    # -- FAISS / Vector DBs (15) --
    "What is the difference between FAISS and Pinecone and which should I use?",
    "How do I save a FAISS index to disk and reload it without rebuilding embeddings?",
    "Getting index out of bounds error when searching FAISS - what does this mean?",
    "How do I add new documents to an existing FAISS index without rebuilding?",
    "What is the difference between IndexFlatL2 and IndexFlatIP in FAISS?",
    "How many vectors can FAISS hold before performance degrades significantly?",
    "How do I use FAISS with cosine similarity instead of L2 Euclidean distance?",
    "When should I use a cloud vector database like Pinecone vs local FAISS?",
    "How do I do batched similarity search in FAISS for multiple queries at once?",
    "My FAISS search results do not seem relevant - could the embedding model be the problem?",
    "How do I store and retrieve metadata alongside vectors in FAISS?",
    "How do I install faiss-cpu on Windows without getting a compilation error?",
    "What does the distance score from FAISS search actually represent?",
    "How do I choose between IVF and flat indexes in FAISS for a 500k vector dataset?",
    "How do I normalize embeddings before adding to FAISS to get cosine similarity scores?",
    # -- Python Debugging (30) --
    "Getting ImportError cannot import name X from Y even though the class exists in that file",
    "My Python script works locally but crashes on the server with a missing module error",
    "What does TypeError unsupported operand type str and int mean and how do I find it?",
    "How do I use pdb to step through a Python script and inspect variables at runtime?",
    "My async function is blocking the event loop - how do I find the culprit?",
    "Getting RecursionError maximum recursion depth exceeded - how do I debug this?",
    "How do I safely handle None values in Python to avoid AttributeError?",
    "Getting KeyError when accessing a dictionary - how do I handle missing keys gracefully?",
    "How do I read a Python traceback properly to find where the actual error originated?",
    "My for loop is modifying the list it is iterating over and skipping elements - is that the bug?",
    "Getting UnicodeDecodeError when reading a CSV file - how do I fix the encoding?",
    "How do I profile Python code to identify which function is causing slowness?",
    "Getting MemoryError when processing a large dataset - how do I handle without loading all?",
    "How do I catch multiple exception types in a single try except block in Python?",
    "My Python script silently crashes after a few minutes - how do I debug?",
    "Getting OSError Address already in use when starting my server - port conflict?",
    "How do I fix cannot unpack non-iterable NoneType object error in Python?",
    "My function is returning None instead of the expected value - how do I trace this?",
    "How do I use Python logging module properly instead of scattered print statements?",
    "Getting circular import error in my Python project - how do I fix it?",
    "How do I process a 5GB JSON file in Python without running out of memory?",
    "My Python virtual environment is broken - how do I safely recreate it?",
    "Getting PermissionError when writing a file - how do I fix permissions?",
    "How do I fix AttributeError datetime object has no attribute utcnow in Python 3.12?",
    "My Python script hangs indefinitely with no output - how do I find where it is stuck?",
    "Getting SSL certificate verify failed when making requests to an external API",
    "How do I fix RuntimeError Event loop is closed in an async Python application?",
    "Getting TypeError object of type coroutine has no len - forgot to await somewhere?",
    "How do I debug a memory leak in a long-running Python process?",
    "How do I safely delete items from a dictionary while iterating over it?",
    # -- Docker / Deployment (15) --
    "My Docker container starts but my app is not accessible on the expected port",
    "How do I pass environment variables securely to a Docker container in production?",
    "My Docker image is 3GB - how do I reduce image size with a multi-stage build?",
    "Getting permission denied error inside a Docker container when writing to a mounted volume",
    "How do I set up Docker Compose for a FastAPI app plus a PostgreSQL database?",
    "My app works fine locally but fails after Docker build with a Python version mismatch",
    "How do I use .dockerignore correctly to exclude node_modules and .env from the build?",
    "How do I deploy my Python FastAPI app to Render as a web service?",
    "My Render deployment keeps failing during the pip install step with a dependency conflict",
    "How do I set environment variables on Render so my app can read them at runtime?",
    "What is the difference between a Render web service and a background worker?",
    "How do I view real-time logs for my deployed app on Render to debug a crash?",
    "How do I set up a GitHub Actions workflow to auto-deploy to Render on push to main?",
    "How do I add a /health endpoint so Render health checks pass and my service stays up?",
    "Getting exec format error when running my Docker container on a different machine",
    # -- LangChain (15) --
    "What is the practical difference between using LangChain vs calling the OpenAI API directly?",
    "How do I create a simple retrieval QA chain in LangChain for document question answering?",
    "Getting deprecation warnings throughout my LangChain code - how do I migrate to LCEL?",
    "How do I use ConversationBufferMemory in LangChain to maintain conversation history?",
    "How do I load and split a PDF document using LangChain document loaders?",
    "My LangChain RetrievalQA chain returns wrong answers even when relevant chunks are there",
    "How do I add a custom system prompt to my LangChain RAG chain?",
    "What is LCEL and should I rewrite my old LangChain chains to use it?",
    "How do I use LangChain callbacks to log every LLM call with its input and output?",
    "How do I switch my LangChain chain from OpenAI to a locally running Ollama model?",
    "How do I create a LangChain agent that can use multiple tools?",
    "Getting OutputParserException in LangChain - the LLM is not returning the expected format",
    "How do I use LangChain with Supabase pgvector as the vector store instead of FAISS?",
    "How do I handle context window overflow in LangChain when documents are too long?",
    "How do I build a simple ReAct agent in LangChain that can search the web?",
    # -- Embeddings (12) --
    "Which embedding model should I use - OpenAI text-embedding-3-small or a HuggingFace model?",
    "Why do I need to embed text at all before doing similarity search?",
    "My embedding similarity scores are all very high regardless of relevance - what is wrong?",
    "How do I batch embed thousands of text chunks efficiently without hitting rate limits?",
    "How do I compare two sentences for semantic similarity and set a meaningful threshold?",
    "What is the right way to normalize embeddings before cosine similarity comparison?",
    "How do I check if two document chunks are semantically duplicate using embeddings?",
    "How do I store text embeddings in PostgreSQL using the pgvector extension?",
    "What is the difference between symmetric and asymmetric embedding models for retrieval?",
    "How do I run HuggingFace sentence-transformers locally without internet access?",
    "How do I reduce embedding storage size without losing too much retrieval quality?",
    "How do I add new embeddings to an existing pgvector table efficiently?",
    # -- Prompt Engineering (10) --
    "How do I write a prompt that reliably gets JSON output from the LLM without extra text?",
    "My LLM keeps ignoring the system prompt constraints - how do I enforce them more strictly?",
    "What is chain-of-thought prompting and when does it actually improve accuracy?",
    "How do I write effective few-shot examples in my prompt to guide the model output format?",
    "My prompt works with GPT-4o but gives poor results with GPT-4o-mini - how do I adapt it?",
    "How do I prevent the LLM from making up information it does not know?",
    "How do I structure a classification prompt so the model only outputs the label?",
    "How do I use XML or markdown delimiters in prompts to separate instructions from content?",
    "My LLM responses vary a lot between runs - how do I make outputs more consistent?",
    "How do I implement a simple guardrail that refuses off-topic questions in my chatbot?",
    # -- General AI (13) --
    "What is the difference between fine-tuning a model and using prompt engineering?",
    "How does the attention mechanism in transformers work at a conceptual level?",
    "What is a context window limit and how does it affect my application design?",
    "How do I evaluate the quality of my AI application responses in an automated way?",
    "What is hallucination in LLMs and what are the main strategies to reduce it?",
    "When should I use an AI agent vs a simple prompt chain vs a single LLM call?",
    "How do I implement token streaming in my web app for a better user experience?",
    "What is the practical difference between temperature and top_p sampling parameters?",
    "How do I build an AI application that stays within a fixed monthly API cost budget?",
    "What is the difference between semantic search and BM25 keyword search?",
    "How do I add persistent conversation memory to my AI chatbot across sessions?",
    "What is grounding in AI systems and how does RAG provide factual grounding?",
    "How do I handle multi-language queries in my AI application?",
    # -- Streamlit (10) --
    "How do I build a simple streaming chat interface using Streamlit and the OpenAI API?",
    "How do I deploy my Streamlit app publicly so others can access it without running locally?",
    "Getting StreamlitAPIException when I use st.session_state - what am I doing wrong?",
    "How do I add simple password authentication to my Streamlit demo app?",
    "How do I stream LLM token responses in Streamlit so the user sees text appear in real time?",
    "What is the difference between st.cache_resource and st.cache_data in Streamlit?",
    "My Streamlit app re-runs the entire script on every button click and resets state - how to fix?",
    "How do I add file upload and parse the uploaded CSV in Streamlit?",
    "How do I display a pandas DataFrame with custom formatting in Streamlit?",
    "How do I customize the layout and colors of my Streamlit app?",
    # -- Auth / Security (10) --
    "How do I implement JWT token authentication in FastAPI with refresh token support?",
    "How do I store API keys and secrets securely so they are not hardcoded in my code?",
    "I accidentally committed my .env file with API keys to a public GitHub repo - what do I do?",
    "How do I add proper rate limiting to prevent my API from being abused?",
    "How do I implement OAuth2 login with Google in a FastAPI backend?",
    "What is the correct way to hash and verify passwords in Python using bcrypt?",
    "How do I validate that an incoming webhook request is genuinely from the expected service?",
    "How do I implement role-based access control so only admins can hit certain endpoints?",
    "How do I prevent prompt injection attacks in my AI application that processes user input?",
    "How do I audit log every user action in my application for compliance and debugging?",
    # -- Supabase / Database (15) --
    "How do I query Supabase from Python and handle the 1000 row default limit correctly?",
    "How do I insert a batch of 500 rows into Supabase without hitting request size limits?",
    "How do I filter Supabase rows by a nested JSON field inside a metadata column?",
    "Getting JWT expired error from Supabase - how do I refresh the service role key?",
    "How do I use Supabase Row Level Security to restrict data access per user?",
    "How do I run raw SQL queries against my Supabase PostgreSQL database from Python?",
    "How do I set up Supabase realtime subscriptions to listen for new row inserts?",
    "What is the difference between the Supabase anon key and service role key?",
    "How do I paginate through all rows in a Supabase table when there are more than 1000?",
    "Getting duplicate key violates unique constraint error on Supabase upsert - how to handle?",
    "How do I enable pgvector extension in Supabase for storing embeddings?",
    "Getting permission denied for table in Supabase - RLS is blocking my query?",
    "How do I run a database migration on Supabase without downtime?",
    "How do I use Supabase realtime to push updates to the browser without polling?",
    "My Supabase query is very slow on a 100k row table - how do I add an index?",
]


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

HOUR_WEIGHTS = [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 8, 7, 8, 9, 10, 9, 8, 7, 6, 5, 3, 2]


def random_timestamp(rng: random.Random) -> datetime:
    day_offset = rng.randint(0, WINDOW_SPAN_DAYS - 1)
    hour       = rng.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
    minute     = rng.randint(0, 59)
    second     = rng.randint(0, 59)
    ts = WINDOW_START + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)
    return min(ts, WINDOW_END)


# ---------------------------------------------------------------------------
# Step 1: Delete ALL events in the 30-day window
# ---------------------------------------------------------------------------

def delete_window_data():
    PAGE = 1000
    deleted = 0
    while True:
        # Fetch IDs in window
        res = (client.table('analytics_events')
               .select('id')
               .gte('created_at', WINDOW_START.isoformat())
               .lte('created_at', WINDOW_END.isoformat())
               .limit(PAGE)
               .execute())
        if not res.data:
            break
        ids = [r['id'] for r in res.data]
        client.table('analytics_events').delete().in_('id', ids).execute()
        deleted += len(ids)
        print(f'[*] Deleted {deleted} rows so far...')
        if len(ids) < PAGE:
            break
    print(f'[OK] Cleared {deleted} rows from window ({WINDOW_START.date()} to {WINDOW_END.date()})')


# ---------------------------------------------------------------------------
# Step 2: Build exactly 1047 records with correct event breakdown
# ---------------------------------------------------------------------------

def build_records() -> list[dict]:
    rng = random.Random(20260308)

    # Cycle query pool to reach TOTAL_QUERIES
    pool = (QUERIES_POOL * (TOTAL_QUERIES // len(QUERIES_POOL) + 1))[:TOTAL_QUERIES]
    rng.shuffle(pool)

    indices = list(range(TOTAL_QUERIES))
    rng.shuffle(indices)

    tag_idx  = set(indices[:N_TAG_CREW])
    got_idx  = set(indices[N_TAG_CREW : N_TAG_CREW + N_GOT_IT])
    cont_idx = set(indices[N_TAG_CREW + N_GOT_IT : N_TAG_CREW + N_GOT_IT + N_CONTINUE])
    # remainder = no engagement

    tag = {'is_demo5': True}
    records = []

    for i, query_text in enumerate(pool):
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

        if i in tag_idx:
            t1 = ts + timedelta(minutes=rng.randint(2, 7), seconds=rng.randint(0, 59))
            t2 = t1 + timedelta(minutes=rng.randint(1, 4), seconds=rng.randint(0, 59))
            records.append({'created_at': t1.isoformat(), 'event_type': 'need_help',
                            'thread_id': thread_id, 'user_id': user_id,
                            'message_id': None, 'metadata': tag})
            records.append({'created_at': t2.isoformat(), 'event_type': 'tag_crew',
                            'thread_id': thread_id, 'user_id': user_id,
                            'message_id': None, 'metadata': tag})
        elif i in got_idx:
            t1 = ts + timedelta(minutes=rng.randint(1, 5), seconds=rng.randint(0, 59))
            records.append({'created_at': t1.isoformat(), 'event_type': 'got_it',
                            'thread_id': thread_id, 'user_id': user_id,
                            'message_id': None, 'metadata': tag})
        elif i in cont_idx:
            t1 = ts + timedelta(minutes=rng.randint(2, 6), seconds=rng.randint(0, 59))
            t2 = t1 + timedelta(minutes=rng.randint(1, 3), seconds=rng.randint(0, 59))
            records.append({'created_at': t1.isoformat(), 'event_type': 'need_help',
                            'thread_id': thread_id, 'user_id': user_id,
                            'message_id': None, 'metadata': tag})
            records.append({'created_at': t2.isoformat(), 'event_type': 'continue_here',
                            'thread_id': thread_id, 'user_id': user_id,
                            'message_id': None, 'metadata': tag})

    return records


# ---------------------------------------------------------------------------
# Step 3: Insert
# ---------------------------------------------------------------------------

def insert_batches(records: list[dict], batch_size: int = 100):
    total, inserted = len(records), 0
    for i in range(0, total, batch_size):
        client.table('analytics_events').insert(records[i:i + batch_size]).execute()
        inserted += len(records[i:i + batch_size])
        print(f'[*] Inserted {inserted}/{total}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 55)
    print('Step 1: Clear all data in window (Feb 5 - Mar 8)')
    print('=' * 55)
    delete_window_data()

    print()
    print('=' * 55)
    print('Step 2: Build 1047 demo records')
    print('=' * 55)
    records = build_records()

    by_type = {}
    for r in records:
        by_type[r['event_type']] = by_type.get(r['event_type'], 0) + 1

    q    = by_type.get('query', 0)
    g    = by_type.get('got_it', 0)
    nh   = by_type.get('need_help', 0)
    tc   = by_type.get('tag_crew', 0)
    cont = by_type.get('continue_here', 0)

    print(f'[OK] Records built: {len(records)} total events')
    print(f'     queries      : {q}')
    print(f'     got_it       : {g}')
    print(f'     need_help    : {nh}')
    print(f'     tag_crew     : {tc}')
    print(f'     continue_here: {cont}')
    print(f'     engagement   : {(g+nh)/q*100:.1f}%  (target 67%)')
    print(f'     escalation   : {tc/q*100:.1f}%  (target 12%)')

    print()
    print('=' * 55)
    print('Step 3: Insert into Supabase')
    print('=' * 55)
    insert_batches(records)

    print()
    print('[OK] Done.')
    print(f'     Total queries in window: {q}')
    print(f'     Engagement             : {(g+nh)/q*100:.1f}%')
    print(f'     Escalation             : {tc/q*100:.1f}%')
    print()
    print('Cleanup SQL (after event):')
    print("  DELETE FROM analytics_events WHERE metadata->>'is_demo5' = 'true';")


if __name__ == '__main__':
    main()
