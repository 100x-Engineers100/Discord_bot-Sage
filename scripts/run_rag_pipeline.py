"""
Run 1111 synthetic queries through Sage's RAG pipeline and insert into Supabase.
- Batch embeds all queries in 2 API calls (not 1111)
- FAISS search is local/free
- Inserts as analytics_events with query_text in metadata
- Tagged is_synthetic=true for cleanup later

Cleanup SQL:
  DELETE FROM analytics_events WHERE metadata->>'is_synthetic' = 'true';
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
import random

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

# Add parent dir to path so we can import config + curriculum_rag
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
import faiss

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
QUERIES_PATH = os.path.join(os.path.dirname(__file__), 'synthetic_queries.json')

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Load FAISS index + metadata
# ---------------------------------------------------------------------------

def load_rag_index():
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f'[OK] Loaded FAISS index: {index.ntotal} vectors')
    return index, metadata


# ---------------------------------------------------------------------------
# Batch embed all queries in one go
# ---------------------------------------------------------------------------

def batch_embed(queries: list[str]) -> np.ndarray:
    """Embed all queries using OpenAI in batches of 500 (well under 2048 limit)"""
    BATCH_SIZE = 500
    all_embeddings = []

    for i in range(0, len(queries), BATCH_SIZE):
        batch = queries[i:i + BATCH_SIZE]
        print(f'[*] Embedding batch {i // BATCH_SIZE + 1}/{(len(queries) - 1) // BATCH_SIZE + 1} ({len(batch)} queries)...')
        response = openai_client.embeddings.create(
            input=batch,
            model=config.EMBEDDING_MODEL
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f'[OK] Embedded {len(embeddings)} queries. Shape: {embeddings.shape}')
    return embeddings


# ---------------------------------------------------------------------------
# FAISS search for all queries (local, free)
# ---------------------------------------------------------------------------

def search_all(index, metadata, embeddings: np.ndarray, top_k: int = 3) -> list[list[dict]]:
    distances, indices = index.search(embeddings, top_k)

    results = []
    for i in range(len(embeddings)):
        matches = []
        for idx, dist in zip(indices[i], distances[i]):
            if idx < 0 or idx >= len(metadata):
                continue
            lec = metadata[idx]
            matches.append({
                'module': lec.get('module'),
                'module_name': lec.get('module_name', ''),
                'lecture_num': lec.get('lecture_num'),
                'lecture_name': lec.get('lecture_name', ''),
                'rag_score': round(float(dist), 4),
            })
        results.append(matches)
    return results


# ---------------------------------------------------------------------------
# Spread timestamps across last 30 days with realistic daily patterns
# ---------------------------------------------------------------------------

def make_timestamp(index: int, total: int) -> str:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=30)

    # Spread evenly + add small random jitter so chart looks natural
    ratio = index / total
    base = start + timedelta(seconds=ratio * 30 * 86400)
    jitter = timedelta(minutes=random.randint(-30, 30))
    ts = base + jitter

    # Clamp to window
    ts = max(ts, start)
    ts = min(ts, now)
    return ts.isoformat()


# ---------------------------------------------------------------------------
# Insert into Supabase in batches
# ---------------------------------------------------------------------------

def insert_batches(records: list[dict], batch_size: int = 100):
    total = len(records)
    inserted = 0
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        supabase.table('analytics_events').insert(batch).execute()
        inserted += len(batch)
        print(f'[*] Inserted {inserted}/{total}')
    print(f'[OK] All {total} records inserted')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('[*] Loading synthetic queries...')
    with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
        synthetic = json.load(f)
    print(f'[OK] {len(synthetic)} queries loaded')

    index, metadata = load_rag_index()

    queries_text = [r['query'] for r in synthetic]

    # Batch embed (2-3 API calls total)
    embeddings = batch_embed(queries_text)

    # FAISS search (local, instant)
    print('[*] Running FAISS search on all queries...')
    all_matches = search_all(index, metadata, embeddings, top_k=3)
    print(f'[OK] Search complete')

    # Build analytics_events records
    records = []
    total = len(synthetic)
    for i, (q_record, matches) in enumerate(zip(synthetic, all_matches)):
        meta = {
            'query_text': q_record['query'],
            'topic': q_record['topic'],
            'module': q_record['module'],
            'type': q_record['type'],
            'matched_lectures': matches,
            'is_synthetic': True,
        }
        if matches:
            meta['rag_score'] = matches[0]['rag_score']

        records.append({
            'created_at': make_timestamp(i, total),
            'event_type': 'query',
            'thread_id': str(1400000000000000000 + i),  # synthetic thread IDs
            'user_id': str(9000000000000000000 + (i % 50)),  # 50 synthetic users
            'message_id': None,
            'metadata': meta,
        })

    print(f'[*] Inserting {len(records)} records into Supabase...')
    insert_batches(records)

    # Print sample results
    print('\n--- Sample RAG results ---')
    for r in records[:5]:
        m = r['metadata']
        top = m['matched_lectures'][0] if m['matched_lectures'] else {}
        print(f"  Query: {m['query_text'][:60]}")
        print(f"  --> {top.get('module_name', '?')} / Lecture {top.get('lecture_num', '?')}: {top.get('lecture_name', '?')} (score: {top.get('rag_score', '?')})")
        print()

    print('[OK] Done. Both dashboard pages will now show synthetic query data.')
    print()
    print('Cleanup after demo:')
    print("  DELETE FROM analytics_events WHERE metadata->>'is_synthetic' = 'true';")


if __name__ == '__main__':
    main()
