"""
Import demo data from CSV into Supabase analytics_events table.
- Shifts all timestamps to fit within last 30 days (proportionally)
- Tags each row with metadata: {"is_demo": true}
- To remove after demo: DELETE FROM analytics_events WHERE metadata->>'is_demo' = 'true'
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'analytics_events_rows.csv')

if not SUPABASE_URL or not SUPABASE_KEY:
    print('[ERROR] Missing SUPABASE_URL or SUPABASE_KEY in .env')
    sys.exit(1)

client = create_client(SUPABASE_URL, SUPABASE_KEY)


def parse_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f'[OK] Loaded {len(rows)} rows from CSV')
    return rows


def shift_timestamps(rows):
    """
    Proportionally shift all timestamps so they span the last 30 days,
    preserving relative ordering and spacing.
    """
    now = datetime.now(timezone.utc)
    window_end = now
    window_start = now - timedelta(days=30)

    times = []
    for row in rows:
        ts_str = row['created_at'].strip()
        # handle both with and without timezone
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
        times.append(ts)

    orig_min = min(times)
    orig_max = max(times)
    orig_span = (orig_max - orig_min).total_seconds()
    target_span = (window_end - window_start).total_seconds()

    shifted = []
    for ts in times:
        if orig_span == 0:
            new_ts = window_start
        else:
            ratio = (ts - orig_min).total_seconds() / orig_span
            new_ts = window_start + timedelta(seconds=ratio * target_span)
        shifted.append(new_ts.isoformat())

    return shifted


def build_records(rows, shifted_timestamps):
    records = []
    for row, new_ts in zip(rows, shifted_timestamps):
        # parse existing metadata
        meta_raw = row.get('metadata', '{}') or '{}'
        try:
            meta = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            meta = {}

        meta['is_demo'] = True

        record = {
            'created_at': new_ts,
            'event_type': row['event_type'].strip(),
            'thread_id': str(row['thread_id']).strip(),
            'user_id': str(row['user_id']).strip() if row.get('user_id') else None,
            'message_id': str(row['message_id']).strip() if row.get('message_id') else None,
            'metadata': meta,
        }
        records.append(record)
    return records


def insert_in_batches(records, batch_size=50):
    total = len(records)
    inserted = 0
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        result = client.table('analytics_events').insert(batch).execute()
        inserted += len(batch)
        print(f'[*] Inserted {inserted}/{total}')
    print(f'[OK] All {total} records inserted successfully')


def main():
    print('[*] Starting demo data import...')
    rows = parse_csv(CSV_PATH)
    shifted = shift_timestamps(rows)
    records = build_records(rows, shifted)
    insert_in_batches(records)
    print()
    print('[OK] Done. To remove demo data after the demo, run:')
    print("     DELETE FROM analytics_events WHERE metadata->>'is_demo' = 'true'")


if __name__ == '__main__':
    main()
