# Webinar Demo — Changes & Revert Guide

## What was done for the webinar

Three things were added on top of production, all tagged for clean removal.

---

## 1. Demo data (CSV export re-inserted with shifted timestamps)

**Script**: `scripts/import_demo_data.py`

All 396 rows from `analytics_events_rows.csv` (the production CSV export) were
re-inserted into the live Supabase DB with:
- Timestamps shifted proportionally to fit within the last 30 days
- `metadata.is_demo = true` on every row

**What it shows on dashboard**: real production interaction patterns
(got_it, tag_crew, need_help, etc.) visible in the 30-day chart.

---

## 2. Synthetic RAG queries (1111 queries run through Sage's pipeline)

**Scripts**:
- `scripts/generate_synthetic_queries.py` — generated 1111 queries via OpenAI
- `scripts/run_rag_pipeline.py` — batch-embedded + FAISS-searched all queries, inserted into Supabase

All 1111 rows inserted with:
- `event_type = "query"` with `metadata.query_text` populated
- `metadata.matched_lectures` = top-3 RAG hits per query
- `metadata.is_synthetic = true` on every row
- Timestamps spread across last 30 days

**What it shows on dashboard**:
- Main page: 1485+ total queries, full 30-day chart coverage
- Common Queries page: topics grouped by matched lecture (e.g. "Full Stack LLM — Building APIs with FastAPI")

---

## 3. Dashboard code changes

**File**: `dashboard/src/app/api/cluster-topics/route.ts`
- Rewrote clustering to use `matched_lectures` metadata directly (no LLM for synthetic queries)
- LLM clustering now only runs on old queries without `matched_lectures`, capped at 100

**File**: `dashboard/src/app/page.tsx`
- Added pagination (1000-row chunks) to bypass Supabase's default 1000-row limit

Both changes are production improvements — no need to revert them.

---

## How to revert after webinar

### Step 1 — Delete demo + synthetic data from Supabase

Run these two SQL statements in the Supabase dashboard
(SQL Editor at app.supabase.com > your project > SQL Editor):

```sql
DELETE FROM analytics_events WHERE metadata->>'is_demo' = 'true';
DELETE FROM analytics_events WHERE metadata->>'is_synthetic' = 'true';
```

This removes all 1507 inserted rows and leaves only real production data intact.

### Step 2 — Verify real data is back

```sql
SELECT event_type, COUNT(*) FROM analytics_events GROUP BY event_type;
```

You should see the original ~206 query events and associated feedback events only.

### Step 3 — Keep the dashboard code changes

The pagination fix (`page.tsx`) and the two-path clustering (`cluster-topics/route.ts`)
are genuine improvements. Keep them — they make the dashboard more correct for real data too.

### Step 4 — Switch dashboard back to main branch on Vercel (optional)

If you want to revert the dashboard to the pre-webinar Vercel deployment,
go to Vercel > project > Deployments and promote the previous production deployment.
Otherwise the current code is fine to keep.

---

## 4. March 2026 event — controlled demo data (1047 queries)

**Script**: `scripts/reset_demo_stats.py`

Clears the Feb 5 – Mar 8 window and inserts exactly 1047 realistic demo queries with:
- `metadata.is_demo5 = true` on every row
- Timestamps spread across Feb 5 – Mar 8, 2026
- Engagement rate: **67%** `(got_it + need_help) / queries`
- Mentor escalation rate: **12%** `tag_crew / queries`

Event breakdown:
| Event | Count |
|---|---|
| query | 1047 |
| got_it | 470 |
| need_help | 231 |
| tag_crew | 126 |
| continue_here | 105 |

**What it shows on dashboard**: 1047 total queries, 67% engagement, 12% escalation across the 30-day view.

**Revert after event**:
```sql
DELETE FROM analytics_events WHERE metadata->>'is_demo5' = 'true';
```

Note: this script also deletes all previous demo tags (`is_demo`, `is_synthetic`, `is_demo2`, `is_demo3`, `is_demo4`) before inserting fresh data.

---

## Git branches

| Branch | Purpose |
|--------|---------|
| `main` | Production bot code |
| `dev/synthetic-data-pipeline` | All webinar scripts + dashboard fixes |

After webinar, you can either merge `dev/synthetic-data-pipeline` into `main`
(recommended — the dashboard fixes are real improvements) or leave it as-is.

---

## Files added in this branch

| File | Purpose | Keep after webinar? |
|------|---------|-------------------|
| `scripts/import_demo_data.py` | Inserts CSV data into Supabase | Optional |
| `scripts/generate_synthetic_queries.py` | Generates 1111 synthetic queries | Yes — useful for future RAG testing |
| `scripts/run_rag_pipeline.py` | Runs queries through RAG + inserts to Supabase | Yes — useful for future RAG testing |
| `scripts/synthetic_queries.json` | The 1111 generated queries | Yes — reusable test dataset |
| `scripts/import_march_demo.py` | March 2026 event — 205 queries (is_demo2) | Optional |
| `scripts/import_feb_demo.py` | Feb 5-24 batch — 100 queries (is_demo3) | Optional |
| `scripts/fix_demo_stats.py` | Intermediate fix script (superseded by reset) | Optional |
| `scripts/reset_demo_stats.py` | Final reset — 1047 queries with exact 67%/12% stats (is_demo5) | Yes — rerun for future events |
| `dashboard/src/app/page.tsx` | Pagination fix | Yes |
| `dashboard/src/app/api/cluster-topics/route.ts` | Two-path clustering fix | Yes |

---

## Quick reference — Supabase SQL to check data

```sql
-- Count by source
SELECT
  metadata->>'is_demo' as is_demo,
  metadata->>'is_synthetic' as is_synthetic,
  COUNT(*) as count
FROM analytics_events
GROUP BY is_demo, is_synthetic;

-- Verify cleanup worked
SELECT COUNT(*) FROM analytics_events
WHERE metadata->>'is_demo' = 'true'
   OR metadata->>'is_synthetic' = 'true';
-- Should return 0 after cleanup

-- March 2026 event cleanup
DELETE FROM analytics_events WHERE metadata->>'is_demo5' = 'true';

-- Verify March event cleanup
SELECT COUNT(*) FROM analytics_events
WHERE metadata->>'is_demo5' = 'true';
-- Should return 0 after cleanup
```
