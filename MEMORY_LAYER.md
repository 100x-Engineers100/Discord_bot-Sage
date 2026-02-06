# Memory Layer — Progress

**Status:** DONE — all 3 stages shipped, build passes, 12/12 tests pass.

---

## What it does
Persists RAG match data (lecture + module + score) on every Sage query event into Supabase `metadata` column. Surfaces it on a new "Common Queries" dashboard page so mentors can see which topics students struggle with most.

Zero new tables. Zero extra API calls. Zero new dependencies.

---

## Stage 1 — bot.py: persist query_text + matched lectures [DONE]

### Bug fix
- `bot.py` line 148: table name `'analytics'` --> `'analytics_events'` (was writing to wrong table)

### Changes
- `log_event()` — added optional `extra_metadata: dict` param. Merges into metadata before insert. All 4 existing callers (got_it, need_help, continue_here, tag_crew) pass nothing — zero change to them.
- Sage `on_message` query call site — builds `query_meta` with:
  - `query_text`: the raw student query
  - `matched_lectures`: top-3 RAG hits filtered by L2 score < 1.5. Each entry has lecture_num, lecture_name, module, module_name, rag_score (rounded 4dp).

### Test: `test_memory_layer.py`
12 unit tests. Mocks supabase — no network. Covers:
- Existing callers unchanged (no extra_metadata)
- 3 good hits / mixed scores / all poor scores
- Empty + None rag_results
- Boundary score == 1.5 (excluded)
- Score rounding to 4dp
- Missing score key defaults to 999 (excluded)
- Full end-to-end payload

Run: `python -m unittest test_memory_layer -v`

---

## Stage 2 — Dashboard: shared layout + nav [DONE]

### Files
- **CREATED:** `dashboard/src/components/DashboardLayout.tsx`
  - Client component. Renders Scene (3D neon bg), shared header, nav tabs.
  - Nav: "Overview" (`/`) and "Common Queries" (`/common-queries`). Active state via `usePathname()`.
  - Pill style matches existing time-range selector (bg-black/40, border-green-500/30, active = bg-green-500/30).

- **MODIFIED:** `dashboard/src/app/page.tsx`
  - Stripped: outer wrapper div, Scene import + div, header block.
  - Added: `<DashboardLayout>` wrapping all content.
  - Moved: time range selector out of header into own row (page-specific state).
  - Kept: all state, fetch logic, metric cards, chart, secondary metrics — unchanged.

- **NO CHANGE:** `layout.tsx` (stays Server Component with metadata export)

---

## Stage 3 — Dashboard: Common Queries page [DONE]

### File
- **CREATED:** `dashboard/src/app/common-queries/page.tsx`

### Data flow (all client-side)
1. Fetch events where event_type IN (query, tag_crew), filtered by time range
2. `escalatedThreads` Set built from tag_crew thread_ids
3. Each query's `metadata.matched_lectures` iterated — each lecture match increments that lecture's count independently (one query can hit multiple lectures)
4. Per lecture: query_count, escalation_count (thread in escalatedThreads), deduplicated query_texts
5. Trending: time range split in half. If second_half > 0 and (first_half == 0 OR second/first > 1.5) --> trending
6. Two sorted lists: escalatedTopics (desc by escalation_count), allTopics (desc by query_count)

### UI (top to bottom)
- Time range selector (own state, same pill style)
- "Topics with Escalations" section — amber AlertTriangle icon, expandable TopicCards. Only renders if any exist.
- Horizontal bar chart — top 10 topics. Recharts BarChart layout="vertical". Gradient fill opacity decreases per bar. Trending topics get brighter green bar.
- "All Common Topics" section — all topics sorted by frequency. Expandable TopicCards.

### TopicCard component
- Header: lecture key, module badge (green), trending badge (blue + TrendingUp icon), escalation badge (amber), query count (green)
- Chevron toggle
- Expanded: query_text strings in dark pill cards
- Fallback: "No query text recorded" if pre-memory-layer data

### Build
`npm run build` in dashboard dir — compiles with zero errors. Routes: `/` and `/common-queries`.

---

## Files changed/created

| Action   | Path                                                  |
|----------|-------------------------------------------------------|
| MODIFY   | `bot.py`                                              |
| CREATE   | `test_memory_layer.py`                                |
| CREATE   | `dashboard/src/components/DashboardLayout.tsx`        |
| MODIFY   | `dashboard/src/app/page.tsx`                          |
| CREATE   | `dashboard/src/app/common-queries/page.tsx`           |

---

## How to run

**Bot (Python) — Terminal 1:**
```
cd discord-support-bot
python bot.py
```

**Dashboard (Next.js) — Terminal 2:**
```
cd discord-support-bot\dashboard
npm run dev
```
Open `http://localhost:3000`. Nav tabs: Overview + Common Queries.

---

## Current state of Common Queries data
- All existing Supabase events are pre-memory-layer (no `matched_lectures` in metadata).
- They will show as "Unknown Topic" until new queries come in via the updated bot.
- Once bot is running with new `bot.py`, fresh queries populate lecture/module/trending data automatically.
