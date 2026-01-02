

# Supabase Database Migrations

Run these SQL scripts in order in your Supabase SQL Editor.

## Setup Instructions

### 1. Create Supabase Project
1. Go to https://supabase.com
2. Sign up/login
3. Click "New Project"
4. Name: `sage-analytics`
5. Save database password
6. Choose region closest to Render
7. Select Free plan
8. Wait ~2 min for provisioning

### 2. Run Migrations

In Supabase Dashboard → SQL Editor → New Query:

**Step 1:** Run `01_create_analytics_tables.sql`
- Creates `analytics_events` table
- Sets up indexes

**Step 2:** Run `02_setup_rls_policies.sql`
- Enables Row Level Security
- Configures access policies

### 3. Get Connection Details

After running migrations, get your credentials:

1. Go to Project Settings → API
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **Project API Key (service_role)**: `eyJhbGc...` (secret key)

### 4. Add to Bot Environment

Add to `.env` file:
```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGc... # service_role key (NOT anon key)
```

## Table Schema

### analytics_events
Tracks all bot interactions and button clicks.

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Auto-increment primary key |
| created_at | TIMESTAMPTZ | Event timestamp (auto) |
| event_type | VARCHAR(50) | Event type (see below) |
| thread_id | BIGINT | Discord thread ID |
| user_id | BIGINT | Discord user ID |
| message_id | BIGINT | Discord message ID (optional) |
| metadata | JSONB | Additional context |

### Event Types
- `query` - Bot answered a question
- `got_it` - User clicked "Got it, thanks"
- `tag_crew` - User clicked "Tag the crew"
- `need_help` - User clicked "Need more help"
- `continue_here` - User clicked "Continue here"

## Security

- **RLS Enabled**: Row Level Security protects data
- **Service Role**: Bot writes with service_role key
- **Authenticated**: Dashboard reads with authenticated users
- **No Public Write**: Only bot can insert data

## Testing

After setup, test with SQL query:
```sql
SELECT event_type, COUNT(*)
FROM analytics_events
GROUP BY event_type;
```

Should return empty result (no data yet).
