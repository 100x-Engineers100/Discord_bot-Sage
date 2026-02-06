# Dual Bot System Deployment Guide

## Overview

Single Render service running TWO Discord bots in parallel:
- **@Sage**: Technical curriculum support (RAG + feedback buttons + analytics)
- **@Scout**: FAQ/program support (RAG + urgency detection)

**Cost savings**: $25/month (down from $50/month for 2 separate services)

---

## Architecture

```
Single Python Process (bot.py)
├── Sage Bot (@Sage)
│   ├── CurriculumRAG (Data_Doc_main.txt)
│   ├── Feedback buttons (FeedbackView, FollowUpView)
│   ├── Analytics (Supabase PostgreSQL)
│   └── Clarification loop prevention
│
├── Scout Bot (@Scout)
│   ├── ScoutRAG (FAQ_Doc.txt)
│   ├── Urgency detection (auto-tag mentors)
│   └── Simple FAQ responses
│
└── Shared Infrastructure
    ├── OpenAI client (text-embedding-3-small + GPT-4.1-mini)
    ├── FAISS indexes (2 separate: curriculum + FAQ)
    ├── Supabase client (analytics - Sage only)
    └── Message utilities (splitting, logging)
```

---

## File Structure

```
discord-support-bot/
├── bot.py                          # [NEW] Dual-bot orchestrator
├── curriculum_rag.py               # Sage RAG (technical)
├── scout_rag.py                    # [NEW] Scout RAG (FAQ)
├── config.py                       # [UPDATED] Both bot configs
├── .env                            # [UPDATED] Both bot tokens
│
├── Data_Doc_main.txt               # Sage data (curriculum)
├── FAQ_Doc.txt                     # [NEW] Scout data (FAQ)
│
├── embeddings/                     # Sage FAISS index
│   ├── curriculum.faiss
│   └── metadata.json
│
├── embeddings_faq/                 # [NEW] Scout FAISS index
│   ├── faq.faiss
│   └── faq_metadata.json
│
├── setup_embeddings.py             # Build Sage index
├── setup_scout_embeddings.py      # [NEW] Build Scout index
├── test_dual_bots.py               # [NEW] Pre-deployment tests
│
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment (unchanged)
└── Dockerfile                      # Docker config (unchanged)
```

---

## Pre-Deployment Checklist

### 1. Environment Variables (.env)

Ensure `.env` has both bot tokens:

```bash
# Sage Bot (Technical Support)
DISCORD_BOT_TOKEN=<sage_bot_token_here>

# Scout Bot (FAQ Support)
SCOUT_BOT_TOKEN=<scout_bot_token_here>

# OpenAI API
OPENAI_API_KEY=<openai_key_here>

# Supabase Analytics (Sage only)
SUPABASE_URL=<supabase_url>
SUPABASE_KEY=<supabase_key>
```

**IMPORTANT**: If you don't have a Scout bot token yet:
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create new application "Scout"
3. Enable bot with same permissions as Sage
4. Copy bot token to `.env`
5. Invite Scout bot to your Discord server

### 2. Build FAISS Indexes

**First time setup**:

```bash
# Build Sage curriculum index (if not exists)
python setup_embeddings.py

# Build Scout FAQ index
python setup_scout_embeddings.py
```

**Verify indexes**:
```bash
ls embeddings/          # Should show curriculum.faiss + metadata.json
ls embeddings_faq/      # Should show faq.faiss + faq_metadata.json
```

### 3. Run Tests

```bash
python test_dual_bots.py
```

Expected output:
```
[OK] Sage RAG test passed!
[OK] Scout RAG test passed!
[OK] All tests passed! Ready to deploy.
```

### 4. Local Testing (Optional)

Run both bots locally:

```bash
python bot.py
```

Expected startup logs:
```
[SAGE] Bot logged in as Sage (ID: ...)
[SHARED] OpenAI client initialized
[SHARED] Supabase client initialized
[SAGE] Loading curriculum RAG system...
[SAGE] RAG system ready!
[SAGE] Ready to help with technical queries!

[SCOUT] Bot logged in as Scout (ID: ...)
[SCOUT] Loading FAQ RAG system...
[SCOUT] RAG system ready!
[SCOUT] Ready to help with program questions!
```

---

## Deployment to Render

### Option 1: Push to Existing Render Service

If you already have Sage deployed on Render:

```bash
git add .
git commit -m "Merge Sage + Scout into dual-bot system"
git push origin main
```

Render will auto-deploy. Watch logs for both bots starting up.

### Option 2: New Render Service

1. Go to Render dashboard
2. Create new **Background Worker**
3. Connect GitHub repo
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python bot.py`
   - **Plan**: $25/month (Starter)

5. Add environment variables:
   - `DISCORD_BOT_TOKEN`
   - `SCOUT_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`

6. Deploy!

### Important: Render Deployment Notes

**FAISS indexes are NOT committed to git** (too large). You have two options:

#### Option A: Build indexes on Render (Recommended)

Add to Render build command:
```bash
pip install -r requirements.txt && python setup_embeddings.py --force && python setup_scout_embeddings.py --force
```

This rebuilds indexes on every deploy (~2 min extra build time, ~$0.02 OpenAI cost).

#### Option B: Persistent Disk (Advanced)

1. Render Dashboard → Service → Disks → Add Disk
2. Mount path: `/opt/render/project/src/embeddings`
3. Build indexes once, they persist across deploys

---

## Testing in Production

Once deployed, test both bots in Discord:

### Test @Sage (Technical Bot)

In a forum thread:
```
@Sage How does ControlNet work?
```

Expected:
- Response with curriculum context
- References Lecture 5
- Feedback buttons appear after 1.5s

```
@Sage What is RAG?
```

Expected:
- Response referencing Lecture 13-14
- Module 2 context
- Feedback buttons

### Test @Scout (FAQ Bot)

In a forum thread:
```
@Scout Where can I find session recordings?
```

Expected:
- Step-by-step instructions
- LMS navigation guide
- No feedback buttons (Scout doesn't use them)

```
@Scout How do I access Jarvis Labs?
```

Expected:
- Login steps
- Team code provided
- No feedback buttons

### Test Urgency Detection (Scout)

```
@Scout My Jarvis Labs crashed and I can't access it urgently!
```

Expected:
- FAQ answer
- **Auto-tags @mekashi @omkar** (urgency detected)

---

## Monitoring

### Check Logs

**Render Dashboard** → Your Service → Logs

Look for:
```
[SAGE] Bot logged in...
[SAGE] RAG system ready!
[SCOUT] Bot logged in...
[SCOUT] RAG system ready!
```

### Analytics Dashboard

Sage analytics still work! Visit your dashboard URL.

**Note**: Scout queries are NOT logged to analytics (only Sage uses Supabase).

To track Scout separately, you could add a second Supabase table or filter by bot type.

---

## Troubleshooting

### Both bots fail to start

**Error**: `SCOUT_BOT_TOKEN not found or not replaced`

**Fix**: Add Scout bot token to Render environment variables.

### Sage starts, Scout doesn't

**Error**: `Index not found at embeddings_faq/faq.faiss`

**Fix**: Run `python setup_scout_embeddings.py` locally and commit indexes, OR add to build command.

### One bot works, other doesn't respond

**Check**:
1. Both bots invited to Discord server?
2. Both bots have correct permissions?
3. Render logs show both bots logged in?

### Analytics not working

**Check**: Sage-only feature. Scout doesn't log analytics by design.

---

## Rollback Plan

If dual-bot system fails, rollback to single Sage bot:

```bash
# Restore original Sage bot
cp bot_sage_only_backup.py bot.py

# Redeploy
git add bot.py
git commit -m "Rollback to Sage-only"
git push origin main
```

---

## Cost Breakdown

**Before (2 bots, 2 services)**:
- Sage Render service: $25/month
- Scout Render service: $25/month
- **Total: $50/month**

**After (2 bots, 1 service)**:
- Dual-bot Render service: $25/month
- **Total: $25/month**

**Savings: $25/month = $300/year**

---

## Performance Notes

**Memory**: ~800MB (both FAISS indexes + OpenAI client loaded once)

**Startup time**: ~15-20 seconds (loads 2 RAG indexes)

**Scaling**: Render $25 plan handles 200+ queries/month easily (current load: 206/month).

**Failure mode**: If one bot crashes, both go down (shared process). Trade-off for cost savings.

---

## Next Steps

After successful deployment:

1. **Pin a message** in Discord forum:
   ```
   🤖 BOT GUIDE:
   - @Sage: Technical questions (code, errors, curriculum concepts)
   - @Scout: Program questions (LMS, recordings, sessions, Jarvis Labs)
   ```

2. **Update CLAUDE.md** with dual-bot architecture

3. **Monitor analytics** to see usage split between Sage and Scout

4. **Optional**: Add Scout analytics if you want separate tracking

---

## Support

Issues? Check:
- [bot.py:890] - Error logs
- Render dashboard logs
- Test locally with `python bot.py`
- Run `python test_dual_bots.py`

Contact: Your friendly AI assistant! 🤖
