\# Sage Bot - Project Memory



\## Project Overview



\*\*Sage\*\* is a RAG-powered Discord support bot built for \*\*100xEngineers AI Cohort 6\*\* students. It provides technical Q\&A assistance in Discord forum threads using curriculum-grounded responses.



\### Core Purpose

\- Answer student technical questions using RAG (Retrieval-Augmented Generation)

\- Maintain conversation context per thread

\- Provide intelligent feedback system with escalation to mentors

\- Help students solve problems efficiently while tracking satisfaction



\### Production Performance (Last 30 Days)

\- 📊 \*\*206 total queries\*\* answered

\- ✅ \*\*92.2% self-service success\*\* (190/206 queries resolved without mentor help)

\- 🎯 \*\*51.5% explicit satisfaction rate\*\* (17 "Got it, thanks!" vs 16 escalations)

\- 📉 \*\*7.8% escalation rate\*\* (only 16 mentor tags needed)

\- 💬 \*\*29.1% engagement rate\*\* (60 button interactions out of 206 queries)

\- 🚀 \*\*Dashboard live\*\* with real-time analytics \& trend tracking



---



\## Current Architecture



\### Tech Stack

\- \*\*Language\*\*: Python 3.12

\- \*\*Discord Framework\*\*: discord.py 2.6.4

\- \*\*LLM\*\*: OpenAI GPT-4.1-mini

\- \*\*RAG Framework\*\*: Langchain

\- \*\*Vector Store\*\*: FAISS (CPU version)

\- \*\*Embeddings\*\*: HuggingFace Sentence-Transformers (all-MiniLM-L6-v2, 384-dim)

\- \*\*Deep Learning\*\*: PyTorch

\- \*\*Database\*\*: PostgreSQL (persistent analytics storage)

\- \*\*Dashboard\*\*: Flask web app (separate Render service)

\- \*\*Deployment\*\*:

&nbsp; - Bot: Render $25/month plan (24/7 background worker)

&nbsp; - Dashboard: Render free tier (web service)



\### Key Components



\#### 1. RAG System (`bot.py` lines 195-244)

\- \*\*Data Source\*\*: `Data\_Doc\_main.txt` (curriculum content ~396KB)

\- \*\*Text Processing\*\*: Removes non-ASCII, lowercases, removes punctuation

\- \*\*Chunking\*\*: 1500 chars with 150 char overlap

\- \*\*Retrieval\*\*: Top 3 relevant chunks per query using FAISS similarity search

\- \*\*Embedding Model\*\*: HuggingFace all-MiniLM-L6-v2



\#### 2. LLM Integration (`bot.py` lines 250-430)

\- \*\*Model\*\*: OpenAI GPT-4.1-mini

\- \*\*Rate Limiting\*\*: Semaphore (max 3 concurrent API calls)

\- \*\*Temperature\*\*: 0.8 with presence/frequency penalties

\- \*\*Max Tokens\*\*: 600 per response

\- \*\*System Prompts\*\*:

&nbsp; - NORMAL MODE: Can ask 1 clarifying question if query vague

&nbsp; - ANSWER MODE: Must provide answer (after clarification)



\#### 3. Conversation Management

\- \*\*Storage\*\*: In-memory dictionary `conversation\_history: Dict\[int, List\[Dict\[str, str]]]`

\- \*\*Scope\*\*: Per-thread history (last 8 messages, 4 exchanges)

\- \*\*Clarification Tracking\*\*: `clarification\_tracker: Dict\[int, int]` limits consecutive clarifying questions

\- \*\*Loop Prevention\*\*: Forces answer after 1 clarifying question



\#### 4. Feedback System (`bot.py` lines 93-189)

\*\*Classes\*\*:

\- `FeedbackView`: Initial feedback buttons ("Got it, thanks" / "Need more help")

\- `FollowUpView`: Follow-up actions ("Continue here" / "Tag the crew")



\*\*User Flow\*\*:

```

Student asks question

&nbsp;   ↓

Bot provides solution

&nbsp;   ↓

\[1.5 sec pause]

&nbsp;   ↓

"🎯 Does this clear things up?"

\[✅ Got it, thanks!] \[🔄 Need more help]

&nbsp;   ↓

&nbsp;   ├─→ "Got it": "Awesome! 🚀 Happy learning!" → DONE

&nbsp;   └─→ "Need more help":

&nbsp;       \[💬 Continue here] \[🏴 Tag the crew]

&nbsp;           ↓

&nbsp;           ├─→ "Continue": Conversation continues

&nbsp;           └─→ "Tag the crew": Tags @mekashi @omkar (IDs: 1389934019030028380, 1352199617877381150)

```



\*\*Smart Detection\*\* (`is\_providing\_solution()` function, lines 433-511):

\- Analyzes responses to determine if solution or clarifying question

\- SKIP buttons when: 2+ questions, clarifying phrases detected

\- SHOW buttons when: 3+ solution indicators, lecture references, step-by-step instructions

\- Prevents feedback requests after clarifying questions



\*\*Button Security\*\*:

\- Only original question author can click

\- Buttons auto-disable after interaction

\- Others see: "This feedback is for the person who asked the question! 😊"



\#### 5. Message Handling (`bot.py` lines 590-661)

\- \*\*Trigger\*\*: Bot must be mentioned in forum thread

\- \*\*Image Support\*\*: Handles jpg, png, gif, webp attachments via OpenAI Vision API

\- \*\*Message Splitting\*\*: Auto-splits responses >1900 chars with 0.5s delay

\- \*\*Thread Detection\*\*: Only responds in Discord forum threads



\#### 6. Analytics \& Database System

\*\*Database Schema\*\*:

\- PostgreSQL database tracks all bot interactions

\- Events logged: `query`, `got\_it`, `tag\_crew`, `need\_help`, `continue\_here`

\- Each event stores: timestamp, thread\_id, user\_id, event\_type, metadata



\*\*Data Collection\*\*:

\- Every bot response → `query` event

\- All button clicks → respective event types

\- Thread context captured for analysis

\- Persistent storage survives bot restarts



\*\*Dashboard Application\*\*:

\- Flask web service on separate Render deployment

\- Single-page analytics UI at `/dashboard` route

\- Real-time metrics calculation from PostgreSQL

\- Time-range filters: 7 days / 30 days views

\- Trend visualization with activity graphs



---



\## Current Data \& Storage



\### Persistent Database (PostgreSQL)

\- ✅ \*\*Analytics Events Table\*\*: All bot interactions logged

\- ✅ \*\*Query Tracking\*\*: Every bot response recorded with timestamp, thread, user

\- ✅ \*\*Button Interactions\*\*: All feedback clicks captured

\- ✅ \*\*Historical Data\*\*: Survives bot restarts

\- ✅ \*\*Time-Series Analysis\*\*: Enables trend tracking over days/weeks



\### In-Memory Cache (Conversation State)

\- `conversation\_history`: Thread conversation histories (last 8 messages)

\- `clarification\_tracker`: Clarifying question counts per thread

\- `pending\_feedback`: Tracks feedback messages awaiting response

\- \*\*Note\*\*: Conversation state intentionally ephemeral (resets on restart)



\### Analytics Dashboard

\- ✅ Real-time metrics calculation

\- ✅ Time-range filters (7 days / 30 days)

\- ✅ Satisfaction \& escalation rates

\- ✅ Activity trend graphs

\- ✅ Self-service success tracking



---



\## PRODUCTION ANALYTICS (Last 30 Days)



\### Business Goal

Show program team how Sage bot is being used and measure its effectiveness in solving mentee problems.



\### Key Performance Metrics



\#### Self-Service Success: 92.2%

\- \*\*190 of 206 queries\*\* resolved without mentor intervention

\- Bot successfully handled vast majority of student questions

\- Only 16 escalations required human mentor help (7.8%)



\#### Total Bot Usage: 206 Queries

\- Last 30 days of production activity

\- Consistent engagement from AI Cohort 6 students

\- Active usage across 5 weeks of data collection



\#### Mentor Escalations: 16 (7.8%)

\- Students tagged @mekashi/@omkar for complex issues

\- Low escalation rate shows bot effectiveness

\- "Tag the crew" button clicked 16 times



\#### Explicit Satisfaction Rate: 51.5%

\- \*\*17 "Got it, thanks!"\*\* clicks

\- \*\*16 "Tag the crew"\*\* escalations

\- Ratio shows users click satisfaction more than escalation

\- Note: Many satisfied users may not click buttons



\#### Engagement Rate: 29.1%

\- \*\*60 total button interactions\*\* out of 206 queries

\- Breakdown:

&nbsp; - 17 "Got it, thanks!"

&nbsp; - 16 "Tag the crew"

&nbsp; - 27 "Need more help" + "Continue here" (combined)

\- 70.9% of users read solutions without clicking feedback



\### Activity Trend Analysis

\- \*\*Week 1-5 Data\*\*: Consistent query volume

\- Activity trends tracked with visual graph

\- Dashboard shows weekly breakdown of:

&nbsp; - "Got it, thanks!" clicks

&nbsp; - "Tag the crew" escalations

&nbsp; - Total query volume



\### Dashboard Features Implemented

\- ✅ Time-range toggle: 7 days / 30 days views

\- ✅ Self-service success percentage calculation

\- ✅ Explicit satisfaction rate tracking

\- ✅ Activity trend line graph (5 weeks)

\- ✅ Real-time metrics from PostgreSQL

\- ✅ Clean, educational UI design

\- ✅ Deployed on Render free tier web service



---



\## Complete System Flow



\### User Journey: Question → Answer → Feedback → Analytics



```

1\. STUDENT ASKS QUESTION

&nbsp;  └─→ Mentions @Sage in Discord forum thread

&nbsp;      └─→ Bot detects mention via on\_message event

&nbsp;          └─→ Extracts thread\_id, user\_id, message content

&nbsp;

2\. RAG RETRIEVAL

&nbsp;  └─→ Query embedded using HuggingFace model

&nbsp;      └─→ FAISS searches top 3 relevant curriculum chunks

&nbsp;          └─→ Context passed to OpenAI GPT-4.1-mini

&nbsp;

3\. LLM RESPONSE GENERATION

&nbsp;  └─→ Checks clarification\_tracker (max 1 question)

&nbsp;      ├─→ NORMAL MODE: May ask clarifying question

&nbsp;      └─→ ANSWER MODE: Must provide solution

&nbsp;          └─→ Response generated with curriculum context

&nbsp;

4\. ANALYTICS EVENT LOGGED

&nbsp;  └─→ \`query\` event written to PostgreSQL

&nbsp;      └─→ Stores: timestamp, thread\_id, user\_id, event\_type="query"

&nbsp;

5\. MESSAGE SENT TO DISCORD

&nbsp;  └─→ Response split if >1900 chars

&nbsp;      └─→ Posted in thread

&nbsp;          └─→ Conversation\_history updated (last 8 messages)

&nbsp;

6\. SMART FEEDBACK DETECTION

&nbsp;  └─→ is\_providing\_solution() analyzes response

&nbsp;      ├─→ If clarifying question: NO buttons

&nbsp;      └─→ If solution provided: SHOW feedback buttons (1.5s delay)

&nbsp;

7\. FEEDBACK BUTTONS APPEAR

&nbsp;  └─→ "🎯 Does this clear things up?"

&nbsp;      ├─→ \[✅ Got it, thanks!\]

&nbsp;      └─→ \[🔄 Need more help\]

&nbsp;

8\. USER INTERACTION - PATH A: SATISFIED

&nbsp;  └─→ Clicks "Got it, thanks!"

&nbsp;      └─→ \`got\_it\` event logged to PostgreSQL

&nbsp;          └─→ "Awesome! 🚀 Happy learning!"

&nbsp;              └─→ Buttons disabled → DONE

&nbsp;

9\. USER INTERACTION - PATH B: NEEDS HELP

&nbsp;  └─→ Clicks "Need more help"

&nbsp;      └─→ \`need\_help\` event logged to PostgreSQL

&nbsp;          └─→ Follow-up buttons appear:

&nbsp;              ├─→ \[💬 Continue here\]

&nbsp;              └─→ \[🏴 Tag the crew\]

&nbsp;

10\. FOLLOW-UP ACTION - CONTINUE

&nbsp;   └─→ Clicks "Continue here"

&nbsp;       └─→ \`continue\_here\` event logged

&nbsp;           └─→ "Let's dive deeper! What specifically..."

&nbsp;               └─→ Conversation continues (back to step 1)

&nbsp;

11\. FOLLOW-UP ACTION - ESCALATE

&nbsp;   └─→ Clicks "Tag the crew"

&nbsp;       └─→ \`tag\_crew\` event logged to PostgreSQL

&nbsp;           └─→ Tags @mekashi @omkar in thread

&nbsp;               └─→ "Bringing in the experts! 🚀"

&nbsp;                   └─→ Human mentor notified → ESCALATED

&nbsp;

12\. DASHBOARD ANALYTICS

&nbsp;   └─→ PostgreSQL continuously aggregates events

&nbsp;       └─→ Flask dashboard queries database

&nbsp;           └─→ Calculates metrics:

&nbsp;               ├─→ Total queries: COUNT(event\_type="query")

&nbsp;               ├─→ Satisfaction: COUNT(event\_type="got\_it")

&nbsp;               ├─→ Escalations: COUNT(event\_type="tag\_crew")

&nbsp;               ├─→ Self-service: (Total - Escalations) / Total × 100

&nbsp;               └─→ Engagement: Total\_button\_clicks / Total\_queries × 100

&nbsp;

13\. PROGRAM TEAM VIEWS DASHBOARD

&nbsp;   └─→ Access /dashboard route

&nbsp;       └─→ Toggle 7-day / 30-day view

&nbsp;           └─→ See:

&nbsp;               ├─→ 92.2% self-service success

&nbsp;               ├─→ 206 total queries

&nbsp;               ├─→ 16 escalations (7.8%)

&nbsp;               ├─→ 17 satisfaction clicks (51.5% of engaged users)

&nbsp;               └─→ Weekly activity trend graph

```



\### Key Decision Points



1\. \*\*Clarification Logic\*\*: Bot tracks per-thread clarification count to prevent loops

2\. \*\*Button Display\*\*: Smart detection skips buttons for clarifying questions

3\. \*\*Analytics Timing\*\*: Events logged immediately (not batched) for real-time tracking

4\. \*\*Button Security\*\*: Only original asker can click (user\_id validation)

5\. \*\*Conversation State\*\*: In-memory cache (ephemeral), analytics persistent (PostgreSQL)



---



\## File Structure



\### Core Files

```

discord-support-bot/

├── bot.py                                    # Main bot (677 lines)

├── requirements.txt                          # Python dependencies (35 packages)

├── .env                                      # Environment variables

├── Dockerfile                                # Docker config

├── Procfile                                  # Heroku/Render deployment

├── Data\_Doc\_main.txt                        # Curriculum data (~396KB)

├── curriculum\_comprehensive\_index.txt        # Curriculum index (~17KB)

├── README.md                                 # Documentation (220 lines)

├── sage bot dashboard.txt                    # Feedback system implementation doc (184 lines)

├── CLAUDE.md                                 # Project memory (this file)

└── SAGE\_BOARD\_PRESENTATION.md               # Executive summary for board meeting

```



\### Key Code Sections in bot.py

\- \*\*Lines 1-88\*\*: Imports, globals, config

\- \*\*Lines 93-142\*\*: `FeedbackView` class (initial feedback buttons)

\- \*\*Lines 145-189\*\*: `FollowUpView` class (follow-up action buttons)

\- \*\*Lines 195-244\*\*: RAG system (text loading, vector store, retrieval)

\- \*\*Lines 250-316\*\*: Conversation history management

\- \*\*Lines 321-430\*\*: OpenAI response generation with loop prevention

\- \*\*Lines 433-511\*\*: `is\_providing\_solution()` detection logic

\- \*\*Lines 551-586\*\*: Bot initialization

\- \*\*Lines 590-661\*\*: Main `on\_message` handler



---



\## Environment Variables (.env)

```

\# Discord Bot

DISCORD\_BOT\_TOKEN=<discord\_token>



\# OpenAI API

OPENAI\_API\_KEY=<openai\_key>



\# PostgreSQL Database (shared by bot \& dashboard)

DATABASE\_URL=<postgresql\_connection\_string>

```



\*\*Note\*\*: Both bot and dashboard services use same DATABASE\_URL for analytics tracking.



---



\## Dependencies (requirements.txt)

Key packages:

\- discord.py==2.6.4

\- openai==2.8.1

\- langchain

\- faiss-cpu

\- sentence-transformers

\- torch

\- pandas

\- numpy

\- python-dotenv



---



\## Known Gaps \& Limitations



\### Current Limitations

1\. \*\*Conversation History Ephemeral\*\*: Thread context lost on bot restart (intentional design)

2\. \*\*No Advanced Monitoring\*\*: No APM, error tracking service, or performance profiling

3\. \*\*No A/B Testing\*\*: Can't experiment with different prompt strategies

4\. \*\*Manual Scaling\*\*: No auto-scaling based on load (Render manual scaling only)

5\. \*\*Single Bot Instance\*\*: No distributed deployment or redundancy

6\. \*\*Limited Analytics Retention\*\*: No data archival policy or long-term trend analysis



\### What Works Well

\- ✅ RAG retrieval accuracy

\- ✅ Smart clarification loop prevention

\- ✅ Intelligent feedback button detection

\- ✅ Per-thread conversation context

\- ✅ Image analysis support

\- ✅ Mentor escalation workflow

\- ✅ Message splitting for Discord limits

\- ✅ Analytics dashboard with real-time metrics

\- ✅ Persistent event tracking (PostgreSQL)

\- ✅ 92.2% self-service success rate

\- ✅ Low escalation rate (7.8%)



---



\## Current Implementation Status



\### ✅ Phase 1: Data Collection (COMPLETED)

\- PostgreSQL database connected via psycopg2

\- `analytics` table schema created with fields:

&nbsp; - `id` (primary key)

&nbsp; - `timestamp` (event time)

&nbsp; - `thread\_id` (Discord thread)

&nbsp; - `user\_id` (Discord user)

&nbsp; - `event\_type` (query, got\_it, tag\_crew, need\_help, continue\_here)

&nbsp; - `metadata` (JSON additional data)

\- Events logged in real-time:

&nbsp; - ✅ `query`: Every bot response

&nbsp; - ✅ `got\_it`: "Got it, thanks" button clicks

&nbsp; - ✅ `tag\_crew`: "Tag the crew" button clicks

&nbsp; - ✅ `need\_help`: "Need more help" button clicks

&nbsp; - ✅ `continue\_here`: "Continue here" button clicks



\### ✅ Phase 2: Dashboard Service (COMPLETED)

\- Flask web application built

\- Single `/dashboard` route with clean UI

\- Metrics displayed:

&nbsp; - ✅ Total queries (206 in last 30 days)

&nbsp; - ✅ Self-service success rate (92.2%)

&nbsp; - ✅ Mentor escalations count \& percentage (16, 7.8%)

&nbsp; - ✅ Explicit satisfaction rate (51.5%)

&nbsp; - ✅ Engagement rate (29.1%)

&nbsp; - ✅ Weekly activity trend graph (5 weeks)

\- Time-range filters: 7 days / 30 days toggle



\### ✅ Phase 3: Deployment (COMPLETED)

\- \*\*Choice Made\*\*: Separate Render free tier web service

\- Bot worker: Render $25/month plan (background worker)

\- Dashboard: Render free tier (web service, separate deployment)

\- Both services connect to same PostgreSQL database

\- Dashboard URL accessible to program team



\### Implementation Architecture

```

\[Discord Bot (Render Worker)\]

&nbsp;         ↓

&nbsp;     (logs events)

&nbsp;         ↓

\[PostgreSQL Database\]

&nbsp;         ↑

&nbsp;     (reads data)

&nbsp;         ↑

\[Flask Dashboard (Render Web)\]

```



---



\## Contact \& Context

\- \*\*User\*\*: 100xEngineers Program Team

\- \*\*Cohort\*\*: AI Cohort 6

\- \*\*Mentors\*\*: @mekashi (ID: 1389934019030028380), @omkar (ID: 1352199617877381150)

\- \*\*Deployment\*\*: Render cloud platform

\- \*\*Bot Name\*\*: Sage



---



\*\*Last Updated\*\*: 2026-01-12

\*\*Status\*\*: Dashboard fully deployed \& operational. Tracking 206 queries over 30 days with 92.2% self-service success.


Master UI Design System Prompt 
<UI_aesthetics>
You are a seasoned, art-driven UI designer known for creating bold, intentional, and deeply human digital interfaces. Your work never looks generic, formulaic, or machine-generated. Instead, it shows personality, strong taste, and artistic direction that feels crafted rather than automated.
Your goal: Create interfaces that tell stories, feel intentional, and stand out through thoughtful design decisions. Every element should serve the brand narrative while maintaining clean, accessible, and visually striking execution. They should be distinctive, context-aware, and visually opinionated
Every choice, from typography and images to colour and the smallest interaction, serves the brand narrative and creates an experience that is both clean and memorable.
DESIGN PRINCIPLES TO FOLLOW
1. Typography
Choose expressive, character-rich typefaces that align with the brand story.
Avoid overused families: Inter, Arial, Roboto, system-UI, and Space Grotesk.
Consider display fonts, serif–sans combos, humanist grotesques, or editorial typography.
Typography should communicate brand voice and create visual hierarchy.
Ensure accessible contrast ratios (WCAG AA minimum).
Type choices must answer: "What story does this tell?"
2. Color & Visual Identity
Commit to one clear aesthetic direction that reflects brand personality.
Use CSS variables or design tokens for consistency.
Prefer high-contrast, accessible color combinations (WCAG AA/AAA standards).
Create opinionated palettes: brutalist black/white, warm editorial tones, sophisticated darks, nature-inspired, vibrant neon, or monochromatic depth.
Avoid cliché AI palettes: white - purple gradient - soft blue UI.
Use selective accent colors with intention, not randomly scattered.
Every color must justify its presence in the narrative.
3. Motion & Interaction Design
Motion should be purposeful and enhance storytelling, not distract from content.
Prefer CSS-based animations for HTML/CSS projects.
For React, use Motion or Framer Motion when impact justifies overhead.
Focus on sequence and rhythm: deliberate staggered reveals and entrance choreography.
Respect user preferences: honour reduced motion settings.
One high-quality, purposeful animation beats many scattered micro-interactions.
Ask: "Does this motion serve the user or just look decorative?"
4. Backgrounds & Spatial Design
Avoid flat, solid-colour backgrounds unless intentionally minimalist.
Use layered gradients, subtle noise textures, grain, geometric grids, or contextual patterns.
Create depth through foreground, midground, and background layering.
Backgrounds should add atmosphere and reinforce brand identity without competing with content.
Design sophisticated dark modes, not just color inversions.
Backgrounds should be felt, not noticed.
WHAT TO AVOID AT ALL COSTS
Overused system or Google-style fonts without justification.
Purple/indigo gradients on plain white backgrounds.
Generic "startup aesthetic" that lacks brand specificity.
Inaccessible color combinations that fail WCAG standards.
Excessive, purposeless animations that distract from content.
Designs that look pretty but tell no story.
Homogenous, bland components (cards, buttons, navbars) with no aesthetic identity.
Repeating the same design patterns across different projects.
Falling back to "safe" defaults when brand context demands boldness.
CREATIVE MANDATE: BE UNEXPECTED
Each interface you create must:
Tell a brand-specific story - Every design choice should support the narrative. Generic templates are forbidden.
Exhibit unique visual identity - No two projects should feel like they came from the same template factory.
Take thoughtful creative risks - Push boundaries while maintaining usability and accessibility. Safe design is invisible design.
Maintain clean, elegant execution - Bold doesn't mean cluttered. Distinctive doesn't mean chaotic. Visual clarity is non-negotiable.
Build accessibility into creativity - Accessibility constraints are design challenges that sharpen your work, not limitations to work around.
Surprise and delight - Create moments that make users pause and notice the craft, not skim past another generic interface.
When interpreting instructions, default to originality over safety. If the result feels familiar or formulaic, rethink it.
Design with conviction. Tell stories worth experiencing. Create interfaces that feel unmistakably human.
</UI_aesthetics>




