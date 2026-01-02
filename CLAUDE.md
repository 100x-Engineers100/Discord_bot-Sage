\# Sage Bot - Project Memory



\## Project Overview



\*\*Sage\*\* is a RAG-powered Discord support bot built for \*\*100xEngineers AI Cohort 6\*\* students. It provides technical Q\&A assistance in Discord forum threads using curriculum-grounded responses.



\### Core Purpose

\- Answer student technical questions using RAG (Retrieval-Augmented Generation)

\- Maintain conversation context per thread

\- Provide intelligent feedback system with escalation to mentors

\- Help students solve problems efficiently while tracking satisfaction



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

\- \*\*Deployment\*\*: Render ($25/month plan, 24/7 background worker)



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



---



\## Current Data \& Storage



\### In-Memory Data (Not Persisted)

\- `conversation\_history`: Thread conversation histories (last 8 messages)

\- `clarification\_tracker`: Clarifying question counts per thread

\- `pending\_feedback`: Tracks feedback messages awaiting response

\- \*\*Problem\*\*: All data lost on bot restart



\### No Analytics Tracking

\- ❌ No database

\- ❌ Button clicks ("Got it, thanks", "Tag the crew") NOT recorded

\- ❌ No query count tracking

\- ❌ No satisfaction metrics

\- ❌ No engagement analytics

\- ❌ No resolution rate calculations



\### Existing Logging

\- Console logging only (vector store creation, errors, bot init)

\- No structured logging framework

\- No persistent logs



---



\## NEW REQUIREMENT: Analytics Dashboard



\### Business Goal

Show program team how Sage bot is being used and measure its effectiveness in solving mentee problems.



\### Required Metrics



\#### Primary Metrics

1\. \*\*Total Bot Usage\*\*: How many times bot has been used (total queries answered)

2\. \*\*Satisfaction Rate\*\*: How many times "Got it, thanks" button clicked

3\. \*\*Escalation Rate\*\*: How many times "Tag the crew" button clicked



\#### Derived Metrics

\- \*\*Resolution Rate\*\*: (Got it clicks / Total queries) × 100

\- \*\*Escalation Rate\*\*: (Tag crew clicks / Total queries) × 100

\- \*\*Continuation Rate\*\*: "Need more help" → "Continue here" clicks



\### Dashboard Requirements

\- Simple, clean UI showing key numbers

\- Track historical trends (optional graphs)

\- Show bot effectiveness to program team

\- Internal use only (for program team)



\### Infrastructure Constraints

\- Bot currently runs on Render $25/month plan (background worker, 24/7)

\- Can use same service or add Render free tier web service

\- Need persistent storage (currently everything in-memory)



\### Open Questions (To Be Answered)

1\. \*\*Hosting\*\*: Same server as bot or separate Render free tier web service?

2\. \*\*Database\*\*: SQLite (simple) vs PostgreSQL (scalable) vs Supabase (managed)?

3\. \*\*Dashboard UI\*\*: Just numbers or include graphs/trends?

4\. \*\*Authentication\*\*: Password-protected or obscure URL only?

5\. \*\*Time Range\*\*: All-time stats or last 7/30 days filter?

6\. \*\*Historical Data\*\*: Track from today or backfill existing data?



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

└── PROJECT\_MEMORY.md                         # This file

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

DISCORD\_BOT\_TOKEN=<discord\_token>

OPENAI\_API\_KEY=<openai\_key>

```



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

1\. \*\*No Persistence\*\*: All conversation history and state in-memory

2\. \*\*No Analytics\*\*: Button clicks and usage not tracked

3\. \*\*No Monitoring\*\*: No error tracking, no performance metrics

4\. \*\*No A/B Testing\*\*: Can't measure feedback system effectiveness

5\. \*\*Manual Scaling\*\*: No auto-scaling based on load

6\. \*\*Single Bot Instance\*\*: No distributed deployment support



\### What Works Well

\- ✅ RAG retrieval accuracy

\- ✅ Smart clarification loop prevention

\- ✅ Intelligent feedback button detection

\- ✅ Per-thread conversation context

\- ✅ Image analysis support

\- ✅ Mentor escalation workflow

\- ✅ Message splitting for Discord limits



---



\## Next Steps: Dashboard Implementation



\### Phase 1: Data Collection (Bot Changes)

\- Add database connection (PostgreSQL or Supabase)

\- Create `analytics` table schema

\- Log events:

&nbsp; - `query`: Every bot response

&nbsp; - `got\_it`: "Got it, thanks" button clicks

&nbsp; - `tag\_crew`: "Tag the crew" button clicks

&nbsp; - `need\_help`: "Need more help" button clicks

&nbsp; - `continue\_here`: "Continue here" button clicks



\### Phase 2: Dashboard Service

\- Create Flask/FastAPI web service

\- Single `/dashboard` route

\- Display metrics:

&nbsp; - Total queries

&nbsp; - "Got it" count + percentage

&nbsp; - "Tag crew" count + percentage

&nbsp; - Optional: Daily/weekly trends graph



\### Phase 3: Deployment

\- Option A: Separate Render free tier web service

\- Option B: Embed Flask in bot worker (if Render allows HTTP on background workers)



---



\## Contact \& Context

\- \*\*User\*\*: 100xEngineers Program Team

\- \*\*Cohort\*\*: AI Cohort 6

\- \*\*Mentors\*\*: @mekashi (ID: 1389934019030028380), @omkar (ID: 1352199617877381150)

\- \*\*Deployment\*\*: Render cloud platform

\- \*\*Bot Name\*\*: Sage



---



\*\*Last Updated\*\*: 2026-01-01

\*\*Status\*\*: Dashboard feature in planning phase


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




