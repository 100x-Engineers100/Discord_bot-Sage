"""
Dual Discord Bot System - Sage + Scout
======================================
Runs two bots in a single process:
- Sage (@Sage): Technical curriculum support with RAG, feedback buttons, analytics
- Scout (@Scout): FAQ/program support with RAG, urgency detection

Both bots share infrastructure (OpenAI client, utilities, database) but have
separate personalities and response logic.
"""

import sys; print("[D1] top", flush=True)
import os
os.environ.setdefault("AIOHTTP_NO_EXTENSIONS", "1")
print("[D2] os+env", flush=True)

import re
import time
import asyncio
print(f"[D3] stdlib, AIOHTTP_NO_EXTENSIONS={os.environ.get('AIOHTTP_NO_EXTENSIONS')}", flush=True)
import aiohttp
print("[D4] aiohttp", flush=True)
import numpy as np
print("[D5] numpy", flush=True)
from typing import List, Dict, Optional
from dotenv import load_dotenv
print("[D6] dotenv", flush=True)

import discord
print("[D7] discord", flush=True)
from discord.ext import commands
from discord.ui import Button, View
print("[D8] discord.ext", flush=True)

from openai import OpenAI
print("[D9] openai", flush=True)
from supabase import create_client, Client
print("[D10] supabase", flush=True)

# Import RAG systems
from curriculum_rag import CurriculumRAG
print("[D11] currRAG", flush=True)
from scout_rag import ScoutRAG
print("[D12] scoutRAG", flush=True)

# Import dual-RAG routing
from bot_routing_utils import route_query, generate_redirect_message
print("[D13] routing done", flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Bot Tokens
SAGE_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')  # Sage = technical support
SCOUT_BOT_TOKEN = os.getenv('SCOUT_BOT_TOKEN')   # Scout = FAQ support

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Supabase Configuration (Analytics - Sage only)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Message Configuration
MAX_DISCORD_MESSAGE_LENGTH = 1500
MAX_CLARIFICATIONS = 1

# Mentor IDs (for both bots)
MENTOR_IDS = [
    "<@1297830510453854228>",  # YATISH
    "<@1352199617877381150>"   # Omkar
]

# Mentor name -> ID map (for resolving LLM plain-text @mentions)
MENTOR_NAME_MAP = {
    "yatish": "1297830510453854228",
    "omkar": "1352199617877381150",
}

# Cross-bot app IDs (for real Discord mentions between bots)
SAGE_BOT_ID = 1448961727222906882
SCOUT_BOT_ID = 1453359679290740830

# Scout Urgency Keywords
URGENT_KEYWORDS = [
    "can't access", "not working", "deadline", "today", "asap",
    "urgent", "credits depleted", "error", "crashed", "help urgently",
    # distress / negative sentiment
    "wasted my money", "waste of money", "frustrated", "frustrating",
    "not understanding anything", "don't understand anything",
    "useless", "worst", "terrible", "refund", "disappointed"
]

# ============================================================================
# GLOBAL STATE (Shared by both bots)
# ============================================================================

# RAG systems
sage_rag = None
scout_rag = None
openai_client = None
supabase_client: Optional[Client] = None
api_semaphore = asyncio.Semaphore(3)

# Sage-specific state (per-thread tracking)
sage_conversation_history: Dict[int, List[Dict[str, str]]] = {}
sage_pending_feedback: Dict[int, Dict[str, int]] = {}
sage_clarification_tracker: Dict[int, int] = {}
sage_feedback_pending: Dict[int, bool] = {}
sage_last_feedback_shown: Dict[int, float] = {}
sage_feedback_completed: Dict[int, bool] = {}

# Scout-specific state (per-thread tracking)
scout_conversation_history: Dict[int, List[Dict[str, str]]] = {}

# Cross-bot forward tracking: thread_id -> 'scout' or 'sage'
# Meaning: "this bot forwarded a question to the other; the other bot
# should process exactly that one message and then stop."
# Cleared (popped) the moment the receiving bot starts processing.
cross_bot_forwards: Dict[int, str] = {}

# ============================================================================
# DISCORD BOT SETUP (Two bots)
# ============================================================================

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

sage_bot = commands.Bot(command_prefix='!sage ', intents=intents)
scout_bot = commands.Bot(command_prefix='!scout ', intents=intents)

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def resolve_mentor_mentions(text: str) -> str:
    """Replace plain-text @name mentions with proper Discord <@ID> mentions.

    LLMs cannot produce Discord mention syntax -- they output @name as plain text.
    This post-processes the response so mentions actually ping."""
    for name, uid in MENTOR_NAME_MAP.items():
        # Match @Name only when NOT already inside <@...> format
        text = re.sub(rf'(?<!<)@{name}\b', f'<@{uid}>', text, flags=re.IGNORECASE)
    return text


def detect_cross_bot_tag(raw_content: str, mentions, current_bot: str) -> bool:
    """Return True when user is asking current bot to forward to the other bot.

    Checks two signals:
      1. A forwarding trigger word is present (tag / ask / mention / forward / pass)
      2. The other bot is referenced -- either as plain text or as a real mention
    """
    content_lower = raw_content.lower()
    other_name = 'sage' if current_bot == 'scout' else 'scout'
    other_bot_id = SAGE_BOT_ID if current_bot == 'scout' else SCOUT_BOT_ID

    triggers = ['tag', 'ask', 'mention', 'forward', 'pass']
    if not any(t in content_lower for t in triggers):
        return False

    # Plain-text name reference
    if other_name in content_lower:
        return True

    # Real Discord mention of the other bot
    for m in mentions:
        if m.id == other_bot_id:
            return True

    return False


def extract_forwarded_question(query: str, thread_id: int, current_bot: str) -> str:
    """Pull the original question out of a tag-request message.

    Strips known forwarding phrases (longest-first to avoid partial clobber).
    If nothing meaningful remains, falls back to the last user message in
    that thread's conversation history.
    """
    strip_phrases = [
        'can you tag sage and then get the answer',
        'can you tag scout and then get the answer',
        'tag sage and then get the answer',
        'tag scout and then get the answer',
        'can you tag sage and get the answer',
        'can you tag scout and get the answer',
        'tag sage and get the answer',
        'tag scout and get the answer',
        'please tag sage',
        'please tag scout',
        'can you tag sage',
        'can you tag scout',
        'can u tag sage',
        'can u tag scout',
        'tag sage',
        'tag scout',
        'ask sage',
        'ask scout',
        'and then get the answer',
        'and get the answer',
        'get the answer',
    ]

    cleaned = query
    for phrase in strip_phrases:
        lower = cleaned.lower()
        idx = lower.find(phrase)
        if idx != -1:
            cleaned = cleaned[:idx] + cleaned[idx + len(phrase):]

    cleaned = cleaned.strip(' .,!?\n')

    if len(cleaned) > 5:
        return cleaned

    # Fall back to last user message in thread history
    history = scout_conversation_history if current_bot == 'scout' else sage_conversation_history
    for msg in reversed(history.get(thread_id, [])):
        if msg['role'] == 'user':
            return msg['content']

    return "Please refer to the previous question in this thread."


def split_long_message(content: str) -> List[str]:
    """Split long messages into Discord-compatible chunks."""
    if len(content) <= MAX_DISCORD_MESSAGE_LENGTH:
        return [content]

    chunks = []
    paragraphs = content.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > MAX_DISCORD_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                words = para.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 > MAX_DISCORD_MESSAGE_LENGTH:
                        chunks.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        current_chunk += " " + word if current_chunk else word
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


async def log_event(event_type: str, thread_id: int, user_id: int, message_id: int, extra_metadata: dict = None):
    """Log analytics event to Supabase (Sage only)."""
    if not supabase_client:
        return

    try:
        metadata = {"message_id": str(message_id)}
        if extra_metadata:
            metadata.update(extra_metadata)
        data = {
            "event_type": event_type,
            "thread_id": str(thread_id),
            "user_id": str(user_id),
            "metadata": metadata
        }
        supabase_client.table('analytics_events').insert(data).execute()
    except Exception as e:
        print(f"Analytics logging error: {e}")


# ============================================================================
# SAGE-SPECIFIC: FEEDBACK SYSTEM
# ============================================================================

class FeedbackView(View):
    """Discord UI View with buttons for initial feedback (Sage only)."""
    def __init__(self, user_id: int, thread_id: int):
        super().__init__(timeout=None)
        self.user_id = user_id
        self.thread_id = thread_id

    @discord.ui.button(label="✅ Got it, thanks!", style=discord.ButtonStyle.success)
    async def got_it_button(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This feedback is for the person who asked the question! 😊",
                ephemeral=True
            )
            return

        await log_event('got_it', self.thread_id, interaction.user.id, interaction.message.id)

        sage_feedback_pending[self.thread_id] = False
        sage_feedback_completed[self.thread_id] = True

        await interaction.response.send_message(
            "Awesome! 🚀 Happy learning!",
            ephemeral=False
        )

        for item in self.children:
            item.disabled = True
        await interaction.message.edit(view=self)

        if interaction.message.id in sage_pending_feedback:
            del sage_pending_feedback[interaction.message.id]

    @discord.ui.button(label="🔄 Need more help", style=discord.ButtonStyle.secondary)
    async def need_help_button(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This feedback is for the person who asked the question! 😊",
                ephemeral=True
            )
            return

        await log_event('need_help', self.thread_id, interaction.user.id, interaction.message.id)

        sage_feedback_pending[self.thread_id] = False

        for item in self.children:
            item.disabled = True
        await interaction.message.edit(view=self)

        follow_up_view = FollowUpView(self.user_id, self.thread_id)
        await interaction.response.send_message(
            "No worries, let's figure this out! What would you prefer?",
            view=follow_up_view,
            ephemeral=False
        )

        if interaction.message.id in sage_pending_feedback:
            del sage_pending_feedback[interaction.message.id]


class FollowUpView(View):
    """Discord UI View with buttons for follow-up actions (Sage only)."""
    def __init__(self, user_id: int, thread_id: int):
        super().__init__(timeout=None)
        self.user_id = user_id
        self.thread_id = thread_id

    @discord.ui.button(label="💬 Continue here", style=discord.ButtonStyle.primary)
    async def continue_button(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This option is for the person who asked the question! 😊",
                ephemeral=True
            )
            return

        await log_event('continue_here', self.thread_id, interaction.user.id, interaction.message.id)

        sage_feedback_completed[self.thread_id] = False

        await interaction.response.send_message(
            "Got it! What's still unclear or what else can I help with?",
            ephemeral=False
        )

        for item in self.children:
            item.disabled = True
        await interaction.message.edit(view=self)

    @discord.ui.button(label="🏴 Tag the crew", style=discord.ButtonStyle.danger)
    async def tag_mentors_button(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This option is for the person who asked the question! 😊",
                ephemeral=True
            )
            return

        await log_event('tag_crew', self.thread_id, interaction.user.id, interaction.message.id)

        sage_feedback_pending[self.thread_id] = False
        sage_feedback_completed[self.thread_id] = True

        mentor_tags = " ".join(MENTOR_IDS)
        await interaction.response.send_message(
            f"Roger that! 📣 Bringing in reinforcements...\n\n"
            f"Hey {mentor_tags}, this one needs your expertise!\n\n"
            f"<@{self.user_id}> - they'll jump in soon to help you out! 🤝",
            ephemeral=False
        )

        for item in self.children:
            item.disabled = True
        await interaction.message.edit(view=self)


# ===========================================================================
# SAGE-SPECIFIC: RAG & CONVERSATION LOGIC
# ============================================================================

def format_curriculum_context(results: list) -> str:
    """Format curriculum RAG results into context string for LLM."""
    if not results:
        return "No relevant curriculum content found."

    context_parts = []
    for result in results:
        module_info = f"Module {result.get('module', '?')}: {result.get('module_name', 'Unknown')}"
        mentor = result.get('mentor', 'Unknown')

        context_parts.append(
            f"[Lecture {result['lecture_num']}: {result['lecture_name']}]\n"
            f"{module_info} (Mentor: {mentor})\n"
            f"Topics: {', '.join(result['topics'][:5])}\n\n"
            f"{result['content'][:2000]}\n"
        )

    return "\n---\n\n".join(context_parts)


def get_sage_thread_history(thread_id: int) -> List[Dict[str, str]]:
    """Get Sage conversation history for thread."""
    return sage_conversation_history.get(thread_id, [])


def add_to_sage_history(thread_id: int, role: str, content: str):
    """Add message to Sage thread history."""
    if thread_id not in sage_conversation_history:
        sage_conversation_history[thread_id] = []

    sage_conversation_history[thread_id].append({
        "role": role,
        "content": content
    })

    sage_conversation_history[thread_id] = sage_conversation_history[thread_id][-10:]


async def generate_sage_response(
    query: str,
    context: str,
    history: List[Dict[str, str]],
    thread_id: int,
    image_urls: List[str] = None
) -> str:
    """Generate Sage response with clarification loop prevention."""
    async with api_semaphore:
        try:
            clarification_count = sage_clarification_tracker.get(thread_id, 0)

            if clarification_count >= MAX_CLARIFICATIONS:
                mode = "ANSWER"
            else:
                mode = "NORMAL"

            if mode == "NORMAL":
                system_prompt = """You are Sage, technical learning assistant for 100xEngineers AI Cohort 6.

Your role:
- Help students understand curriculum concepts (Diffusion, LLMs, Agents)
- Debug code issues and explain implementations
- Answer general technical/programming questions using your knowledge
- Ask 1 clarifying question if query is too vague

Response rules:
- If curriculum context is available: use it to ground your answer. ALWAYS cite the specific lecture name and module you are referencing (e.g., "As covered in Lecture 3: RAG Basics, Module 2: Full Stack LLM...")
- If no curriculum context: answer directly from your general technical knowledge
- Never refuse a technical question -- always try to help
- If query is vague: ask ONE specific clarifying question
- Be extra concise and helpful
"""
            else:
                system_prompt = """You are Sage, technical learning assistant for 100xEngineers AI Cohort 6.

IMPORTANT: You MUST provide an answer now. No more clarifying questions allowed.

Your role:
- Help students with curriculum concepts and general technical questions
- Debug code issues and explain implementations

Response rules:
- Provide best answer possible with available information
- If curriculum context is available: use it and ALWAYS cite the lecture name and module (e.g., "As covered in Lecture 5: Agents, Module 3...")
- If no curriculum context: use general technical knowledge
- Make reasonable assumptions if needed
- Be extra concise and helpful
"""

            messages = [{"role": "system", "content": system_prompt}]

            messages.append({"role": "user", "content": f"Curriculum Context:\n{context}"})

            messages.extend(history[-8:])

            # Build final user message: text + optional images
            if image_urls:
                user_content = [{"type": "text", "text": query}]
                for url in image_urls:
                    user_content.append({"type": "image_url", "image_url": {"url": url}})
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": query})

            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )

            assistant_reply = response.choices[0].message.content

            if '?' in assistant_reply and mode == "NORMAL":
                sage_clarification_tracker[thread_id] = clarification_count + 1
            else:
                sage_clarification_tracker[thread_id] = 0

            return assistant_reply

        except Exception as e:
            print(f"Sage response generation error: {e}")
            return "I encountered an error. Try rephrasing or tag a mentor if needed."


def should_show_sage_feedback(query: str, response: str, thread_id: int) -> bool:
    """Determine if Sage should show feedback buttons."""
    meta_patterns = [
        'hello', 'hi', 'hey', 'thanks', 'thank you',
        'help', 'who are you', 'what are your capabilities'
    ]
    user_lower = query.lower()
    if any(pattern in user_lower for pattern in meta_patterns):
        return False

    if sage_feedback_pending.get(thread_id, False):
        return False

    if thread_id in sage_last_feedback_shown:
        elapsed = time.time() - sage_last_feedback_shown[thread_id]
        if elapsed < 30:
            return False

    if sage_feedback_completed.get(thread_id, False):
        return False

    response_lower = response.lower()
    question_count = response.count('?')

    if question_count >= 2:
        return False

    clarifying_phrases = [
        'could you clarify', 'need more information', 'which one',
        'can you provide', 'what specifically', 'could you please share'
    ]
    if any(phrase in response_lower for phrase in clarifying_phrases):
        return False

    if '```' in response and len(response) >= 50:
        sage_feedback_pending[thread_id] = True
        sage_last_feedback_shown[thread_id] = time.time()
        return True

    if ('lecture' in response_lower or 'module' in response_lower) and len(response) >= 80:
        sage_feedback_pending[thread_id] = True
        sage_last_feedback_shown[thread_id] = time.time()
        return True

    instructional = ["here's how", "try this", "you can", "you need to", "step 1", "first,"]
    if any(phrase in response_lower for phrase in instructional) and len(response) >= 80:
        sage_feedback_pending[thread_id] = True
        sage_last_feedback_shown[thread_id] = time.time()
        return True

    if len(response) >= 200:
        sage_feedback_pending[thread_id] = True
        sage_last_feedback_shown[thread_id] = time.time()
        return True

    return False


# ============================================================================
# SCOUT-SPECIFIC: RAG & CONVERSATION LOGIC
# ============================================================================

def format_faq_context(results: list) -> str:
    """Format FAQ RAG results into context string for LLM."""
    if not results:
        return "No relevant FAQ found."

    context_parts = []
    for result in results:
        context_parts.append(
            f"Q: {result['question']}\n"
            f"A: {result['answer']}\n"
            f"Category: {result.get('category', 'general')}\n"
        )

    return "\n---\n\n".join(context_parts)


def get_scout_thread_history(thread_id: int) -> List[Dict[str, str]]:
    """Get Scout conversation history for thread."""
    return scout_conversation_history.get(thread_id, [])


def add_to_scout_history(thread_id: int, role: str, content: str):
    """Add message to Scout thread history."""
    if thread_id not in scout_conversation_history:
        scout_conversation_history[thread_id] = []

    scout_conversation_history[thread_id].append({
        "role": role,
        "content": content
    })

    scout_conversation_history[thread_id] = scout_conversation_history[thread_id][-5:]


def is_urgent_query(query: str) -> bool:
    """Check if query contains urgent keywords."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in URGENT_KEYWORDS)


async def generate_scout_response(
    query: str,
    context: str,
    history: List[Dict[str, str]]
) -> str:
    """Generate Scout response for FAQ queries."""
    async with api_semaphore:
        try:
            system_prompt = """You are Scout, program support assistant for 100xEngineers AI Cohort 6.

Your role:
- Answer questions about LMS, Launchpad, sessions, recordings, Jarvis Labs
- Provide exact links  only if available in documentation, do not hallucinate links or give fake links(must )
- Give step-by-step instructions when needed
- when u get urgent query tag omkar and yatish and do not give links or anything. just tag them and say that the query is urgent.

Guidelines:
- Be concise and helpful
- include exact links when available only from the document, not all the time. 
- If query needs human intervention, say so clearly
- For curriculum queries, refer to module/week/lecture numbers from index and importantly also suggest to tag SAGE.
- if u dont know the answer, say so clearly and tag omkar and yatish and dont tag Sage in it.
- for the distress queries and queries that negative reviews about cohort/course/lecture, tag omkar and yatish and dont tag Sage in it. in this type of queries, dont give links or anything just tag them and say that the query is urgent.
"""

            messages = [{"role": "system", "content": system_prompt}]

            messages.append({
                "role": "user",
                "content": f"Relevant FAQ Documentation:\n{context}"
            })

            messages.extend(history[-4:])

            messages.append({"role": "user", "content": query})

            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Scout response generation error: {e}")
            return "I encountered an error. Please try again or contact the program team by tagging Omkar or Yatish"


# ============================================================================
# SAGE BOT: EVENT HANDLERS
# ============================================================================

@sage_bot.event
async def on_ready():
    """Initialize Sage bot when connected."""
    global openai_client, sage_rag, supabase_client

    print(f'[SAGE] Bot logged in as {sage_bot.user.name} (ID: {sage_bot.user.id})')

    if not openai_client:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print('[SHARED] OpenAI client initialized')

    if not supabase_client and SUPABASE_URL and SUPABASE_KEY:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print('[SHARED] Supabase client initialized')

    print('[SAGE] Loading curriculum RAG system...')
    sage_rag = CurriculumRAG()
    print('[SAGE] RAG system ready!')

    print('[SAGE] Ready to help with technical queries!')


@sage_bot.event
async def on_message(message):
    """Sage message handler."""
    if message.author == sage_bot.user:
        return

    if message.channel.type not in [discord.ChannelType.public_thread, discord.ChannelType.private_thread]:
        return

    if sage_bot.user not in message.mentions:
        return

    thread_id = message.channel.id

    # Bot-message guard: only process messages from other bots if Scout
    # explicitly forwarded a question to us. Pop the flag so we consume
    # exactly one forwarded message -- any subsequent replies from Scout
    # (e.g. triggered by Discord auto-mention on reply) are ignored.
    if message.author.bot:
        if cross_bot_forwards.pop(thread_id, None) != 'scout':
            return

    query = message.content
    for mention in message.mentions:
        query = query.replace(f'<@{mention.id}>', '').strip()

    # Extract image attachments (Vision API)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    image_urls = [
        a.url for a in message.attachments
        if any(a.filename.lower().endswith(ext) for ext in image_extensions)
    ]

    # Extract text/code file attachments and append content to query
    text_extensions = {'.txt', '.py', '.js', '.ts', '.json', '.md', '.yaml', '.yml', '.csv', '.html', '.css', '.sh', '.env', '.toml', '.ini', '.xml'}
    for attachment in message.attachments:
        if any(attachment.filename.lower().endswith(ext) for ext in text_extensions):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        file_content = await resp.text(encoding='utf-8', errors='replace')
                        query += f"\n\n[File: {attachment.filename}]\n```\n{file_content[:4000]}\n```"
            except Exception as e:
                print(f"[SAGE] Failed to read attachment {attachment.filename}: {e}")

    if not query and not image_urls:
        await message.reply(
            "Hi! I'm Sage, your technical learning assistant. Ask me about curriculum concepts, code issues, or implementations!"
        )
        return

    if not query and image_urls:
        query = "Please analyze this image and help me understand what's shown."

    # Cross-bot forward: if user asks Sage to tag Scout, post a real mention + original question
    if not message.author.bot and detect_cross_bot_tag(message.content, message.mentions, 'sage'):
        original_q = extract_forwarded_question(query, thread_id, 'sage')
        cross_bot_forwards[thread_id] = 'sage'
        await message.reply("Got it! Tagging Scout with your question now.")
        await message.channel.send(f"<@{SCOUT_BOT_ID}> {original_q}")
        add_to_sage_history(thread_id, "user", query)
        return

    async with message.channel.typing():
        try:
            # Embed query once -- reuse for both RAG searches
            embed_response = openai_client.embeddings.create(
                input=[query[:8192]],  # embedding has token limit
                model=EMBEDDING_MODEL
            )
            query_embedding = np.array([embed_response.data[0].embedding], dtype=np.float32)

            # Run curriculum search (primary) + FAQ cross-search (for routing)
            curr_results = sage_rag.search(query, top_k=3, query_embedding=query_embedding) if sage_rag and sage_rag.index else []
            faq_results = scout_rag.search(query, top_k=1, query_embedding=query_embedding) if scout_rag and scout_rag.index else []

            # Route: redirect to Scout if FAQ is clearly the better match
            should_redirect, reason = route_query('sage', curr_results, faq_results)

            if should_redirect:
                redirect_msg = generate_redirect_message('sage', reason)
                await message.reply(redirect_msg)
                return

            # Format context: use curriculum if good match, else signal general knowledge
            has_curriculum_match = curr_results and curr_results[0].get('score', 999) < 1.5
            if has_curriculum_match:
                context = format_curriculum_context(curr_results)
            else:
                context = "No relevant curriculum content found for this query."

            # Generate Sage response
            history = get_sage_thread_history(thread_id)

            response = await generate_sage_response(query, context, history, thread_id, image_urls=image_urls or None)

            message_chunks = split_long_message(response)

            first_message = await message.reply(message_chunks[0], mention_author=not message.author.bot)

            for chunk in message_chunks[1:]:
                await message.channel.send(chunk)
                await asyncio.sleep(0.5)

            add_to_sage_history(thread_id, "user", query)
            add_to_sage_history(thread_id, "assistant", response)

            await log_event('query', thread_id, message.author.id, first_message.id, extra_metadata={"query_text": query})

            if should_show_sage_feedback(query, response, thread_id):
                await asyncio.sleep(1.5)

                feedback_view = FeedbackView(user_id=message.author.id, thread_id=thread_id)
                feedback_message = await message.channel.send(
                    "🎯 Does this clear things up?",
                    view=feedback_view
                )

                sage_pending_feedback[feedback_message.id] = {
                    "thread_id": thread_id,
                    "user_id": message.author.id
                }

        except Exception as e:
            print(f"[SAGE] Error processing message: {e}")
            await message.reply(
                "I encountered an error. Try rephrasing or tag a mentor if needed."
            )


# ============================================================================
# SCOUT BOT: EVENT HANDLERS
# ============================================================================

@scout_bot.event
async def on_ready():
    """Initialize Scout bot when connected."""
    global openai_client, scout_rag

    print(f'[SCOUT] Bot logged in as {scout_bot.user.name} (ID: {scout_bot.user.id})')

    if not openai_client:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print('[SHARED] OpenAI client initialized')

    print('[SCOUT] Loading FAQ RAG system...')
    scout_rag = ScoutRAG()
    print('[SCOUT] RAG system ready!')

    print('[SCOUT] Ready to help with program questions!')


@scout_bot.event
async def on_message(message):
    """Scout message handler."""
    if message.author == scout_bot.user:
        return

    if message.channel.type not in [discord.ChannelType.public_thread, discord.ChannelType.private_thread]:
        return

    if scout_bot.user not in message.mentions:
        return

    thread_id = message.channel.id

    # Bot-message guard: only process messages from other bots if Sage
    # explicitly forwarded a question to us. Pop the flag so we consume
    # exactly one forwarded message -- any subsequent replies from Sage
    # (e.g. triggered by Discord auto-mention on reply) are ignored.
    if message.author.bot:
        if cross_bot_forwards.pop(thread_id, None) != 'sage':
            return

    query = message.content
    for mention in message.mentions:
        query = query.replace(f'<@{mention.id}>', '').strip()

    if not query:
        await message.reply(
            "Hi! I'm Scout, your program support assistant. Ask me about LMS, Launchpad, sessions, recordings, or Jarvis Labs!"
        )
        return

    # Cross-bot forward: if user asks Scout to tag Sage, post a real mention + original question
    if not message.author.bot and detect_cross_bot_tag(message.content, message.mentions, 'scout'):
        original_q = extract_forwarded_question(query, thread_id, 'scout')
        cross_bot_forwards[thread_id] = 'scout'
        await message.reply("Got it! Tagging Sage with your question now.")
        await message.channel.send(f"<@{SAGE_BOT_ID}> {original_q}")
        add_to_scout_history(thread_id, "user", query)
        return

    async with message.channel.typing():
        try:
            # Embed query once -- reuse for both RAG searches
            embed_response = openai_client.embeddings.create(
                input=[query],
                model=EMBEDDING_MODEL
            )
            query_embedding = np.array([embed_response.data[0].embedding], dtype=np.float32)

            # Run FAQ search (primary) + curriculum cross-search (for routing)
            faq_results = scout_rag.search(query, top_k=3, query_embedding=query_embedding) if scout_rag and scout_rag.index else []
            curr_results = sage_rag.search(query, top_k=1, query_embedding=query_embedding) if sage_rag and sage_rag.index else []

            # Route: redirect to Sage if curriculum is clearly better or FAQ has no match
            should_redirect, reason = route_query('scout', curr_results, faq_results)

            if should_redirect:
                redirect_msg = generate_redirect_message('scout', reason)
                await message.reply(redirect_msg)
                return

            # Format FAQ context
            context = format_faq_context(faq_results)

            # Generate Scout response
            is_urgent = is_urgent_query(query)
            history = get_scout_thread_history(thread_id)

            response = await generate_scout_response(query, context, history)

            if is_urgent:
                urgency_note = f"\n\n⚠️ **Urgent Query Detected** - Tagging team: {' '.join(MENTOR_IDS)}"
                response = response + urgency_note

            # Resolve any plain-text @name mentions the LLM produced into real Discord pings
            response = resolve_mentor_mentions(response)

            message_chunks = split_long_message(response)

            first_message = await message.reply(message_chunks[0], mention_author=not message.author.bot)

            for chunk in message_chunks[1:]:
                await message.channel.send(chunk)
                await asyncio.sleep(0.5)

            add_to_scout_history(thread_id, "user", query)
            add_to_scout_history(thread_id, "assistant", response)

        except Exception as e:
            print(f"[SCOUT] Error processing message: {e}")
            await message.reply(
                "I encountered an error. Please try rephrasing or contact the program team at cohort@100xengineers.com"
            )


# ============================================================================
# MAIN: RUN BOTH BOTS
# ============================================================================

async def main():
    """Run both Sage and Scout bots in parallel."""
    if not SAGE_BOT_TOKEN:
        print("[ERROR] DISCORD_BOT_TOKEN (Sage) not found in .env")
        return

    if not SCOUT_BOT_TOKEN or SCOUT_BOT_TOKEN == "PLACEHOLDER_REPLACE_WITH_SCOUT_TOKEN":
        print("[ERROR] SCOUT_BOT_TOKEN not found or not replaced in .env")
        return

    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not found in .env")
        return

    print("=" * 60)
    print("Starting Dual Bot System: Sage + Scout")
    print("=" * 60)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(sage_bot.start(SAGE_BOT_TOKEN))
        tg.create_task(scout_bot.start(SCOUT_BOT_TOKEN))


if __name__ == "__main__":
    print("[DEBUG] about to asyncio.run(main())", flush=True)  # temp debug
    asyncio.run(main())
