"""
Discord Support Bot V4 - Technical Learning Assistant (RAG 2.0)
====================================================
A RAG-powered Discord bot that helps students with technical queries in a forum setting.

IMPROVEMENTS IN V4:
- New RAG pipeline with OpenAI embeddings (better accuracy)
- Proper conversation history (OpenAI message format)
- Re-retrieval on every message (no stale context)
- Structured lecture tracking with modules/mentors
- Removed truncation limits (full context)
"""

import os
import re
import time
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dotenv import load_dotenv

import discord
from discord.ext import commands
from discord.ui import Button, View

from openai import OpenAI
from supabase import create_client, Client

# Import new RAG system
from curriculum_rag import CurriculumRAG

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Discord Bot Configuration
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4.1-mini"

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# RAG Configuration (V4 - New System)
MAX_HISTORY_MESSAGES = 10  # Keep last 10 messages (5 exchanges)
MAX_CONTEXT_TOKENS = 8000  # Safeguard for token limit

# Message Configuration
MAX_DISCORD_MESSAGE_LENGTH = 1500

# Clarification Loop Prevention
MAX_CLARIFICATIONS = 1  # Only allow 1 clarifying question before forcing an answer

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, "Data_Doc_main.txt")

# Mentor IDs
MENTOR_IDS = [
    "<@1389934019030028380>",  # Mekashi
    "<@1352199617877381150>"   # Omkar
]

# Supported Text File Extensions
TEXT_FILE_EXTENSIONS = [
    '.txt', '.json', '.py', '.js', '.ts', '.jsx', '.tsx',
    '.md', '.csv', '.log', '.yaml', '.yml', '.env',
    '.config', '.ini', '.toml', '.xml', '.html', '.css',
    '.sh', '.bash', '.sql', '.java', '.cpp', '.c', '.go'
]

# ============================================================================
# GLOBAL STATE
# ============================================================================

conversation_history: Dict[int, List[Dict[str, str]]] = {}
pending_feedback: Dict[int, Dict[str, int]] = {}

# Track consecutive clarifying responses per thread
clarification_tracker: Dict[int, int] = {}

# Intelligent feedback state tracking (research-based UX)
feedback_pending: Dict[int, bool] = {}        # thread_id -> awaiting user interaction?
last_feedback_shown: Dict[int, float] = {}    # thread_id -> timestamp (30s cooldown)
feedback_completed: Dict[int, bool] = {}      # thread_id -> user gave feedback (Got it/Tag crew)

# New RAG system (V4)
rag = None
openai_client = None
supabase_client: Optional[Client] = None
api_semaphore = asyncio.Semaphore(3)

# ============================================================================
# DISCORD BOT SETUP
# ============================================================================

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)

# ============================================================================
# FEEDBACK SYSTEM - DISCORD UI VIEWS (keeping your existing implementation)
# ============================================================================

class FeedbackView(View):
    """Discord UI View with buttons for initial feedback."""
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

        # Log analytics event
        await log_event('got_it', self.thread_id, interaction.user.id, interaction.message.id)

        # Update intelligent feedback state
        feedback_pending[self.thread_id] = False
        feedback_completed[self.thread_id] = True

        await interaction.response.send_message(
            "Awesome! 🚀 Happy learning!",
            ephemeral=False
        )

        for item in self.children:
            item.disabled = True
        await interaction.message.edit(view=self)

        if interaction.message.id in pending_feedback:
            del pending_feedback[interaction.message.id]
    
    @discord.ui.button(label="🔄 Need more help", style=discord.ButtonStyle.secondary)
    async def need_help_button(self, interaction: discord.Interaction, button: Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This feedback is for the person who asked the question! 😊",
                ephemeral=True
            )
            return

        # Log analytics event
        await log_event('need_help', self.thread_id, interaction.user.id, interaction.message.id)

        # Clear pending state (don't mark completed - they need more help)
        feedback_pending[self.thread_id] = False

        for item in self.children:
            item.disabled = True
        await interaction.message.edit(view=self)

        follow_up_view = FollowUpView(self.user_id, self.thread_id)
        await interaction.response.send_message(
            "No worries, let's figure this out! What would you prefer?",
            view=follow_up_view,
            ephemeral=False
        )

        if interaction.message.id in pending_feedback:
            del pending_feedback[interaction.message.id]


class FollowUpView(View):
    """Discord UI View with buttons for follow-up actions."""
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

        # Log analytics event
        await log_event('continue_here', self.thread_id, interaction.user.id, interaction.message.id)

        # Reset feedback state - allow new feedback after continuation
        feedback_completed[self.thread_id] = False

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

        # Log analytics event
        await log_event('tag_crew', self.thread_id, interaction.user.id, interaction.message.id)

        # Mark feedback completed (escalation = resolved state)
        feedback_pending[self.thread_id] = False
        feedback_completed[self.thread_id] = True

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

# ============================================================================
# RAG SYSTEM FUNCTIONS (V4 - New Implementation)
# ============================================================================

def retrieve_curriculum_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant lectures from curriculum using new RAG system.

    Returns formatted context with module/lecture metadata for LLM.
    """
    if not rag or not rag.index:
        return "[ERROR: RAG system not initialized]"

    try:
        results = rag.search(query, top_k=top_k)

        if not results:
            return "[No relevant curriculum content found]"

        # Format results with metadata
        context_parts = []
        for i, result in enumerate(results, 1):
            lecture_num = result.get('lecture_num', '?')
            lecture_name = result.get('lecture_name', 'Unknown')
            module = result.get('module', '?')
            module_name = result.get('module_name', 'Unknown')
            mentor = result.get('mentor', 'Unknown')
            topics = result.get('topics', [])
            content = result.get('content', '')

            # Format header with metadata
            header = f"[Module {module}: {module_name} | Lecture {lecture_num} | Mentor: {mentor}]"
            header += f"\nTitle: {lecture_name}"
            if topics:
                header += f"\nTopics: {', '.join(topics[:5])}"

            # Truncate content to ~2000 chars to fit in context
            truncated_content = content[:2000] + "..." if len(content) > 2000 else content

            context_parts.append(f"{header}\n\n{truncated_content}")

        return "\n\n" + "="*60 + "\n\n".join(context_parts)

    except Exception as e:
        print(f"[ERROR] RAG search failed: {e}")
        return "[ERROR: Failed to retrieve curriculum context]"

# ============================================================================
# CONVERSATION MANAGEMENT (V4 - Proper OpenAI Format)
# ============================================================================

def get_thread_history(thread_id: int) -> List[Dict[str, str]]:
    """
    Retrieve conversation history for thread in OpenAI message format.

    Returns list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    return conversation_history.get(thread_id, [])


def add_to_thread_history(thread_id: int, role: str, content: str):
    """
    Add message to thread history.

    Args:
        thread_id: Discord thread ID
        role: "user" or "assistant"
        content: Full message content (NO truncation)
    """
    if thread_id not in conversation_history:
        conversation_history[thread_id] = []

    conversation_history[thread_id].append({
        "role": role,
        "content": content
    })

    # Keep only last 10 messages (5 exchanges)
    if len(conversation_history[thread_id]) > MAX_HISTORY_MESSAGES:
        conversation_history[thread_id] = conversation_history[thread_id][-MAX_HISTORY_MESSAGES:]


def estimate_token_count(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters"""
    return len(text) // 4


def build_conversation_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
    current_query: str,
    image_url: Optional[str] = None
) -> List[Dict]:
    """
    Build proper OpenAI messages array from conversation history.

    Format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "current question"}
    ]
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (already in correct format)
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current query
    if image_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": current_query},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        })
    else:
        messages.append({"role": "user", "content": current_query})

    return messages


def get_clarification_count(thread_id: int) -> int:
    """Get number of consecutive clarifying questions in this thread."""
    return clarification_tracker.get(thread_id, 0)


def increment_clarification_count(thread_id: int):
    """Increment clarification counter."""
    clarification_tracker[thread_id] = clarification_tracker.get(thread_id, 0) + 1


def reset_clarification_count(thread_id: int):
    """Reset clarification counter (call when providing solution)."""
    clarification_tracker[thread_id] = 0


def has_student_provided_clarification(history: List[Dict[str, str]]) -> bool:
    """
    Check if student has responded to a clarifying question.
    Returns True if the last exchange was: bot asked → student answered
    """
    if len(history) < 2:
        return False
    
    last_two = history[-2:]
    
    # Check if pattern is: assistant (clarifying) → user (response)
    if last_two[0]["role"] == "assistant" and last_two[1]["role"] == "user":
        # Check if bot's message was a question
        bot_msg = last_two[0]["content"].lower()
        if "?" in bot_msg or any(word in bot_msg for word in ["what", "which", "how", "can you", "did you"]):
            return True
    
    return False

# ============================================================================
# OPENAI INTEGRATION (IMPROVED WITH LOOP PREVENTION)
# ============================================================================

async def generate_response(
    query: str,
    context: str,
    history: List[Dict[str, str]],
    thread_id: int,
    image_url: Optional[str] = None,
    file_context: Optional[str] = None
) -> str:
    """
    Generate response using proper OpenAI conversation format.

    V4 Changes:
    - Passes full conversation history as messages array
    - No truncation of messages
    - Context included in system prompt (not repeated in messages)
    """
    async with api_semaphore:
        try:
            # Get clarification state
            clarify_count = get_clarification_count(thread_id)
            student_clarified = has_student_provided_clarification(history)

            # Determine response mode
            if clarify_count >= MAX_CLARIFICATIONS or student_clarified:
                response_mode = "ANSWER"
            else:
                response_mode = "NORMAL"

            # Build system prompt with curriculum context
            if response_mode == "ANSWER":
                system_prompt = f"""You are Sage, technical mentor for 100xEngineers AI Cohort 6.

Students use you instead of ChatGPT because you know THEIR curriculum.
Your job: Help them learn. Not coddle them.
You're not their friend. You're their senior dev who respects their time enough to tell the truth.

{f"UPLOADED FILE CONTEXT:\\n{file_context}\\n\\n" if file_context else ""}CURRICULUM CONTEXT (with module/lecture metadata):
{context}

CRITICAL: You've already asked clarifying questions. Now you MUST provide a concrete detailed contextual answer based on available information.

REFERENCING CURRICULUM:
- ALWAYS reference the exact Module/Lecture from context headers (e.g., [Module 1: Diffusion | Lecture 5])
- ONLY cite lectures that appear in the CURRICULUM CONTEXT above
- DO NOT invent lecture numbers, module names, or week numbers

COMMUNICATION:
- Brutally honest. If they're overthinking, say it (softly).
- Concise. Sacrifice grammar for clarity.
- No fluff: skip "great question!", restating, disclaimers.
- Call out mistakes: "You're wrong because X"
- Use contractions: "you're" not "you are", "nah" not "I don't think so"

FORMATTING (Discord):
- Short paragraphs (2-4 lines max)
- Code blocks for code
- Bullets for lists
- Bold for key terms"""

            else:
                # NORMAL MODE
                system_prompt = f"""You are Sage, technical mentor for 100xEngineers AI Cohort 6.

Students use you instead of ChatGPT because you know THEIR curriculum.
Your job: Help them learn. Not coddle them.
You're not their friend. You're their senior dev who respects their time enough to tell the truth.

{f"UPLOADED FILE CONTEXT:\\n{file_context}\\n\\n" if file_context else ""}CURRICULUM CONTEXT (with module/lecture metadata):
{context}

Response strategy:
1. If query specific with enough detail → answer directly
2. If query vague (like "help with X") → ask ONE clarifying question MAX

REFERENCING CURRICULUM:
- ALWAYS reference the exact Module/Lecture from context headers (e.g., [Module 1: Diffusion | Lecture 5])
- ONLY cite lectures that appear in the CURRICULUM CONTEXT above
- DO NOT invent lecture numbers, module names, or week numbers

COMMUNICATION:
- Brutally honest. If they're overthinking, say it (softly).
- Concise. Sacrifice grammar for clarity.
- No fluff: skip "great question!", restating, disclaimers.
- Call out mistakes: "You're wrong because X"
- Use contractions: "you're" not "you are", "nah" not "I don't think so"

FORMATTING (Discord):
- Short paragraphs (2-4 lines max)
- Code blocks for code
- Bullets for lists
- Bold for key terms

Keep it brief. One clarifying question MAX, then answer what you can."""

            # Build messages array (proper OpenAI format)
            messages = build_conversation_messages(
                system_prompt=system_prompt,
                history=history,
                current_query=query,
                image_url=image_url
            )

            # Call OpenAI with full conversation context
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=600,  # Increased from 300 - allow fuller answers
                temperature=0.6,
                presence_penalty=0.4,
                frequency_penalty=0.2
            )

            response_text = response.choices[0].message.content

            # Track clarification state
            is_clarifying = is_asking_clarification(response_text)

            if is_clarifying and response_mode == "NORMAL":
                increment_clarification_count(thread_id)
            else:
                reset_clarification_count(thread_id)

            return response_text

        except Exception as e:
            print(f"[ERROR] Response generation failed: {e}")
            return "I'm having trouble generating a response right now. Please try again in a moment."


def is_asking_clarification(response: str) -> bool:
    """
    Detect if response is primarily asking clarifying questions.
    
    Returns True if:
    - Multiple questions (2+) without solution indicators
    - Ends with question
    - Contains clarifying phrases
    """
    response_lower = response.lower()
    
    # Count questions
    question_marks = response.count("?")
    
    # Check for solution indicators
    solution_indicators = [
        "here's",
        "try this",
        "the issue is",
        "you need to",
        "add this",
        "check line",
        "lecture",
        "covered in",
        "week"
    ]
    
    has_solution = any(indicator in response_lower for indicator in solution_indicators)
    
    # If 2+ questions and no solution → clarifying
    if question_marks >= 2 and not has_solution:
        return True
    
    # If ends with question and no solution → clarifying
    if response.strip().endswith("?") and not has_solution:
        return True
    
    return False


def should_show_feedback(user_message: str, bot_response: str, thread_id: int) -> bool:
    """
    Intelligent detection: Show feedback buttons ONLY when bot provides substantial solution.

    Based on research:
    - Prevent survey fatigue (target only meaningful interactions)
    - Show after "successful problem resolution" not every response
    - One feedback prompt per thread until user interacts

    Returns True only when:
    - Bot provides substantial answer (150+ chars, 3+ solution indicators)
    - NOT a clarifying question
    - NOT a meta query about bot capabilities
    - No pending feedback awaiting response
    - Cooldown period passed (30 seconds)
    - User hasn't already completed feedback
    """

    # 1. Meta query detection (simple heuristic)
    meta_patterns = [
        'what can you do',
        'how do you work',
        'help',
        'who are you',
        'what are your capabilities',
        'what is this bot',
        'how to use you'
    ]
    user_lower = user_message.lower()
    if any(pattern in user_lower for pattern in meta_patterns):
        return False

    # 2. Feedback already pending (user hasn't clicked buttons yet)
    if feedback_pending.get(thread_id, False):
        return False

    # 3. Cooldown check (30 seconds)
    if thread_id in last_feedback_shown:
        elapsed = time.time() - last_feedback_shown[thread_id]
        if elapsed < 30:
            return False

    # 4. Feedback already completed (until "Continue here" resets)
    if feedback_completed.get(thread_id, False):
        return False

    # 5. Clarifying question detection
    response_lower = bot_response.lower()
    question_count = bot_response.count('?')

    # Override: 2+ questions = definitely clarifying
    if question_count >= 2:
        return False

    # Override: clarifying phrases detected
    clarifying_phrases = [
        'could you clarify',
        'need more information',
        'which one',
        'can you provide',
        'what specifically',
        'could you please share'
    ]
    if any(phrase in response_lower for phrase in clarifying_phrases):
        return False

    # 6. SIMPLIFIED SOLUTION DETECTION
    # Show buttons if response meets ANY of these criteria (OR logic):

    # Criterion A: Has code block + reasonable length
    if '```' in bot_response and len(bot_response) >= 50:
        feedback_pending[thread_id] = True
        last_feedback_shown[thread_id] = time.time()
        return True

    # Criterion B: References curriculum + reasonable length
    if ('lecture' in response_lower or 'module' in response_lower) and len(bot_response) >= 80:
        feedback_pending[thread_id] = True
        last_feedback_shown[thread_id] = time.time()
        return True

    # Criterion C: Has instructional language + reasonable length
    instructional = ["here's how", "try this", "you can", "you need to", "you should", "step 1", "first,"]
    if any(phrase in response_lower for phrase in instructional) and len(bot_response) >= 80:
        feedback_pending[thread_id] = True
        last_feedback_shown[thread_id] = time.time()
        return True

    # Criterion D: Multiple solution keywords + decent length
    solution_keywords = ["the fix", "the issue", "the problem", "to solve", "the answer", "check", "add", "change", "update"]
    keyword_count = sum(1 for word in solution_keywords if word in response_lower)
    if keyword_count >= 2 and len(bot_response) >= 80:
        feedback_pending[thread_id] = True
        last_feedback_shown[thread_id] = time.time()
        return True

    # Criterion E: Long detailed response (likely comprehensive answer)
    if len(bot_response) >= 200:
        feedback_pending[thread_id] = True
        last_feedback_shown[thread_id] = time.time()
        return True

    return False

# ============================================================================
# FILE ATTACHMENT HANDLING
# ============================================================================

async def handle_text_file_attachment(attachment: discord.Attachment) -> Optional[str]:
    """
    Download and process text file. Auto-truncates to 20KB if larger.

    Returns: Formatted file content or None if unsupported/error
    """
    # Check extension
    if not any(attachment.filename.lower().endswith(ext) for ext in TEXT_FILE_EXTENSIONS):
        return None

    # Download file
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                content = await resp.text(encoding='utf-8', errors='ignore')

        if len(content.strip()) == 0:
            return None

        # Truncate if >20KB
        max_chars = 20480
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[... file truncated, showing first 20KB only]"

        return f"UPLOADED FILE: {attachment.filename}\n\n{content}"

    except Exception as e:
        print(f"File download error: {e}")
        return None

# ============================================================================
# MESSAGE HANDLING (keeping your implementation)
# ============================================================================

def split_long_message(content: str) -> List[str]:
    """Split long messages for Discord."""
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

# ============================================================================
# ANALYTICS LOGGING
# ============================================================================

async def log_event(event_type: str, thread_id: int, user_id: int, message_id: Optional[int] = None):
    """Log analytics event to Supabase."""
    if not supabase_client:
        return  # Skip if Supabase not configured

    try:
        supabase_client.table('analytics_events').insert({
            'event_type': event_type,
            'thread_id': thread_id,
            'user_id': user_id,
            'message_id': message_id
        }).execute()
    except Exception as e:
        print(f"Analytics logging error: {e}")

# ============================================================================
# DISCORD EVENT HANDLERS
# ============================================================================

@bot.event
async def on_ready():
    """Initialize bot components."""
    global openai_client, supabase_client

    print(f'Bot logged in as {bot.user.name} (ID: {bot.user.id})')
    print('----------------------------------------')

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print('✓ OpenAI client initialized')

    # Initialize Supabase client
    if SUPABASE_URL and SUPABASE_KEY:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print('✓ Supabase analytics client initialized')
    else:
        print('⚠ Supabase credentials not found - analytics disabled')

    # Initialize new RAG system (V4)
    bot.loop.create_task(initialize_rag_system())

    print('✓ RAG system initialization started in background')
    print('----------------------------------------')
    print('Bot is ready! V4 with improved RAG + conversation history')


async def initialize_rag_system():
    """Initialize new RAG system with OpenAI embeddings."""
    global rag

    try:
        print('[*] Loading RAG system...')

        # Initialize RAG (loads pre-built index from disk)
        rag = CurriculumRAG()

        if rag.index:
            print(f'[OK] RAG system ready! {len(rag.metadata)} lectures indexed')
        else:
            print('[WARN] RAG index not found. Run setup_embeddings.py first.')

    except Exception as e:
        print(f'[ERROR] Failed to initialize RAG system: {e}')
        print('Bot will continue without RAG functionality')



@bot.event
async def on_message(message):
    """Main message handler with loop prevention."""
    if message.author == bot.user:
        return
    
    if message.channel.type not in [discord.ChannelType.public_thread, discord.ChannelType.private_thread]:
        return
    
    if bot.user not in message.mentions:
        return
    
    thread_id = message.channel.id
    
    query = message.content
    for mention in message.mentions:
        query = query.replace(f'<@{mention.id}>', '').strip()
    
    if not query and not message.attachments:
        await message.reply(
            "Hi! I'm Sage. Share your question and I'll help you out!"
        )
        return
    
    async with message.channel.typing():
        try:
            image_url = None
            file_context = None
            if message.attachments:
                for attachment in message.attachments:
                    # Check images first (preserve existing behavior)
                    if any(attachment.filename.lower().endswith(ext)
                           for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        image_url = attachment.url
                        break

                    # Check text files
                    file_context = await handle_text_file_attachment(attachment)
                    if file_context:
                        break

            # V4: Re-retrieve curriculum context on EVERY message (fresh context)
            context = retrieve_curriculum_context(query, top_k=3)
            history = get_thread_history(thread_id)

            # Generate response with full conversation history
            response = await generate_response(query, context, history, thread_id, image_url, file_context)
            
            message_chunks = split_long_message(response)
            
            first_message = await message.reply(message_chunks[0])
            
            for chunk in message_chunks[1:]:
                await message.channel.send(chunk)
                await asyncio.sleep(0.5)
            
            # Update history
            add_to_thread_history(thread_id, "user", query)
            add_to_thread_history(thread_id, "assistant", response)

            # Log analytics event
            await log_event('query', thread_id, message.author.id, first_message.id)

            # Intelligent feedback detection (research-based UX)
            if should_show_feedback(query, response, thread_id):
                await asyncio.sleep(1.5)

                feedback_view = FeedbackView(user_id=message.author.id, thread_id=thread_id)
                feedback_message = await message.channel.send(
                    "🎯 Does this clear things up?",
                    view=feedback_view
                )

                pending_feedback[feedback_message.id] = {
                    "thread_id": thread_id,
                    "user_id": message.author.id
                }
            
        except Exception as e:
            print(f"Error processing message: {e}")
            await message.reply(
                "I encountered an error. Try rephrasing or tag a mentor if needed."
            )

# ============================================================================
# BOT STARTUP
# ============================================================================

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not found")
        exit(1)
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found")
        exit(1)
    
    print("Starting improved Discord bot...")
    bot.run(DISCORD_BOT_TOKEN)