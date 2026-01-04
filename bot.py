"""
Discord Support Bot V3 - Technical Learning Assistant (IMPROVED)
====================================================
A RAG-powered Discord bot that helps students with technical queries in a forum setting.

IMPROVEMENTS IN V3:
- Fixed clarifying question loop bug
- Better conversation flow management
- Smarter history tracking
- Adaptive response strategy
"""

import os
import re
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dotenv import load_dotenv

import discord
from discord.ext import commands
from discord.ui import Button, View

import torch
from openai import OpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from supabase import create_client, Client

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

# RAG Configuration
CHUNK_SIZE = 800  # Reduced from 1500 for more focused chunks
CHUNK_OVERLAP = 100  # Reduced from 150
MAX_HISTORY_MESSAGES = 8  # Reduced from 10 to prevent context bloat

# Message Configuration
MAX_DISCORD_MESSAGE_LENGTH = 1900

# Clarification Loop Prevention
MAX_CLARIFICATIONS = 1 # Only allow 1 clarifying question before forcing an answer

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

# NEW: Track consecutive clarifying responses per thread
clarification_tracker: Dict[int, int] = {}

vector_store = None
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
# RAG SYSTEM FUNCTIONS (keeping your existing implementation)
# ============================================================================

def load_and_preprocess_text(file_path: str) -> str:
    """Load curriculum with MINIMAL preprocessing - preserve structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Only basic cleanup - preserve punctuation, case, structure
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII only
    text = re.sub(r'\s+', ' ', text)            # Normalize whitespace
    return text.strip()  # NO lowercasing, NO punctuation removal


def extract_metadata_from_chunk(chunk_text: str, chunk_index: int) -> dict:
    """Extract module/lecture metadata from chunk content."""
    metadata = {
        'chunk_index': chunk_index,
        'module': None,
        'module_name': None,
        'lecture': None,
        'lecture_title': None
    }

    # Extract Module (e.g., "Module1: Diffusion" or "Module 1")
    module_match = re.search(r'Module\s*(\d+)[\s:]*([^\n]+)?', chunk_text, re.IGNORECASE)
    if module_match:
        metadata['module'] = module_match.group(1)
        if module_match.group(2):
            # Clean module name - remove extra colons, parentheses info
            module_name = module_match.group(2).strip()
            # Remove anything in parentheses
            module_name = re.sub(r'\s*\([^)]*\)', '', module_name)
            metadata['module_name'] = module_name.strip()

    # Extract Lecture number and title (e.g., "1. Orientation Session: Evolution of GenAI")
    # Pattern: number + optional period + title
    lecture_match = re.search(r'(?:Lecture\s*)?(\d+)\.?\s+([^\n]+?)(?:\n|$)', chunk_text)
    if lecture_match:
        metadata['lecture'] = lecture_match.group(1)
        # Clean lecture title - remove colons and extra whitespace
        lecture_title = lecture_match.group(2).strip()
        # If there's a colon, take everything (e.g., "Orientation Session: Evolution of GenAI")
        metadata['lecture_title'] = lecture_title

    return metadata


def create_vector_store(text: str) -> FAISS:
    """Create FAISS vector store with metadata-enriched chunks."""
    print("Creating vector store with metadata...")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Added ". " for better breaks
    )

    # Split text into chunks FIRST
    initial_chunks = text_splitter.split_text(text)

    # Enrich each chunk with metadata
    documents = []
    for i, chunk_text in enumerate(initial_chunks):
        if not chunk_text.strip():
            continue

        # Extract metadata from chunk content
        metadata = extract_metadata_from_chunk(chunk_text, i)

        # Create Document with metadata
        doc = Document(
            page_content=chunk_text,
            metadata=metadata
        )
        documents.append(doc)

    if not documents:
        raise ValueError("No valid document chunks created.")

    print(f"Created {len(documents)} metadata-enriched chunks")

    # Create embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    # Build FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Vector store created successfully with metadata")
    return vector_store


def retrieve_relevant_context(query: str, k: int = 5) -> str:
    """Retrieve relevant chunks WITH metadata headers."""
    if not vector_store:
        return ""

    # Get more chunks now that they're smaller (800 chars vs 1500)
    docs = vector_store.similarity_search(query, k=k)

    # Format chunks with metadata headers
    formatted_chunks = []
    for doc in docs:
        meta = doc.metadata

        # Build header from metadata
        header_parts = []
        if meta.get('module'):
            module_str = f"Module {meta['module']}"
            if meta.get('module_name'):
                module_str += f": {meta['module_name']}"
            header_parts.append(module_str)

        if meta.get('lecture'):
            lecture_str = f"Lecture {meta['lecture']}"
            if meta.get('lecture_title'):
                lecture_str += f" - {meta['lecture_title']}"
            header_parts.append(lecture_str)

        # Format: [Module 1: Diffusion | Lecture 5 - ControlNet]
        header = " | ".join(header_parts) if header_parts else "General Curriculum"

        formatted_chunks.append(f"[{header}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)

# ============================================================================
# CONVERSATION MANAGEMENT (IMPROVED)
# ============================================================================

def get_thread_history(thread_id: int) -> List[Dict[str, str]]:
    """Retrieve conversation history for thread."""
    return conversation_history.get(thread_id, [])


def add_to_thread_history(thread_id: int, role: str, content: str):
    """Add message to thread history."""
    if thread_id not in conversation_history:
        conversation_history[thread_id] = []
    
    conversation_history[thread_id].append({
        "role": role,
        "content": content
    })
    
    # Keep only last N messages
    conversation_history[thread_id] = conversation_history[thread_id][-MAX_HISTORY_MESSAGES:]


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """Format history for LLM context."""
    if not history:
        return "No previous conversation in this thread."
    
    formatted = []
    for msg in history:
        role_label = "Student" if msg["role"] == "user" else "You (Sage)"
        # Show full recent context for better understanding
        formatted.append(f"{role_label}: {msg['content'][:500]}")  # Increased from 200
    
    return "\n".join(formatted)


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
    thread_id: int,  # NEW: Need thread_id for tracking
    image_url: Optional[str] = None,
    file_context: Optional[str] = None
) -> str:
    """
    Generate response with clarification loop prevention.
    """
    async with api_semaphore:
        try:
            # Get clarification state for this thread
            clarify_count = get_clarification_count(thread_id)
            student_clarified = has_student_provided_clarification(history)
            
            # Determine response mode
            if clarify_count >= MAX_CLARIFICATIONS or student_clarified:
                # FORCE ANSWER MODE - no more questions allowed
                response_mode = "ANSWER"
            else:
                # NORMAL MODE - can ask clarifying questions if needed
                response_mode = "NORMAL"
            
            # Build adaptive system prompt based on mode
            if response_mode == "ANSWER":
                system_prompt = f"""You are Sage, technical mentor for 100xEngineers AI Cohort 6.

Students use you instead of ChatGPT because you know THEIR curriculum.
Your job: Help them learn. Not coddle them.
You're not their friend. You're their senior dev who respects their time enough to tell the truth.

{f"UPLOADED FILE CONTEXT:\\n{file_context}\\n\\n" if file_context else ""}CURRICULUM CONTEXT (with module/lecture metadata):
{context}

RECENT CONVERSATION:
{format_history_for_prompt(history)}

CRITICAL: You've already asked clarifying questions. Now you MUST provide a concrete answer based on available information.

REFERENCING CURRICULUM:
- ALWAYS reference the exact Module/Lecture from context headers (e.g., [Module 1: Diffusion | Lecture 5])
- ONLY cite lectures that appear in the CURRICULUM CONTEXT above
- DO NOT invent lecture numbers, module names, or week numbers
- If concept NOT in context, say "not in the curriculum I have access to - tag @mekashi @omkar"

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

EXAMPLES:
Good: "Module 1, Lecture 5 on ControlNet covered this"
Bad: "Week 8's segmentation lecture covered this" (inventing)

Good: "Bounding box. Rectangle coords around objects. Check Module 1, Lecture 3."
Bad: "Great question! A bounding box is a really important concept..."

Student's question: {query}"""
            
            else:
                # NORMAL MODE
                system_prompt = f"""You are Sage, technical mentor for 100xEngineers AI Cohort 6.

Students use you instead of ChatGPT because you know THEIR curriculum.
Your job: Help them learn. Not coddle them.
You're not their friend. You're their senior dev who respects their time enough to tell the truth.

{f"UPLOADED FILE CONTEXT:\\n{file_context}\\n\\n" if file_context else ""}CURRICULUM CONTEXT (with module/lecture metadata):
{context}

RECENT CONVERSATION:
{format_history_for_prompt(history)}

Response strategy:
1. If query specific with enough detail → answer directly
2. If query vague (like "help with X") → ask ONE clarifying question MAX

REFERENCING CURRICULUM:
- ALWAYS reference the exact Module/Lecture from context headers (e.g., [Module 1: Diffusion | Lecture 5])
- ONLY cite lectures that appear in the CURRICULUM CONTEXT above
- DO NOT invent lecture numbers, module names, or week numbers
- If concept NOT in context, say "not in the curriculum I have access to - tag @mekashi @omkar"

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

EXAMPLES:
Good: "What's the error? Using Module 1, Lecture 5's workflow or custom?"
Bad: "I'm getting errors with ControlNet" (no clarification)

Good: "Module 2 FastAPI lecture - add your key to .env file. Check line 23-ish."
Bad: "Week 2 FastAPI lecture..." (inventing week numbers)

Good: "Bounding box. Rectangle coords. Module 1, Lecture 3 covered this."
Bad: "Great question! A bounding box is a really important concept..."

Keep it brief. One clarifying question MAX, then answer what you can.

Student's question: {query}"""

            # Prepare messages
            messages = [{"role": "system", "content": system_prompt}]
            
            if image_url:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                messages.append({"role": "user", "content": query})
            
            # Call OpenAI
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=250,  # Reduced from 400 - force concise responses
                temperature=0.6,  # Reduced from 0.7 - more grounded, less creative
                presence_penalty=0.4,  # Reduced - allow some repetition for clarity
                frequency_penalty=0.2  # Reduced
            )
            
            response_text = response.choices[0].message.content
            
            # Track if this was a clarifying question or answer
            is_clarifying = is_asking_clarification(response_text)
            
            if is_clarifying and response_mode == "NORMAL":
                increment_clarification_count(thread_id)
            else:
                reset_clarification_count(thread_id)  # Reset on providing answer
            
            return response_text
            
        except Exception as e:
            print(f"Error generating response: {e}")
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


def is_providing_solution(response: str) -> bool:
    """
    Detect if response provides a solution (show feedback buttons).
    """
    response_lower = response.lower()
    
    question_count = response.count("?")
    
    # Solution indicators
    solution_indicators = [
        "here's how",
        "try this",
        "the fix",
        "the issue",
        "the problem is",
        "you need to",
        "you should",
        "add this",
        "check line",
        "in lecture",
        "lecture",
        "module",
        "week",
        "covered this",
        "covered in"
    ]
    
    solution_count = sum(1 for indicator in solution_indicators if indicator in response_lower)
    
    # Strong solution = show buttons
    if solution_count >= 2:
        return True
    
    # Multiple questions without solution = don't show buttons
    if question_count >= 2 and solution_count == 0:
        return False
    
    # Default: show buttons if not purely clarifying
    return solution_count >= 1 or question_count <= 1

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

    # Start vector store initialization in background (non-blocking)
    bot.loop.create_task(initialize_vector_store())

    print('✓ Vector store initialization started in background')
    print('----------------------------------------')
    print('Bot is ready! Clarification loop prevention: ACTIVE')


async def initialize_vector_store():
    """Background task to initialize vector store without blocking heartbeat."""
    global vector_store
    
    try:
        print('Loading curriculum data...')
        text = load_and_preprocess_text(DATA_FILE_PATH)
        
        print('Creating vector store (this may take a minute)...')
        # Run CPU-intensive task in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(None, create_vector_store, text)
        
        print('✓ RAG system fully initialized and ready!')
    except Exception as e:
        print(f'❌ Error initializing vector store: {e}')
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

            context = retrieve_relevant_context(query, k=3)
            history = get_thread_history(thread_id)

            # PASS thread_id to track clarification state
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

            # Show feedback buttons only if providing solution
            if is_providing_solution(response):
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