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

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Discord Bot Configuration
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4.1-mini"

# RAG Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MAX_HISTORY_MESSAGES = 6  # Reduced from 10 to prevent context bloat

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

# ============================================================================
# GLOBAL STATE
# ============================================================================

conversation_history: Dict[int, List[Dict[str, str]]] = {}
pending_feedback: Dict[int, Dict[str, int]] = {}

# NEW: Track consecutive clarifying responses per thread
clarification_tracker: Dict[int, int] = {}

vector_store = None
openai_client = None
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
    """Load and preprocess curriculum text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def create_vector_store(text: str) -> FAISS:
    """Create FAISS vector store from text."""
    print("Creating vector store...")
    doc = Document(page_content=text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents([doc])
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    if not chunks:
        raise ValueError("No valid document chunks created.")
    
    print(f"Created {len(chunks)} document chunks")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully")
    return vector_store


def retrieve_relevant_context(query: str, k: int = 3) -> str:
    """Retrieve most relevant chunks for query."""
    if not vector_store:
        return ""
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context

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
    image_url: Optional[str] = None
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
                system_prompt = f"""You're Sage - technical mentor for 100xEngineers AI Cohort 6.

CURRICULUM CONTEXT:
{context}

RECENT CONVERSATION:
{format_history_for_prompt(history)}

CRITICAL: You've already asked clarifying questions. Now you MUST provide a concrete answer based on available information.

Rules for this response:
- Provide the best answer you can with the information available
- Reference specific lectures from the context
- Be direct and helpful
- NO MORE CLARIFYING QUESTIONS - give your best guidance now
- If truly stumped, suggest they tag mentors

Student's question: {query}"""
            
            else:
                # NORMAL MODE
                system_prompt = f"""You're Sage - technical mentor for 100xEngineers AI Cohort 6.

CURRICULUM CONTEXT:
{context}

RECENT CONVERSATION:
{format_history_for_prompt(history)}

Response strategy:
1. If query is specific with enough detail → answer directly
2. If query is vague (like "help with X") → ask ONE clarifying question MAX
3. Reference lectures naturally: "Week 8 covered this"

Examples:
Student: "I'm getting errors with ControlNet"
You: "What's the error? Are you using Lecture 5's workflow or custom?"

Student: "Error: 'API key not found' in FastAPI"
You: "Week 2 FastAPI lecture - add your key to .env file. Check line 23-ish in your code."

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
                max_tokens=600,  # Increased for fuller answers
                temperature=0.8,  # Slightly more creative
                presence_penalty=0.6,
                frequency_penalty=0.3
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
# DISCORD EVENT HANDLERS
# ============================================================================

@bot.event
async def on_ready():
    """Initialize bot components."""
    global openai_client, vector_store
    
    print(f'Bot logged in as {bot.user.name} (ID: {bot.user.id})')
    print('----------------------------------------')
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print('✓ OpenAI client initialized')
    
    print('Loading curriculum data...')
    text = load_and_preprocess_text(DATA_FILE_PATH)
    vector_store = create_vector_store(text)
    print('✓ RAG system initialized')
    
    print('----------------------------------------')
    print('Bot is ready! Clarification loop prevention: ACTIVE')


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
            if message.attachments:
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) 
                           for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        image_url = attachment.url
                        break
            
            context = retrieve_relevant_context(query, k=3)
            history = get_thread_history(thread_id)
            
            # PASS thread_id to track clarification state
            response = await generate_response(query, context, history, thread_id, image_url)
            
            message_chunks = split_long_message(response)
            
            first_message = await message.reply(message_chunks[0])
            
            for chunk in message_chunks[1:]:
                await message.channel.send(chunk)
                await asyncio.sleep(0.5)
            
            # Update history
            add_to_thread_history(thread_id, "user", query)
            add_to_thread_history(thread_id, "assistant", response)
            
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