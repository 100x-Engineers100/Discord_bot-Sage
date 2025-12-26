# Discord Support Bot - Sage

A RAG-powered Discord bot that serves as a technical learning assistant for students in forum settings. Built for **100xEngineers AI Cohort 6**, Sage helps students with technical queries using Retrieval-Augmented Generation (RAG) technology.

## 🤖 Features

- **RAG-Powered Responses**: Uses FAISS vector store with HuggingFace embeddings to retrieve relevant curriculum content
- **Per-Thread Conversation History**: Maintains context of the last 5 exchanges per forum thread
- **Image Analysis**: Supports image analysis via OpenAI Vision API
- **Forum Thread Detection**: Automatically responds when mentioned in Discord forum threads
- **Smart Message Splitting**: Automatically splits long responses to comply with Discord's message length limits
- **Curriculum-Grounded Answers**: References specific lectures, modules, and weeks from the curriculum

## 🏗️ Architecture

### Components

1. **RAG System**
   - Text preprocessing and chunking (1500 chars with 150 char overlap)
   - FAISS vector store for fast similarity search
   - HuggingFace `all-MiniLM-L6-v2` embeddings (384-dimensional)

2. **LLM Integration**
   - OpenAI GPT-4.1-mini for response generation
   - Custom system prompt tailored for technical assistance
   - Rate limiting via semaphore (max 3 concurrent API calls)

3. **Discord Integration**
   - Forum thread detection
   - Bot mention handling
   - Conversation history management per thread

## 📋 Prerequisites

- Python 3.12 or higher
- Discord Bot Token
- OpenAI API Key
- Curriculum data file (`Data_Doc_main.txt`)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd discord-support-bot
```

### 2. Create Virtual Environment

```bash
python -m venv bot
```

### 3. Activate Virtual Environment

**Windows:**
```bash
bot\Scripts\activate
```

**Linux/Mac:**
```bash
source bot/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Environment Configuration

Create a `.env` file in the root directory:

```env
DISCORD_BOT_TOKEN=your_discord_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 6. Prepare Curriculum Data

Ensure `Data_Doc_main.txt` is present in the root directory with your curriculum content.

## 🔧 Configuration

The bot can be configured by modifying constants in `bot.py`:

- `CHUNK_SIZE`: Size of text chunks for embedding (default: 1500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `MAX_HISTORY_MESSAGES`: Number of conversation exchanges to remember (default: 5)
- `MAX_DISCORD_MESSAGE_LENGTH`: Maximum message length before splitting (default: 1900)
- `OPENAI_MODEL`: OpenAI model to use (default: "gpt-4.1-mini")

## 💻 Usage

### Running Locally

```bash
python bot.py
```

The bot will:
1. Initialize the OpenAI client
2. Load and preprocess the curriculum data
3. Create the FAISS vector store
4. Connect to Discord
5. Start listening for mentions in forum threads

### Using Docker

Build the Docker image:

```bash
docker build -t discord-support-bot .
```

Run the container:

```bash
docker run --env-file .env discord-support-bot
```

### Deployment (Heroku)

The project includes a `Procfile` for Heroku deployment:

```
worker: python bot.py
```

Ensure your Heroku app has the following environment variables set:
- `DISCORD_BOT_TOKEN`
- `OPENAI_API_KEY`

## 📖 How It Works

1. **Message Detection**: Bot listens for messages in Discord forum threads where it's mentioned
2. **Query Processing**: Extracts the user's question and checks for image attachments
3. **Context Retrieval**: Uses RAG to find the 3 most relevant curriculum chunks
4. **History Retrieval**: Loads the last 5 conversation exchanges for the thread
5. **Response Generation**: Sends query, context, and history to OpenAI GPT-4.1-mini
6. **Message Delivery**: Splits long responses and sends them to Discord
7. **History Update**: Stores the exchange in thread conversation history

## 📁 Project Structure

```
discord-support-bot/
├── bot.py                    # Main bot implementation
├── main.py                   # Legacy/alternative entry point (commented)
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── Procfile                 # Heroku deployment config
├── .env                     # Environment variables (create this)
├── Data_Doc_main.txt       # Curriculum data file
├── curriculum_comprehensive_index.txt  # Curriculum index
└── README.md               # This file
```

## 🔑 Key Dependencies

- `discord.py`: Discord API wrapper
- `openai`: OpenAI API client
- `langchain`: RAG framework
- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Embedding models
- `torch`: Deep learning framework (for embeddings)

## 🎯 Bot Behavior

- **Responds only when mentioned** in forum threads
- **Maintains conversation context** per thread (last 5 exchanges)
- **References curriculum** with specific lecture/module/week numbers
- **Handles images** via OpenAI Vision API
- **Splits long messages** automatically to comply with Discord limits
- **Uses emojis sparingly** (1-2 per response max)
- **Provides brief, structured answers** with step-by-step explanations

## 🛠️ Troubleshooting

### Bot not responding
- Check that the bot is mentioned in the message
- Verify the message is in a forum thread
- Ensure `DISCORD_BOT_TOKEN` is set correctly

### RAG not working
- Verify `Data_Doc_main.txt` exists and contains data
- Check that the file is readable and properly formatted
- Review console logs for embedding/vector store errors

### API errors
- Verify `OPENAI_API_KEY` is valid and has credits
- Check rate limits (bot uses semaphore to limit concurrent calls)
- Review OpenAI API status

## 📝 Notes

- The bot uses GPU if available for faster embedding generation
- Conversation history is stored in memory (resets on bot restart)
- Text preprocessing removes non-ASCII characters and converts to lowercase
- The bot ignores its own messages to prevent infinite loops

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Add your license here]

## 👤 Author

Built for 100xEngineers AI Cohort 6

---

**Note**: Make sure to keep your `.env` file secure and never commit it to version control. Add `.env` to your `.gitignore` file.

