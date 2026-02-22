"""
Generate 1000+ synthetic student queries for RAG evaluation.
- Uses real query examples from analytics CSV as style anchors
- Generates per-topic batches via OpenAI
- Deduplicates by normalized string overlap
- Output: scripts/synthetic_queries.json
"""

import json
import os
import sys
import csv
import time
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'analytics_events_rows.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'synthetic_queries.json')

# ---------------------------------------------------------------------------
# Real query examples extracted from CSV metadata (style anchors)
# ---------------------------------------------------------------------------
REAL_QUERIES = [
    "help me to setup teh pyhton virtual environment for my project, give me code",
    "help me steup teh github repo for my project that is ther ein my local ssyte.",
    "can u help me understand the OPT framework",
    "tell me where are the lecture recordings?",
    "What is covered in Module 3: AI AGENTS?",
    "Where can I give feedback for yesterday's session?",
    "help in pushing code to github",
    "how can i get the secret and anon key in supabase databse?",
    "where did we learn the database building?",
    "How to test my Apis",
    "getting this error how do i fix it?",
    "hoe to fic this error",
    "This is not working",
    "how do i fix this",
    "Hey WTF is this course, i really am not understanding anything",
    "Do you have image analysis?",
    "tell me the content in requirements.txt",
]

REAL_EXAMPLES_STR = "\n".join(f"- {q}" for q in REAL_QUERIES)

# ---------------------------------------------------------------------------
# Topic batches: (batch_label, module, type_hint, topic_description, count)
# ---------------------------------------------------------------------------
BATCHES = [
    ("diffusion_basics",        "Module 1: Diffusion",          "conceptual",  "GenAI history, orientation, diffusion models overview, what is diffusion, why it matters, generative AI evolution, milestones", 50),
    ("diffusion_sdxl_comfyui",  "Module 1: Diffusion",          "how-to",      "SDXL architecture, ComfyUI interface, node-based workflows, image generation pipelines, setting up ComfyUI", 50),
    ("controlnet_ip_adapters",  "Module 1: Diffusion",          "how-to",      "ControlNet, IP-Adapters, pose control, depth maps, Canny edges, style transfer, conditioning images, advanced image control", 50),
    ("video_lora",              "Module 1: Diffusion",          "how-to",      "Wan video model, LoRA training, FLUX fine-tuning, text-to-video, video inpainting, video outpainting, video ControlNets", 50),
    ("fullstack_ui",            "Module 2: Full Stack LLM",     "how-to",      "full stack orientation, building UI with Gradio, HuggingFace Spaces deployment, Python UI basics, Streamlit", 50),
    ("apis_fastapi",            "Module 2: Full Stack LLM",     "debugging",   "REST APIs, FastAPI, CRUD operations, HTTP methods, Postman, curl, Swagger UI, API endpoints, API testing errors", 50),
    ("database_supabase",       "Module 2: Full Stack LLM",     "how-to",      "Supabase, SQL queries, database schemas, ERD, domain modeling, anon key, service role key, connecting frontend to database", 50),
    ("vibecode_mvp",            "Module 2: Full Stack LLM",     "how-to",      "VibeCode, Cursor IDE, Lovable platform, Claude Code, MVP building, rapid prototyping, AI-assisted coding, GitHub integration", 50),
    ("llm_fundamentals",        "Module 2: Full Stack LLM",     "conceptual",  "LLM fundamentals, transformer architecture, attention mechanism, training process, open-source LLMs, chatbot deployment, model comparison", 50),
    ("prompt_engineering",      "Module 2: Full Stack LLM",     "conceptual",  "prompt engineering, LLM wrappers, system prompts, GPT integration, LLaMA, GroqCloud, prompt optimization, few-shot prompting", 50),
    ("tool_calling_mcp",        "Module 2: Full Stack LLM",     "how-to",      "tool calling, function calling, Model Context Protocol (MCP), building MCPs, context engineering, tool integration, LLM capabilities", 50),
    ("rag_fundamentals",        "Module 2: Full Stack LLM",     "conceptual",  "RAG basics, vector embeddings, vector databases, FAISS, similarity search, chunking, retrieval augmented generation, what is RAG", 50),
    ("advanced_rag_memory",     "Module 2: Full Stack LLM",     "how-to",      "advanced RAG, hybrid search, knowledge graphs, memory RAG, Mem0, Supermemory, indexing workflows, RAG optimization, memory types", 50),
    ("finetuning_llms",         "Module 2: Full Stack LLM",     "conceptual",  "fine-tuning LLMs, when to fine-tune, LoRA fine-tuning, QLoRA, dataset preparation, training costs, model customization", 50),
    ("data_prep_eval",          "Module 2: Full Stack LLM",     "how-to",      "data preparation for fine-tuning, dataset curation, data formatting, LLM evaluation, performance metrics, GenAI architecture patterns", 50),
    ("agents_intro",            "Module 3: AI Agents",          "conceptual",  "AI agents introduction, what are agents, LLM vs agentic apps, agent design patterns, agentic workflows, agent levels framework", 50),
    ("multi_agent_systems",     "Module 3: AI Agents",          "conceptual",  "multi-agent systems, agent communication protocols, agent orchestration, distributed agents, coordination strategies, when to use multi-agent", 50),
    ("agent_sdks_guardrails",   "Module 3: AI Agents",          "how-to",      "OpenAI Agents SDK, Claude Agents SDK, Google ADK, agent-to-agent (A2A), guardrails, monitoring agents, evaluation frameworks, safety", 50),
    ("agentic_workflows_prod",  "Module 3: AI Agents",          "how-to",      "agentic workflow design, production deployment of agents, agent scaling, cost optimization, deployment strategies, enterprise agents", 50),
    ("program_logistics",       "Program Logistics",            "logistics",   "LMS access, where are the recordings, Launchpad, Jarvis Labs GPU setup, session timings, cohort schedule, assignment submission, feedback forms", 50),
    ("debugging_errors",        "General",                      "debugging",   "Python errors, import errors, ModuleNotFoundError, API key errors, CUDA errors, environment setup issues, pip install problems, virtual env", 50),
    ("general_confused",        "General",                      "vague",       "students confused, frustrated, not understanding, vague questions like 'this doesnt work', 'i dont get it', asking for help without context", 50),
]


SYSTEM_PROMPT = """You generate realistic Discord messages from AI students asking their support bot questions.

Style rules (IMPORTANT):
- Vary tone: curious, confused, frustrated, casual, urgent
- Include realistic typos and informal language (like the examples below)
- Mix short vague queries with longer specific ones
- Some students write in broken English or make spelling mistakes
- Some are frustrated or lost
- Do NOT use formal/polished language - these are Discord messages
- Each query must be UNIQUE and on a different aspect of the topic

Real examples of how students actually write:
{examples}

Output ONLY a JSON array of strings. No markdown, no explanation."""


def generate_batch(label, module, qtype, topic_desc, count):
    user_prompt = f"""Generate exactly {count} student Discord queries about: {topic_desc}

Module context: {module}
Query type focus: {qtype}

Rules:
- Each query must be different (different aspect, different phrasing, different tone)
- Mix short (3-7 words) and long (15-40 words) queries
- About 20% should have typos or broken English
- About 10% should be frustrated or confused in tone
- Include variations: "what is X", "how to X", "i'm getting error X", "where is X", "help me with X"

Return ONLY a JSON array of {count} strings."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(examples=REAL_EXAMPLES_STR)},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()
    # strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    queries = json.loads(raw)
    return queries


def normalize(q):
    return q.lower().strip()


def deduplicate(all_records):
    seen = set()
    unique = []
    for r in all_records:
        norm = normalize(r["query"])
        if norm not in seen:
            seen.add(norm)
            unique.append(r)
    return unique


def main():
    all_records = []
    qid = 1

    total_batches = len(BATCHES)
    for idx, (label, module, qtype, topic_desc, count) in enumerate(BATCHES, 1):
        print(f"[*] Batch {idx}/{total_batches}: {label} ({count} queries)...")
        try:
            queries = generate_batch(label, module, qtype, topic_desc, count)
            for q in queries:
                all_records.append({
                    "id": qid,
                    "query": q.strip(),
                    "topic": label,
                    "module": module,
                    "type": qtype,
                })
                qid += 1
            print(f"    [OK] Got {len(queries)} queries (total so far: {len(all_records)})")
        except Exception as e:
            print(f"    [ERROR] Batch {label} failed: {e}")

        # small delay to avoid rate limits
        time.sleep(1)

    print(f"\n[*] Before dedup: {len(all_records)} queries")
    unique_records = deduplicate(all_records)
    print(f"[*] After dedup:  {len(unique_records)} queries")

    if len(unique_records) < 1000:
        print(f"[WARN] Only {len(unique_records)} unique queries generated (target: 1000)")
    else:
        print(f"[OK] Target met: {len(unique_records)} unique queries")

    # re-number ids after dedup
    for i, r in enumerate(unique_records, 1):
        r["id"] = i

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(unique_records, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved to {OUTPUT_PATH}")

    # summary by module
    print("\n--- Summary by module ---")
    from collections import Counter
    counts = Counter(r["module"] for r in unique_records)
    for mod, cnt in sorted(counts.items()):
        print(f"  {mod}: {cnt} queries")


if __name__ == '__main__':
    main()
