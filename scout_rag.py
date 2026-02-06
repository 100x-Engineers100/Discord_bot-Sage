"""RAG system for FAQ search using FAISS + OpenAI embeddings

Optimized for Scout Discord bot with:
- OpenAI text-embedding-3-small (latest model)
- Q&A pair parsing with urgency detection
- Persistent FAISS index (fast loading)
- Production-ready error handling
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from openai import OpenAI

import config


class ScoutRAG:
    """Manages FAQ embeddings and retrieval"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            max_retries=config.MAX_RETRIES,
            timeout=config.TIMEOUT
        )
        self.index = None
        self.metadata = []

        # Load index if exists
        if config.FAQ_FAISS_INDEX_PATH.exists():
            self.load_index()

    def parse_faq(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse FAQ_Doc.txt into Q&A chunks with metadata

        Strategy:
        1. Split FAQ section from curriculum index
        2. Extract Q&A pairs
        3. Add urgency/category metadata
        4. Handle curriculum index separately
        """
        faqs = []

        # Split document into FAQ section and curriculum index
        curriculum_split = re.search(
            r'={3,}\s*CURRICULUM INDEX',
            text,
            re.IGNORECASE
        )

        if curriculum_split:
            faq_section = text[:curriculum_split.start()].strip()
            curriculum_section = text[curriculum_split.start():].strip()
            print("[*] Detected curriculum index section")
        else:
            faq_section = text
            curriculum_section = ""
            print("[WARN] No curriculum index section found")

        # Parse FAQ Q&A pairs
        faq_entries = self._parse_faq_section(faq_section)
        faqs.extend(faq_entries)

        # Parse curriculum index if exists
        if curriculum_section:
            curriculum_entries = self._parse_curriculum_index(curriculum_section)
            faqs.extend(curriculum_entries)

        print(f"\n[OK] Parsed {len(faqs)} total FAQ entries")
        return faqs

    def _parse_faq_section(self, text: str) -> List[Dict[str, Any]]:
        """Parse general FAQ Q&A pairs"""
        entries = []

        # Pattern: Question followed by "A-" answer
        # Questions typically end with "?"
        qa_pattern = r'([^\n]+\?)\s*\n\s*A[-\s]+(.+?)(?=\n\s*[^\n]+\?|$)'

        matches = re.finditer(qa_pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()

            # Clean up answer
            answer = re.sub(r'\n\s*\n', '\n', answer)  # Remove excessive newlines
            answer = re.sub(r' +', ' ', answer)  # Normalize spaces

            # Skip if too short (likely parsing error)
            if len(answer) < 20:
                continue

            # Detect category
            category = self._detect_category(question, answer)

            # Detect urgency keywords
            is_urgent = self._is_urgent_topic(question, answer)

            # Extract mentioned links/emails
            links = self._extract_links(answer)
            emails = self._extract_emails(answer)

            entries.append({
                "question": question,
                "answer": answer,
                "category": category,
                "is_urgent": is_urgent,
                "links": links,
                "emails": emails,
                "type": "faq"
            })

            print(f"  [OK] Q: {question[:60]}...")

        print(f"[OK] Parsed {len(entries)} FAQ Q&A pairs")
        return entries

    def _parse_curriculum_index(self, text: str) -> List[Dict[str, Any]]:
        """Parse curriculum index section"""
        entries = []

        # Extract module sections
        module_pattern = r'MODULE\s+(\d+):\s+([^\n]+)'
        module_matches = list(re.finditer(module_pattern, text, re.IGNORECASE))

        for i, match in enumerate(module_matches):
            module_num = int(match.group(1))
            module_name = match.group(2).strip()

            # Extract content until next module or end
            start_pos = match.end()
            end_pos = module_matches[i + 1].start() if i + 1 < len(module_matches) else len(text)

            module_content = text[start_pos:end_pos].strip()

            # Parse lectures within module
            lecture_pattern = r'Lecture\s+\d+:\s+[^\n]+\s+—\s+Tags:\s+[^\n]+'
            lectures = re.findall(lecture_pattern, module_content, re.IGNORECASE)

            # Create single entry for this module with all lectures
            if lectures:
                entries.append({
                    "question": f"What is covered in Module {module_num}: {module_name}?",
                    "answer": f"Module {module_num}: {module_name}\n\n" + "\n\n".join(lectures),
                    "category": "curriculum",
                    "module": module_num,
                    "module_name": module_name,
                    "is_urgent": False,
                    "links": [],
                    "emails": [],
                    "type": "curriculum_index"
                })

                print(f"  [OK] Module {module_num}: {module_name} ({len(lectures)} lectures)")

        print(f"[OK] Parsed {len(entries)} curriculum modules")
        return entries

    def _detect_category(self, question: str, answer: str) -> str:
        """Detect FAQ category from content"""
        text = (question + " " + answer).lower()

        # Category keywords
        categories = {
            "lms": ["lms", "learning management", "dashboard"],
            "launchpad": ["launchpad", "form", "track", "discovery call"],
            "sessions": ["session", "lecture", "recording", "calendar", "timing"],
            "jarvis": ["jarvis", "credits", "instance", "gpu"],
            "technical": ["error", "code", "debug", "not working"],
            "curriculum": ["module", "week", "lecture", "prerequisite", "python"]
        }

        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return category

        return "general"

    def _is_urgent_topic(self, question: str, answer: str) -> bool:
        """Check if FAQ topic is typically urgent"""
        text = (question + " " + answer).lower()

        urgent_keywords = [
            "can't access", "not working", "error", "crashed",
            "deadline", "credits depleted", "today", "asap"
        ]

        return any(kw in text for kw in urgent_keywords)

    def _extract_links(self, text: str) -> List[str]:
        """Extract URLs from answer"""
        url_pattern = r'https?://[^\s<>"\']+'
        links = re.findall(url_pattern, text)
        return links[:5]  # Limit to 5 links

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from answer"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return list(set(emails))  # Deduplicate

    def embed_faqs(self, faqs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed all FAQs using OpenAI text-embedding-3-small
        Returns: numpy array of embeddings (n_faqs x embedding_dim)
        """
        print(f"Embedding {len(faqs)} FAQ entries...")

        # Prepare texts (combine question + answer for better search)
        MAX_CHARS = 24000  # Stay under 8192 token limit
        texts = []
        for faq in faqs:
            combined = f"Q: {faq['question']}\n\nA: {faq['answer']}"
            if len(combined) > MAX_CHARS:
                combined = combined[:MAX_CHARS]
            texts.append(combined)

        # Batch embed (OpenAI allows up to 2048 inputs per request)
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            response = self.client.embeddings.create(
                input=batch,
                model=config.EMBEDDING_MODEL
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        print(f"[OK] Embedded {len(embeddings)} FAQs. Shape: {embeddings.shape}")

        return embeddings

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings"""
        dimension = embeddings.shape[1]

        # Use L2 distance (FAISS default)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print(f"[OK] Built FAISS index with {self.index.ntotal} vectors")

    def save_index(self, metadata: List[Dict[str, Any]]):
        """Save FAISS index and metadata to disk"""
        # Save index
        faiss.write_index(self.index, str(config.FAQ_FAISS_INDEX_PATH))

        # Save metadata
        self.metadata = metadata
        with open(config.FAQ_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved index to {config.FAQ_FAISS_INDEX_PATH}")
        print(f"[OK] Saved metadata to {config.FAQ_METADATA_PATH}")

    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if not config.FAQ_FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"Index not found at {config.FAQ_FAISS_INDEX_PATH}")

        self.index = faiss.read_index(str(config.FAQ_FAISS_INDEX_PATH))

        with open(config.FAQ_METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"[OK] Loaded FAQ index with {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = None,
        category: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Search FAQs for relevant answers

        Args:
            query: Search query
            top_k: Number of results to return (default: config.TOP_K_RESULTS)
            category: Filter by category (lms, launchpad, sessions, jarvis, etc.)

        Returns:
            List of FAQ metadata with scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Run setup_scout_embeddings.py first.")

        # Validate query
        if not query or not query.strip():
            return []

        top_k = top_k or config.TOP_K_RESULTS

        # Embed query (skip if pre-computed for cross-bot routing)
        if query_embedding is None:
            response = self.client.embeddings.create(
                input=[query],
                model=config.EMBEDDING_MODEL
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

        # Search more results if filtering
        search_k = top_k * 3 if category else top_k
        distances, indices = self.index.search(query_embedding, search_k)

        # Format results and apply filters
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            result = self.metadata[idx].copy()
            result['score'] = float(dist)

            # Apply category filter
            if category is not None and result.get('category') != category:
                continue

            results.append(result)

            # Stop once we have enough filtered results
            if len(results) >= top_k:
                break

        return results[:top_k]


if __name__ == "__main__":
    # Quick test
    rag = ScoutRAG()

    if config.FAQ_FAISS_INDEX_PATH.exists():
        results = rag.search("Where can I find session recordings?")
        print("\nTest search results:")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['question'][:80]}")
            print(f"   Category: {res.get('category', 'unknown')}")
            print(f"   Score: {res['score']:.4f}")
            if res.get('links'):
                print(f"   Links: {', '.join(res['links'][:2])}")
    else:
        print("Index not found. Run setup_scout_embeddings.py first.")
