"""RAG system for curriculum search using FAISS + OpenAI embeddings

Optimized for Sage Discord bot with:
- OpenAI text-embedding-3-small (latest model)
- Structured lecture parsing with module/mentor metadata
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


class CurriculumRAG:
    """Manages curriculum embeddings and retrieval"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            max_retries=config.MAX_RETRIES,
            timeout=config.TIMEOUT
        )
        self.index = None
        self.metadata = []

        # Load index if exists
        if config.FAISS_INDEX_PATH.exists():
            self.load_index()

    def parse_curriculum(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse Data_Doc_main.txt into lecture chunks with module metadata

        Strategy:
        1. Split by module boundaries (Module1, Module 2, Module 3)
        2. Within each module, find all lectures
        3. Add module and mentor metadata
        """
        lectures = []

        # Module boundaries with mentor assignments
        MODULE_INFO = {
            1: {"name": "Diffusion", "mentor": "Boson", "pattern": r"Module1:\s*Diffusion"},
            2: {"name": "Full Stack LLM", "mentor": "Sid", "pattern": r"Module\s*2:\s*LLM"},
            3: {"name": "Agents", "mentor": "Sid", "pattern": r"Module\s*3:\s*AI Agents"}
        }

        # Find module boundaries
        module_starts = {}
        for module_num, info in MODULE_INFO.items():
            match = re.search(info["pattern"], text, re.IGNORECASE)
            if match:
                module_starts[module_num] = match.start()
            else:
                print(f"[WARN] Could not find Module {module_num} ({info['name']})")

        # Fallback if modules not found
        if not module_starts:
            print("[WARN] No module boundaries found. Parsing without modules.")
            return self._parse_without_modules(text)

        # Sort modules by position
        sorted_modules = sorted(module_starts.items(), key=lambda x: x[1])

        # Process each module
        for i, (module_num, start_pos) in enumerate(sorted_modules):
            # Determine end position (start of next module, or end of text)
            end_pos = sorted_modules[i + 1][1] if i + 1 < len(sorted_modules) else len(text)

            module_text = text[start_pos:end_pos]
            module_info = MODULE_INFO[module_num]

            print(f"\n[*] Parsing Module {module_num}: {module_info['name']} ({module_info['mentor']})")

            # Track lecture numbers within this module
            module_lecture_nums = []

            # Find all lectures: "Lecture N: Title" or "Lecture N. Title"
            lecture_pattern = r'^\s*Lecture\s+(\d+)[\.:]\s+(.+?)$'
            lecture_matches = list(re.finditer(lecture_pattern, module_text, re.MULTILINE | re.IGNORECASE))

            for j, match in enumerate(lecture_matches):
                lecture_num = int(match.group(1))
                lecture_title = match.group(2).strip()

                # Clean title: remove trailing punctuation
                lecture_title = re.sub(r'[:\.\-]+$', '', lecture_title).strip()
                lecture_title = ' '.join(lecture_title.split())  # normalize whitespace

                # Extract content: from current match to next lecture (or end of module)
                content_start = match.end()
                content_end = lecture_matches[j + 1].start() if j + 1 < len(lecture_matches) else len(module_text)

                lecture_content = module_text[content_start:content_end].strip()

                # Skip if content too short (likely parsing error)
                if len(lecture_content) < 100:
                    print(f"  [SKIP] Lecture {lecture_num}: {lecture_title} (content too short)")
                    continue

                # Check for duplicate lecture numbers within module
                if lecture_num in module_lecture_nums:
                    print(f"  [WARN] Duplicate Lecture {lecture_num} in Module {module_num}, skipping")
                    continue

                module_lecture_nums.append(lecture_num)

                # Extract topics
                topics = self._extract_topics(lecture_content, lecture_title)

                lectures.append({
                    "lecture_num": lecture_num,
                    "lecture_name": lecture_title,
                    "content": lecture_content,
                    "topics": topics,
                    "module": module_num,
                    "module_name": module_info["name"],
                    "mentor": module_info["mentor"]
                })

                print(f"  [OK] Lecture {lecture_num}: {lecture_title[:60]}")

        print(f"\n[OK] Parsed {len(lectures)} total lectures across {len(sorted_modules)} modules")
        return lectures

    def _parse_without_modules(self, text: str) -> List[Dict[str, Any]]:
        """Fallback parser when modules can't be detected"""
        lectures = []
        lecture_pattern = r'^\s*Lecture\s+(\d+)[\.:]\s+(.+?)$'
        lecture_matches = list(re.finditer(lecture_pattern, text, re.MULTILINE | re.IGNORECASE))

        for i, match in enumerate(lecture_matches):
            lecture_num = int(match.group(1))
            lecture_title = match.group(2).strip()

            # Clean title
            lecture_title = re.sub(r'[:\.\-]+$', '', lecture_title).strip()
            lecture_title = ' '.join(lecture_title.split())

            # Extract content
            content_start = match.end()
            content_end = lecture_matches[i + 1].start() if i + 1 < len(lecture_matches) else len(text)

            lecture_content = text[content_start:content_end].strip()

            if len(lecture_content) < 100:
                continue

            # Check for duplicates
            if any(lec['lecture_num'] == lecture_num for lec in lectures):
                continue

            topics = self._extract_topics(lecture_content, lecture_title)

            lectures.append({
                "lecture_num": lecture_num,
                "lecture_name": lecture_title,
                "content": lecture_content,
                "topics": topics
            })

        return lectures

    def _extract_topics(self, text: str, lecture_title: str = "") -> List[str]:
        """
        Extract key topics from lecture content

        Strategy:
        1. Parse "Key Concepts & Ideas" section
        2. Parse "Tools & Frameworks Introduced" section
        3. Extract capitalized technical terms
        4. Return top 8 most relevant topics
        """
        topics = set()

        # Extract from "Key Concepts & Ideas" section
        concepts_match = re.search(
            r'Key Concepts & Ideas\s*\n(.*?)(?=\n\s*(?:Tools & Frameworks|Implementation|What Was Covered|$))',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if concepts_match:
            concepts_text = concepts_match.group(1)
            # Extract capitalized phrases (2-4 words)
            capitalized = re.findall(r'\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*){0,3})\b', concepts_text)
            topics.update(capitalized[:15])

        # Extract from "Tools & Frameworks Introduced" section
        tools_match = re.search(
            r'Tools & Frameworks Introduced\s*\n(.*?)(?=\n\s*(?:Implementation|Key Concepts|What Was Covered|$))',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if tools_match:
            tools_text = tools_match.group(1)
            # Extract tool names (capitalized terms)
            tool_names = re.findall(r'\b([A-Z][A-Za-z0-9]*(?:[A-Z][a-z0-9]*)*)\b', tools_text)
            topics.update(tool_names[:10])

        # Extract from lecture title
        if lecture_title:
            title_terms = re.findall(r'\b([A-Z][A-Za-z0-9]*(?:[A-Z][a-z0-9]*)*)\b', lecture_title)
            topics.update(title_terms)

        # Filter stopwords
        stopwords = {
            'The', 'This', 'That', 'These', 'Those', 'What', 'Where', 'When', 'Why', 'How',
            'Introduction', 'Session', 'Lecture', 'Overview', 'Summary', 'Covered', 'Was',
            'Building', 'Using', 'Creating', 'Making', 'Advanced', 'Basic', 'Practical',
            'AI', 'A', 'An', 'And', 'Or', 'But', 'For', 'To', 'From', 'With', 'In', 'On',
            'Key', 'Main', 'Important', 'Core', 'Primary', 'Secondary', 'Additional',
            'First', 'Second', 'Third', 'Final', 'Next', 'Previous', 'Current', 'New',
            'Different', 'Similar', 'Same', 'Other', 'Another', 'Each', 'Every', 'All',
            'Evolution', 'History', 'Background', 'Context', 'Concepts', 'Ideas', 'Insights',
            'Structure', 'Pattern', 'Approach', 'Method', 'Technique', 'Strategy', 'Process',
            'Migration', 'Transition', 'Implementation', 'Development', 'Deployment',
            'Intro', 'Knowing', 'Subject', 'Advantages', 'Interface', 'Graphical', 'Training',
            'Instead', 'Stable', 'Within', 'Tuning', 'Quality', 'Image', 'Images',
            'WHAT', 'WAS', 'COVERED', 'Your', 'Were', 'Have'
        }

        # Deduplicate case-insensitively, filter stopwords and short terms
        topics_lower = {}
        for t in topics:
            # Filter: not stopword, at least 4 chars, skip all-caps headers
            if t not in stopwords and len(t) >= 4:
                if t.isupper() and len(t) > 4:
                    continue  # Skip section headers like "COVERED"
                # Keep first occurrence
                if t.lower() not in topics_lower:
                    topics_lower[t.lower()] = t

        topics = set(topics_lower.values())

        # Fallback: extract from first 500 chars if too few topics
        if len(topics) < 3:
            first_500 = text[:500]
            fallback_terms = re.findall(r'\b([A-Z][A-Za-z0-9]+(?:[A-Z][a-z0-9]*)*)\b', first_500)
            topics.update(t for t in fallback_terms if t not in stopwords and len(t) > 2)

        # Return top 8 topics (sorted by length for stability)
        return sorted(list(topics), key=lambda x: len(x), reverse=True)[:8]

    def embed_curriculum(self, lectures: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed all lectures using OpenAI text-embedding-3-small
        Returns: numpy array of embeddings (n_lectures x embedding_dim)
        """
        print(f"Embedding {len(lectures)} lectures...")

        # Prepare texts (truncate to ~6000 tokens = ~24000 chars to stay under 8192 limit)
        MAX_CHARS = 24000
        texts = []
        for lec in lectures:
            content = lec['content'][:MAX_CHARS] if len(lec['content']) > MAX_CHARS else lec['content']
            texts.append(f"{lec['lecture_name']}\n\n{content}")

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
        print(f"[OK] Embedded {len(embeddings)} lectures. Shape: {embeddings.shape}")

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
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))

        # Save metadata
        self.metadata = metadata
        with open(config.METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved index to {config.FAISS_INDEX_PATH}")
        print(f"[OK] Saved metadata to {config.METADATA_PATH}")

    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if not config.FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"Index not found at {config.FAISS_INDEX_PATH}")

        self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))

        with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"[OK] Loaded index with {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = None,
        module: Optional[int] = None,
        mentor: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search curriculum for relevant lectures

        Args:
            query: Search query
            top_k: Number of results to return (default: config.TOP_K_RESULTS)
            module: Filter by module (1=Diffusion, 2=LLM, 3=Agents)
            mentor: Filter by mentor name ("Boson" or "Sid")

        Returns:
            List of lecture metadata with scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Run setup_embeddings.py first.")

        # Validate query
        if not query or not query.strip():
            return []

        top_k = top_k or config.TOP_K_RESULTS

        # Embed query
        response = self.client.embeddings.create(
            input=[query],
            model=config.EMBEDDING_MODEL
        )
        query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

        # Search more results if filtering (to compensate for post-filter reduction)
        search_k = top_k * 3 if (module or mentor) else top_k
        distances, indices = self.index.search(query_embedding, search_k)

        # Format results and apply filters
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            result = self.metadata[idx].copy()
            result['score'] = float(dist)

            # Apply filters
            if module is not None and result.get('module') != module:
                continue
            if mentor is not None and result.get('mentor') != mentor:
                continue

            results.append(result)

            # Stop once we have enough filtered results
            if len(results) >= top_k:
                break

        return results[:top_k]

    def get_lecture_by_num(self, lecture_num: int) -> Optional[Dict[str, Any]]:
        """Retrieve full lecture by number"""
        for lec in self.metadata:
            if lec['lecture_num'] == lecture_num:
                return lec
        return None


if __name__ == "__main__":
    # Quick test
    rag = CurriculumRAG()

    if config.FAISS_INDEX_PATH.exists():
        results = rag.search("agents and tool calling")
        print("\nTest search results:")
        for i, res in enumerate(results, 1):
            module_info = f"Module {res.get('module', '?')}" if 'module' in res else "No module"
            print(f"{i}. {res['lecture_name']} (Lecture {res['lecture_num']}) - {module_info}")
            print(f"   Score: {res['score']:.4f}")
            print(f"   Topics: {', '.join(res['topics'][:5])}")
    else:
        print("Index not found. Run setup_embeddings.py first.")
