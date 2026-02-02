"""One-time setup script to build FAISS index from curriculum data

Run this script once to:
1. Parse Data_Doc_main.txt into structured lectures
2. Generate OpenAI embeddings
3. Build FAISS index
4. Save index and metadata to disk

Usage:
    python setup_embeddings.py          # Interactive mode
    python setup_embeddings.py --force  # Force rebuild without prompt
"""

import sys
import time
from pathlib import Path

import config
from curriculum_rag import CurriculumRAG


def main():
    print("=" * 60)
    print("Sage RAG System - Embeddings Setup")
    print("=" * 60)

    # Check if data file exists
    if not config.DATA_PATH.exists():
        print(f"\n[ERROR] Data file not found: {config.DATA_PATH}")
        print("Please ensure Data_Doc_main.txt exists in the project root.")
        return

    # Check for --force flag
    force_rebuild = "--force" in sys.argv

    # Check if index already exists
    if config.FAISS_INDEX_PATH.exists():
        print(f"\n[WARN] Index already exists at {config.FAISS_INDEX_PATH}")
        if not force_rebuild:
            response = input("Rebuild index? This will overwrite existing data. (y/N): ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                return
        print("\nRebuilding index...")

    # Initialize RAG system
    print("\n[1/5] Initializing RAG system...")
    rag = CurriculumRAG()

    # Load and parse curriculum
    print(f"\n[2/5] Loading curriculum from {config.DATA_PATH}...")
    with open(config.DATA_PATH, 'r', encoding='utf-8') as f:
        curriculum_text = f.read()

    print(f"[OK] Loaded {len(curriculum_text):,} characters")

    print("\n[3/5] Parsing curriculum into lectures...")
    start_time = time.time()
    lectures = rag.parse_curriculum(curriculum_text)
    parse_time = time.time() - start_time

    if not lectures:
        print("\n[ERROR] No lectures parsed. Check Data_Doc_main.txt format.")
        return

    print(f"\n[OK] Parsed {len(lectures)} lectures in {parse_time:.2f}s")

    # Show sample
    print("\nSample lecture:")
    sample = lectures[0]
    print(f"  - Lecture {sample['lecture_num']}: {sample['lecture_name']}")
    print(f"  - Module: {sample.get('module_name', 'N/A')}")
    print(f"  - Mentor: {sample.get('mentor', 'N/A')}")
    print(f"  - Topics: {', '.join(sample['topics'][:5])}")
    print(f"  - Content length: {len(sample['content'])} chars")

    # Generate embeddings
    print("\n[4/5] Generating embeddings with OpenAI...")
    print(f"  Model: {config.EMBEDDING_MODEL}")
    print(f"  Dimension: {config.EMBEDDING_DIMENSION}")
    start_time = time.time()
    embeddings = rag.embed_curriculum(lectures)
    embed_time = time.time() - start_time

    print(f"\n[OK] Generated embeddings in {embed_time:.2f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Cost estimate: ~$0.{int(len(curriculum_text) / 1000 * 0.00002 * 100):02d}")

    # Build and save index
    print("\n[5/5] Building and saving FAISS index...")
    rag.build_index(embeddings)
    rag.save_index(lectures)

    # Verify
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {config.FAISS_INDEX_PATH}")
    print(f"  - {config.METADATA_PATH}")
    print(f"\nIndex stats:")
    print(f"  - {len(lectures)} lectures indexed")
    print(f"  - {embeddings.shape[1]} dimensions per vector")
    print(f"  - Ready for search!")

    print("\nNext steps:")
    print("  1. Run: python test_rag.py (to test search)")
    print("  2. Update bot.py to use new RAG system")


if __name__ == "__main__":
    main()
