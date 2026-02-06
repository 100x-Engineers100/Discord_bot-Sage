"""One-time setup script to build FAISS index from FAQ data

Run this script once to:
1. Parse FAQ_Doc.txt into structured Q&A pairs
2. Generate OpenAI embeddings
3. Build FAISS index
4. Save index and metadata to disk

Usage:
    python setup_scout_embeddings.py          # Interactive mode
    python setup_scout_embeddings.py --force  # Force rebuild without prompt
"""

import sys
import time
from pathlib import Path

import config
from scout_rag import ScoutRAG


def main():
    print("=" * 60)
    print("Scout RAG System - FAQ Embeddings Setup")
    print("=" * 60)

    # Check if data file exists
    if not config.FAQ_DATA_PATH.exists():
        print(f"\n[ERROR] FAQ file not found: {config.FAQ_DATA_PATH}")
        print("Please ensure FAQ_Doc.txt exists in the project root.")
        return

    # Check for --force flag
    force_rebuild = "--force" in sys.argv

    # Check if index already exists
    if config.FAQ_FAISS_INDEX_PATH.exists():
        print(f"\n[WARN] Index already exists at {config.FAQ_FAISS_INDEX_PATH}")
        if not force_rebuild:
            response = input("Rebuild index? This will overwrite existing data. (y/N): ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                return
        print("\nRebuilding index...")

    # Initialize RAG system
    print("\n[1/5] Initializing Scout RAG system...")
    rag = ScoutRAG()

    # Load and parse FAQs
    print(f"\n[2/5] Loading FAQ data from {config.FAQ_DATA_PATH}...")
    with open(config.FAQ_DATA_PATH, 'r', encoding='utf-8') as f:
        faq_text = f.read()

    print(f"[OK] Loaded {len(faq_text):,} characters")

    print("\n[3/5] Parsing FAQ into Q&A pairs...")
    start_time = time.time()
    faqs = rag.parse_faq(faq_text)
    parse_time = time.time() - start_time

    if not faqs:
        print("\n[ERROR] No FAQs parsed. Check FAQ_Doc.txt format.")
        return

    print(f"\n[OK] Parsed {len(faqs)} FAQ entries in {parse_time:.2f}s")

    # Show sample
    print("\nSample FAQ entries:")
    for i, sample in enumerate(faqs[:3], 1):
        print(f"\n{i}. Q: {sample['question'][:70]}...")
        print(f"   Category: {sample.get('category', 'unknown')}")
        print(f"   Type: {sample.get('type', 'unknown')}")
        if sample.get('is_urgent'):
            print(f"   [URGENT]")

    # Generate embeddings
    print("\n[4/5] Generating embeddings with OpenAI...")
    print(f"  Model: {config.EMBEDDING_MODEL}")
    print(f"  Dimension: {config.EMBEDDING_DIMENSION}")
    start_time = time.time()
    embeddings = rag.embed_faqs(faqs)
    embed_time = time.time() - start_time

    print(f"\n[OK] Generated embeddings in {embed_time:.2f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Cost estimate: ~$0.{int(len(faq_text) / 1000 * 0.00002 * 100):02d}")

    # Build and save index
    print("\n[5/5] Building and saving FAISS index...")
    rag.build_index(embeddings)
    rag.save_index(faqs)

    # Verify
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {config.FAQ_FAISS_INDEX_PATH}")
    print(f"  - {config.FAQ_METADATA_PATH}")
    print(f"\nIndex stats:")
    print(f"  - {len(faqs)} FAQ entries indexed")
    print(f"  - {embeddings.shape[1]} dimensions per vector")

    # Category breakdown
    categories = {}
    for faq in faqs:
        cat = faq.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")

    print("\n[OK] Ready for search!")

    print("\nNext steps:")
    print("  1. Test Scout RAG: python scout_rag.py")
    print("  2. Run dual bot: python bot.py")


if __name__ == "__main__":
    main()
