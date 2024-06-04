import asyncio
import time

from sentence_transformers import SentenceTransformer  # type: ignore

from crawler.db import *
from crawler.utils.db import AsyncDB as DB
from crawler.utils.db import create_db


async def load_contents(limit: int = 1000, offset: int = 0):
    db = create_db()
    urls = await db.select("websites", ["url", "content"], limit=limit, offset=offset)
    db.close()

    return urls


async def generate_embeddings(device="mps", limit=1000, offset=0):
    start = time.time()

    urls = await load_contents(limit=limit, offset=offset)
    print(f"Loaded {len(urls)} urls in {time.time() - start} seconds")
    start = time.time()

    results = []
    for url, content in urls:
        embeddings = get_embeddings(content)
        results.append((url, embeddings))
    print(f"Embeddings generated in {time.time() - start} seconds")

    return results


async def move_to_redis(start_offset: int = 0):
    start = time.time()
    db = create_db()

    total_count = await db.count("websites")
    total_count = total_count[0]
    print(f"Total count: {total_count}")
    page_size = 32
    put_chunk_size = 1
    print(f"Initialized in {time.time() - start} seconds")
    start = time.time()

    for i in range(start_offset, total_count, page_size):
        start = time.time()

        urls = await db.select(
            "websites", ["url", "content"], limit=page_size, offset=i
        )

        dataset = []
        for url, content in urls:
            dataset.append((url, content))

        embeddings = get_embeddings([content for url, content in dataset])

        print(f"Moved {page_size} urls in {time.time() - start} seconds")
        print(f"Total moved: {i + page_size}")
        start = time.time()
