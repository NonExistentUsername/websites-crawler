import asyncio
import time
import uuid
from typing import Dict, List

import numpy as np
import redis.asyncio as redis
import torch
import torch.nn.functional as F
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, pipeline  # type: ignore

from crawler.constants import REDIS_DB, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT

EMBEDDING_MODEL_NAME = "andersonbcdefg/bge-small-4096"
VECTOR_DIMENSION = 384
TOKENS_LIMIT = 4096 - 16  # To be safe
DEVICE = "mps"
INDEX_NAME = "idx:pages_vss"

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, truncation=True)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).half().to(DEVICE)

pipe = pipeline(
    "feature-extraction",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=DEVICE,
)


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
    )


async def init_database_for_crawler():
    # clear redis
    client = get_redis_client()
    await client.flushdb()

    schema = (
        TextField(name="url"),
        VectorField(
            "embeddings",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSION,
                "DISTANCE_METRIC": "COSINE",
            },
        ),
    )
    definition = IndexDefinition(
        prefix=["pages"],
        index_type=IndexType.HASH,
    )
    result = await client.ft(INDEX_NAME).create_index(
        fields=schema, definition=definition
    )
    await client.aclose()
    return result


async def put_crawled_url(url: str, embedings):
    client = get_redis_client()
    result = await client.hset(
        f"pages:{url}", mapping={"url": url, "embeddings": embedings}
    )
    await client.aclose()
    return result


async def generate_random_embeding():
    return np.random.rand(VECTOR_DIMENSION).astype(np.float32).tobytes()


def generate_random_url():
    return f"https://example.com/{uuid.uuid4().hex}"


async def generate_urls(count: int):
    client = get_redis_client()

    chunk_size = 512
    for _ in range(0, count, chunk_size):
        urls = [generate_random_url() for _ in range(chunk_size)]
        embedings = [await generate_random_embeding() for _ in range(chunk_size)]
        tasks = [
            asyncio.create_task(put_crawled_url(url, embeding))
            for url, embeding in zip(urls, embedings)
        ]
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    await client.aclose()


async def random_search(k: int = 10):
    start = time.time()
    query_embeding = await generate_random_embeding()

    print(f"Query embeding generated in {time.time() - start:.2f}")
    start = time.time()

    client = get_redis_client()
    print(f"Redis client created in {time.time() - start:.2f}")
    start = time.time()

    query_obj = (
        Query(f"(*)=>[KNN {k} @embeddings $query_vector AS vector_score]")
        .return_fields("url", "vector_score")
        .sort_by("vector_score", asc=True)
        .dialect(2)
    )
    print(f"Query object created in {time.time() - start:.2f}")
    start = time.time()

    results = await client.ft(INDEX_NAME).search(
        query_obj,
        {
            "query_vector": query_embeding,
        },
    )
    print(f"Search executed in {time.time() - start:.2f}")
    start = time.time()

    await client.aclose()
    print(f"Redis client closed in {time.time() - start:.2f}")
    start = time.time()

    return results.docs


async def get_crawled_url(url: str):
    client = get_redis_client()
    result = await client.ft(INDEX_NAME).load_document(f"pages:{url}")
    await client.aclose()
    return result


async def delete_url(url: str):
    client = get_redis_client()
    result = await client.delete(f"pages:{url}")
    await client.aclose()
    return result


async def get_all_urls_from_redis():
    client = get_redis_client()
    keys = await client.keys("pages:*")
    urls = [key.decode("utf-8")[6:] for key in keys]
    await client.aclose()
    return urls


async def delete_bad_urls(chunk_size: int = 512):
    bad_embeding = "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f"

    client = get_redis_client()

    keys = await client.keys("pages:*")
    urls = [key.decode("utf-8")[6:] for key in keys]
    print(f"Found {len(urls)} urls")
    print(f"Chunks count: {len(urls) // chunk_size}")

    for i in range(0, len(urls), chunk_size):
        urls_chunk = urls[i : i + chunk_size]
        loaded_docs = [
            asyncio.create_task(client.ft(INDEX_NAME).load_document(f"pages:{url}"))
            for url in urls_chunk
        ]
        loaded_docs = await asyncio.gather(*loaded_docs)
        urls_to_delete = [
            url
            for url, doc in zip(urls_chunk, loaded_docs)
            if doc is None
            or doc.embeddings == bad_embeding
            or url.startswith("http://himasoku.com")
        ]
        to_delete_tasks = [
            asyncio.create_task(client.delete(f"pages:{url}")) for url in urls_to_delete
        ]
        await asyncio.gather(*to_delete_tasks)
        if i % 10 == 0:
            print(f"Processed {i} urls")

    await client.aclose()


def merge_embeddings(embeddings):
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Merge embeddings
    embeddings = embeddings.mean(dim=0)

    return embeddings


def average_pool(states: Tensor) -> Tensor:
    return states.mean(dim=0)


def prepare_text(text: str):
    tokens = tokenizer(text, padding=False, truncation=False)
    chunks = []
    for i in range(0, len(tokens["input_ids"]), TOKENS_LIMIT):
        chunk = {
            "input_ids": tokens["input_ids"][i : i + TOKENS_LIMIT],
            "attention_mask": tokens["attention_mask"][i : i + TOKENS_LIMIT],
        }
        chunks.append(chunk)

    texts = []
    for chunk in chunks:
        text = tokenizer.decode(
            chunk["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        texts.append(text)

    return texts


def get_embeddings(text: str) -> Tensor:
    texts = prepare_text(text)

    outputs: List[List[float]] = []

    for text in texts:
        output = pipe(text)[0]
        outputs.extend(output)

    embeddings_list = torch.tensor(outputs)

    return average_pool(embeddings_list).cpu().numpy().astype(np.float32).tobytes()


def get_embeddings_batch(batch: List[str]) -> List[Tensor]:
    texts = [prepare_text(text) for text in batch]
    sizes = [len(text) for text in texts]
    texts = [text for text in texts for text in text]

    embeddings: List[Tensor] = []

    output = pipe(texts)
    iterator = iter(output)

    for chunk_size in sizes:
        chunk = []
        for _ in range(chunk_size):
            chunk.extend(next(iterator)[0])

        embeddings.append(
            average_pool(torch.tensor(chunk)).cpu().numpy().astype(np.float32).tobytes()
        )

    return embeddings


async def put_url(url: str, text: str):
    embedings = get_embeddings(text)
    return await put_crawled_url(url, embedings)


async def get_info():
    client = get_redis_client()
    result = await client.ft(INDEX_NAME).info()
    await client.aclose()
    return result


async def search(query: str, k: int = 10, results_count: int = 10):
    client = get_redis_client()
    query_obj = (
        Query(f"(*)=>[KNN {k} @embeddings $query_vector AS vector_score]")
        .return_fields("url", "vector_score")
        .sort_by("vector_score", asc=True)
        .paging(0, results_count)
        .dialect(2)
    )
    results = await client.ft(INDEX_NAME).search(
        query_obj,
        {
            "query_vector": get_embeddings(query),
        },
    )
    print(results)
    await client.aclose()
    return results.docs


async def range_search(query: str, radius: float = 0.5):
    client = get_redis_client()
    query_obj = (
        Query(
            f"@vector:[VECTOR_RANGE {radius} $query_vector]=>{{$YIELD_DISTANCE_AS: vector_score}}"
        )
        .sort_by("vector_score", asc=True)
        .return_fields("url", "vector_score")
        .dialect(2)
    )
    results = await client.ft(INDEX_NAME).search(
        query_obj,
        {
            "query_vector": get_embeddings(query),
        },
    )
    await client.aclose()
    return results.docs
