# start listening to the queue and start crawling the web pages

import asyncio
import json
import logging
import os
import sys
import threading
import time
from multiprocessing import Pool, Process, Queue, cpu_count
from typing import List
from urllib.parse import urljoin, urlparse

from crawler import QueueCrawler
from crawler.constants import (
    ASYNC_CRAWLERS,
    CHUNK_SIZE,
    CRAWL_AS_AGENT,
    ENABLE_AUTO_CRAWL,
    PROXY,
    REDIS_QUEUE_HOST,
    REDIS_QUEUE_NAME,
    REDIS_QUEUE_PASSWORD,
    REDIS_QUEUE_PORT,
    TIMEOUT,
)


async def run_crawlers(queue: Queue):
    crawler = QueueCrawler()
    tasks = [asyncio.create_task(crawler.run(queue))]
    print("Worker running")
    await asyncio.gather(*tasks)


def worker(queue: Queue):
    print("Worker started")

    # create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(run_crawlers(queue))

    print("Worker finished")


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


try:
    import redis
    import requests  # type: ignore
except ImportError:
    logger.error("Please install the required dependencies")
    sys.exit(1)


def get_redis_connection() -> redis.Redis:
    if not ENABLE_AUTO_CRAWL:
        logger.info("Auto crawl is disabled")

    if not REDIS_QUEUE_HOST:
        logger.error("REDIS_QUEUE_HOST is not set")
        sys.exit(1)

    if not REDIS_QUEUE_PORT:
        logger.error("REDIS_QUEUE_PORT is not set")
        sys.exit(1)

    if not REDIS_QUEUE_NAME:
        logger.error("REDIS_QUEUE_NAME is not set")
        sys.exit(1)

    if not REDIS_QUEUE_PASSWORD:
        logger.error("REDIS_QUEUE_PASSWORD is not set")
        sys.exit(1)

    return redis.Redis(
        host=REDIS_QUEUE_HOST,
        port=REDIS_QUEUE_PORT,
        password=REDIS_QUEUE_PASSWORD,
    )


def start_listening():
    logger.info("Starting the listener")

    redis_conn = get_redis_connection()

    while True:
        # get the next domain from the queue
        domain = redis_conn.blpop(REDIS_QUEUE_NAME, timeout=0)
        if not domain:
            logger.info("No domain found in the queue")
            time.sleep(1)
            continue

        domain = domain[1].decode("utf-8")
        logger.info(f"Got a domain: {domain}")

        # start crawling the domain
        worker(Queue(), 1)

        time.sleep(1)


if __name__ == "__main__":
    start_listening()
