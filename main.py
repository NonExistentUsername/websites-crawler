import asyncio
from multiprocessing import Pool, Process, Queue, cpu_count
from typing import List

from crawler import QueueCrawler


async def run_crawlers(queue: Queue, async_crawlers: int):
    tasks = []
    for _ in range(async_crawlers):
        crawler = QueueCrawler()
        tasks.append(asyncio.create_task(crawler.run(queue)))

    print("Worker running")
    await asyncio.gather(*tasks)


def worker(queue: Queue, async_crawlers: int):
    print("Worker started")

    # create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(run_crawlers(queue, async_crawlers))

    print("Worker finished")


def run(
    domains_file_path: str,
    processes_count: int,
    async_crawlers: int,
    chunk_size: int = 32,
):
    print("Starting")
    with open(domains_file_path) as f:
        domains = f.read().splitlines()

    MAX_DOMAINS = 10000
    domains = domains[:MAX_DOMAINS]

    print(f"Domains: {len(domains)}")
    print("Initializing queue")
    queue: Queue = Queue(maxsize=len(domains))
    for domain in domains:
        queue.put_nowait(domain)

    processes_count = min(processes_count, cpu_count())
    processes: List[Process] = []

    print(f"Starting {processes_count} processes")

    for _ in range(processes_count):
        p = Process(target=worker, args=(queue, async_crawlers))
        p.start()
        processes.append(p)

    print("Waiting for processes to finish")

    for p in processes:
        p.join()

    print("All done")


if __name__ == "__main__":
    run("domains.txt", 1, 1)
