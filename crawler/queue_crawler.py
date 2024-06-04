from multiprocessing import Process, Queue
from typing import List

from crawler import Crawler, CrawlerFactory


class QueueCrawler:
    def __init__(self):
        pass

    async def run(self, dommains_queue: Queue):
        factory = CrawlerFactory

        while not dommains_queue.empty():
            domain = dommains_queue.get()

            crawler = factory.create(domain=domain)
            await crawler.crawl()
