import asyncio
import logging
import time
from multiprocessing import Pool
from typing import Set

from crawler.constants import CHUNK_SIZE, ENABLE_AUTO_CRAWL

# from crawler.db.redis import *
from crawler.manager import DomainManager
from crawler.utils.content_processor import ContentProcessor
from crawler.utils.db import AsyncDB as DB
from crawler.utils.db import create_db

# from crawler.utils.db import create_db
from crawler.utils.delay_controller import DelayController
from crawler.utils.url_loader import UrlLoader

logger = logging.getLogger()


class Crawler:
    def __init__(
        self,
        domain: str,
        user_agent: str,
        url_loader: UrlLoader,
        content_processor: ContentProcessor,
    ):
        self._domain = domain
        self._user_agent = user_agent
        self._url_loader = url_loader
        self._content_processor = content_processor

    async def _check_if_already_processed(self, url: str, db: DB) -> bool:
        try:
            result = await db.select("websites", ["1"], where=f"url = '{url}'")
        except Exception as e:
            logger.error(f"Error during checking if already processed: {e}")
            return False
        else:
            return bool(result)

    async def _crawl_url(self, url: str, db: DB) -> Set[str]:
        try:
            if await self._check_if_already_processed(url, db):
                print(f"Skipped {url}")
                logger.info(f"Skipped {url}")
                return set()

            print(f"Crawling URL: {url}")
            logger.info(f"Crawling URL: {url}")

            await self._delay_controller.wait()

            status_code, content = await self._url_loader(
                url,
                protected=True,
                raise_exceptions=False,
            )
            self._delay_controller.step()

            if status_code == 200:
                return await self._content_processor(url, content, db)

            print(f"Error during crawling URL: {url} - status code: {status_code}")
            logger.error(
                f"Error during crawling URL: {url} - status code: {status_code}"
            )
        except Exception as e:
            print(f"Error during crawling URL: {url} - {e}")
            logger.error(f"Error during crawling URL: {url} - {e}")

        return set()

    async def crawl(self):
        print(f"Starting to crawl domain: {self._domain}")
        logger.info(f"Starting to crawl domain: {self._domain}")

        self._domain_manager = DomainManager(
            self._user_agent, self._domain, self._url_loader
        )
        self._delay_controller = DelayController(
            await self._domain_manager.delay, await self._domain_manager.request_rate
        )

        urls = await self._domain_manager.urls
        print(f"Loaded URLs {self._domain}: {len(urls)}")
        logger.info(f"Loaded URLs {self._domain}: {len(urls)}")

        if not urls:
            logger.info("No URLs to crawl for domain: {self._domain}")
            return

        db = create_db()

        crawled_urls_count = 0
        tasks = list(urls)

        print(f"Starting to crawl URLs: {len(urls)}")

        while tasks:
            tasks_chunks = [
                tasks[i : i + CHUNK_SIZE] for i in range(0, len(tasks), CHUNK_SIZE)
            ]

            # Wait for all tasks to finish
            results = []
            for chunk in tasks_chunks:
                chunk_tasks = [
                    asyncio.create_task(self._crawl_url(url, db)) for url in chunk
                ]
                chunk_results = await asyncio.wait(
                    chunk_tasks, return_when=asyncio.ALL_COMPLETED
                )
                temp_results = [task.result() for task in chunk_results[0]]
                results.extend(temp_results)

            if not ENABLE_AUTO_CRAWL:
                break

            # Get all results
            total_count = len(results)
            filtered = [result for result in results if result is not None]
            merged_urls = set().union(*filtered)
            merged_urls = await self._domain_manager._filter(
                merged_urls, check_can_fetch=True
            )
            print(f"Discovered URLs: {len(urls)}")

            # Update tasks and crawled URLs count
            crawled_urls_count += total_count - len(filtered)
            tasks = list(merged_urls)

        db.close()

        print(
            f"Finished crawling domain: {self._domain}, crawled URLs: {crawled_urls_count}"
        )
        logger.info(
            f"Finished crawling domain: {self._domain}, crawled URLs: {crawled_urls_count}"
        )
