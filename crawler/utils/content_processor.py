import asyncio
import logging
import random
import time
from typing import Dict, Generator, List, Optional, Set

import aiopg

# import text2text as t2t  # type: ignore
import tika  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

# from crawler.db.redis import *
from crawler.utils.db import AsyncDB as DB

tika.initVM()

from tika import parser

logger = logging.getLogger()

# disable warnings from bs4
logging.getLogger("bs4").setLevel(logging.ERROR)


def tokenize(text: str) -> List[str]:
    # Old implementation
    # words = t2t.Tokenizer().transform(text)[0]
    words: List[str] = text.split()

    return [word.replace('"', "@q").replace("'", "@q") for word in words]


class ContentProcessor:
    def __init__(self):
        pass

    def __try_extract_urls(self, content: bytes) -> Set[str]:
        try:
            soup = BeautifulSoup(content, "html.parser")
            return {
                a["href"]
                for a in soup.find_all("a", href=True)
                if not a["href"].startswith("#")
            }
        except Exception as e:
            print(f"Error during extracting urls: {e}")
            return set()

    def parse_text(self, content: bytes):
        return parser.from_buffer(content)["content"]

    async def __call__(self, url: str, content: bytes, db: DB) -> Set[str]:
        text = self.parse_text(content)

        await self._process(url, text, db)

        return self.__try_extract_urls(content)

    async def _process(self, url: str, text: str, db: DB) -> None:
        await db.insert(
            "websites",
            ["url", "content"],
            [url, text],
        )

    # async def _process(self, url: str, text: str, db: DB) -> None:
    #     start = time.time()
    #     words = tokenize(text)

    #     word_to_count = {}
    #     word_positions = []

    #     for index, word in enumerate(words):
    #         if word not in word_to_count:
    #             word_to_count[word] = 0

    #         word_to_count[word] += 1

    #         word_positions.append(index)  # TODO

    #     retry_count = 0
    #     while retry_count < 3:
    #         try:
    #             await db.insert(
    #                 "websites",
    #                 ["url", "word_count"],
    #                 [url, len(words)],
    #             )

    #             await db.insertMany(
    #                 "website_keywords",
    #                 ["keyword_id", "website_id", "occurrences", "position"],
    #                 [
    #                     (
    #                         word,
    #                         url,
    #                         word_to_count[word],
    #                         word_positions[index],
    #                     )
    #                     for index, word in enumerate(words)
    #                     if word
    #                 ],
    #             )

    #             retry_count = 3
    #         except Exception as e:
    #             if retry_count == 2:
    #                 print(f"Error during inserting words: {e}, {e.__class__}")
    #                 logger.error(f"Error during inserting words: {e}, {e.__class__}")

    #             retry_count += 1
    #             await asyncio.sleep(random.randint(1, 10))
    #         else:
    #             print(f"Inserted {len(words)} words for {url}")
    #             logger.info(f"Inserted {len(words)} words for {url}")
