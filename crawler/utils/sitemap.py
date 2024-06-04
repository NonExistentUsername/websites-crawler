import logging
from typing import Set, Tuple
from xml.etree import ElementTree as ET

import requests  # type: ignore

from crawler.utils.url_loader import UrlLoader

logger = logging.getLogger()


class Sitemap:
    def __init__(self, url: str, url_loader: UrlLoader):
        self._url = url
        self._url_loader = url_loader

    def _check_if_xml(self, content: bytes) -> bool:
        return content.startswith(b"<?xml") or content.startswith(b"<urlset")

    def _check_if_txt(self, content: bytes) -> bool:
        return content.startswith(b"http")

    def _parse_content(self, content: bytes):
        urls: Set[str] = set()
        sitemap_index_urls: Set[str] = set()

        if self._check_if_xml(content):
            root = ET.fromstring(content)

            for child in root:
                if child.tag.endswith("url"):
                    for url in child:
                        if url.tag.endswith("loc") and url.text:
                            urls.add(url.text)

                elif child.tag.endswith("sitemap"):
                    for url in child:
                        if url.tag.endswith("loc") and url.text:
                            sitemap_index_urls.add(url.text)

        elif self._check_if_txt(content):
            txt_content = content.decode("utf-8")
            txt_content = txt_content.replace(" ", "\n")
            urls = set(txt_content.split("\n"))
        else:
            logger.warning(f"Unknown sitemap format: {self._url}")
            urls = set()

        return urls, sitemap_index_urls

    async def get_urls(self) -> Tuple[Set[str], Set[str]]:
        status_code, content = await self._url_loader(
            self._url, protected=False, raise_exceptions=False
        )

        return self._parse_content(content) if status_code == 200 else (set(), set())
