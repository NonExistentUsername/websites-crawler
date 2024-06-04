import asyncio
import logging
from typing import Optional, Set, Tuple
from urllib.robotparser import RequestRate

from crawler.utils import CustomRobotFileParser, Sitemap, UrlLoader

logger = logging.getLogger()


class DomainManager:
    def __init__(self, user_agent: str, domain: str, url_loader: UrlLoader):
        self._user_agent = user_agent
        self._domain = domain
        self._url = f"https://{domain}"
        self._url_loader = url_loader

        self._urls: Optional[Set[str]] = None
        self._robots: Optional[CustomRobotFileParser] = None

    def _get_url_with_domain(self, url: str) -> str:
        return f"{self._url}{url}" if url.startswith("/") else url

    async def get_robots(self) -> CustomRobotFileParser:
        robots = CustomRobotFileParser(f"{self._url}/robots.txt", self._url_loader)
        await robots.read()
        return robots

    async def _get_urls_from_sitemap(
        self, sitemap_url: str
    ) -> Tuple[Set[str], Set[str]]:
        sitemap: Sitemap = Sitemap(
            self._get_url_with_domain(sitemap_url), self._url_loader
        )
        urls, sitemap_index_urls = await sitemap.get_urls()
        return urls, sitemap_index_urls

    async def _get_urls(self) -> Tuple[Set[str], Set[str]]:
        urls: Set[str] = {f"{self._url}/"}
        sitemap_index_urls: Set[str] = set()

        tasks = [
            asyncio.create_task(self._get_urls_from_sitemap(sitemap))
            for sitemap in await self.sitemaps
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            urls.update(result[0])
            sitemap_index_urls.update(result[1])

        return urls, sitemap_index_urls

    async def _filter(self, urls: Set[str], check_can_fetch=False):
        robots = await self.robots
        urls_filtered = set()
        for url in urls:
            url = self._get_url_with_domain(url)

            if not url.startswith(f"http://{self._domain}") and not url.startswith(
                f"https://{self._domain}"
            ):
                continue

            t = url.replace(f"https://{self._domain}", "").replace(
                f"http://{self._domain}", ""
            )
            if check_can_fetch and not robots.can_fetch(self._user_agent, t):
                continue

            urls_filtered.add(url)

        return urls_filtered

    @property
    async def robots(self) -> CustomRobotFileParser:
        if self._robots is None:
            self._robots = await self.get_robots()

        return self._robots

    @property
    async def sitemaps(self):
        robots = await self.robots
        sitemaps = robots.site_maps() or []

        default_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap/sitemap.xml",
            "/sitemap-index.xml",
            "/sitemap.php",
            "/sitemap.txt",
            "/sitemap.xml.gz",
            "/sitemap/",
            "/sitemap/sitemap.xml",
            "/sitemapindex.xml",
            "/sitemap/index.xml",
            "/sitemap1.xml",
        ]
        for path in default_paths:
            if path not in sitemaps:
                sitemaps.append(path)

        return sitemaps

    @property
    async def urls(self):
        if self._urls is None:
            urls, sitemap_index_urls = await self._get_urls()
            sitemap_index_urls = await self._filter(
                sitemap_index_urls, check_can_fetch=False
            )
            self._urls = await self._filter(urls, check_can_fetch=True)

            while len(sitemap_index_urls) > 0:
                tasks = [
                    asyncio.create_task(self._get_urls_from_sitemap(sitemap))
                    for sitemap in sitemap_index_urls
                ]
                urls = set()
                sitemap_index_urls = set()

                results = await asyncio.gather(*tasks)
                for result in results:
                    urls.update(result[0])
                    sitemap_index_urls.update(result[1])

                self._urls.update(await self._filter(urls))

        return self._urls

    @property
    async def delay(self) -> float:
        try:
            robots = await self.robots
            return float(robots.crawl_delay(self._user_agent) or 0.0)
        except ValueError:
            return 0.0

    @property
    async def request_rate(self) -> Optional[RequestRate]:
        robots = await self.robots
        return robots.request_rate(self._user_agent) or None
