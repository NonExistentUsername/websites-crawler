from crawler.constants import CRAWL_AS_AGENT, PROXY, TIMEOUT
from crawler.crawler import Crawler
from crawler.utils.content_processor import ContentProcessor
from crawler.utils.headers_generator import HeaderGenerator
from crawler.utils.url_loader import UrlLoader


class CrawlerFactory:
    @staticmethod
    def create(domain: str):
        url_loader = UrlLoader(
            proxy=PROXY,
            timeout=TIMEOUT,
            headers_generator=HeaderGenerator(),
        )
        content_processor = ContentProcessor()
        return Crawler(domain, CRAWL_AS_AGENT, url_loader, content_processor)
