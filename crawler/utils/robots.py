import logging
from urllib.robotparser import RobotFileParser

import requests  # type: ignore

from crawler.utils.url_loader import UrlLoader

logger = logging.getLogger()


class CustomRobotFileParser(RobotFileParser):
    def __init__(self, url, url_loader: UrlLoader):
        super().__init__(url)
        self._url_loader = url_loader

    async def read(self):
        """Reads the robots.txt URL and feeds it to the parser."""
        try:
            _, raw = await self._url_loader(
                self.url, protected=False, raise_exceptions=True
            )
        except requests.exceptions.RequestException as err:
            print(f"Error reading robots.txt: {err}")
            logger.error(f"Error reading robots.txt: {err}")

            if err.response.status_code in (401, 403):
                self.disallow_all = True
            elif err.response.status_code >= 400 and err.response.status_code < 500:
                self.allow_all = True
        except Exception as e:
            print(f"Error reading robots.txt: {e}")
            logger.error(f"Error reading robots.txt: {e}")
            self.disallow_all = True
        else:
            self.parse(raw.decode("utf-8").splitlines())
            print(f"Loaded robots.txt for {self.url}")
