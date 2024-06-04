import asyncio
import logging
import random
from typing import Dict, Optional, Tuple

import aiohttp

from crawler.utils.headers_generator import HeaderGenerator

logger = logging.getLogger()


class UrlLoader:
    def __init__(
        self,
        proxy: Optional[str] = None,
        timeout: int = 10,
        headers_generator: Optional[HeaderGenerator] = None,
    ):
        self._proxy = proxy
        self._timeout = timeout  # seconds
        self._headers_generator = headers_generator or HeaderGenerator()

    def __is_proxy_error(self, response):
        return response.status == 403 and self._proxy

    async def __call__(
        self, url, raise_exceptions=True, protected=True
    ) -> Tuple[int, bytes]:
        args = (url,)
        kwargs: Dict = {
            "timeout": self._timeout,
            "headers": self._headers_generator(),
            "ssl": False,
        }

        if protected and self._proxy:
            kwargs["proxy"] = self._proxy

        async with aiohttp.ClientSession() as session:
            retry_step = 0
            while retry_step < 3:
                try:
                    async with session.get(*args, **kwargs) as response:
                        if raise_exceptions or (
                            self.__is_proxy_error(response) and retry_step == 2
                        ):  # raise if required or if not last retry, so we can try again
                            response.raise_for_status()

                        result = (
                            await response.read() if response.status == 200 else b""
                        )
                        status_code = response.status

                        return status_code, result
                except Exception as e:
                    if raise_exceptions:
                        raise e

                    retry_step += 1
                    await asyncio.sleep(random.randint(1, 10))

            return 404, b""
