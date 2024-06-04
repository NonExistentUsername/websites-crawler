import asyncio
import enum
import time
from datetime import timedelta
from typing import Optional
from urllib.robotparser import RequestRate


class DelayController:
    def __init__(
        self,
        delay: float,
        request_rate: Optional[RequestRate],
    ):
        if delay < 0:
            raise ValueError("Delay must be a non-negative number")

        self._delay = delay
        self._request_rate = request_rate

        self._last_request_time = time.time()
        self._requests_per_unit = 0
        self._unit_start_time = time.time()

        if request_rate is not None:
            self._unit_distance = self._get_unit_distance()
        else:
            self._unit_distance = None

    def _get_unit_distance(self):
        return timedelta(seconds=self._request_rate.seconds)

    def step(self):
        self._last_request_time = time.time()
        self._requests_per_unit += 1
        self._check_unit()

    def _check_unit(self):
        if self._unit_start_time is None:
            self._unit_start_time = self._last_request_time
        elif (
            self._unit_distance
            and self._last_request_time - self._unit_start_time >= self._unit_distance
        ):
            self._requests_per_unit = 0
            self._unit_start_time = None

    async def wait(self):
        delay = timedelta(seconds=self._delay)

        if self._unit_distance and self._requests_per_unit >= self._request_rate:
            delay = max(
                delay, self._unit_start_time + self._unit_distance - time.time()
            )

        await asyncio.sleep(delay.total_seconds())
