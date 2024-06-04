from typing import Dict

from random_header_generator import HeaderGenerator as HG  # type: ignore


class HeaderGenerator:
    def __init__(self):
        self._generator = HG()

    def __call__(self) -> Dict:
        return dict(self._generator())
