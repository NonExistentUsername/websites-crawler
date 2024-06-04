import asyncio

from crawler import *

if __name__ == "__main__":
    asyncio.run(move_to_redis())
    print("Done")
