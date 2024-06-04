import os

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path, override=True)

ENABLE_AUTO_CRAWL = os.getenv("ENABLE_AUTO_CRAWL", "False") == "True"

CRAWL_AS_AGENT = os.getenv("CRAWL_AS_AGENT", "Googlebot")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 32))
PROXY = os.getenv("PROXY", None)
TIMEOUT = int(os.getenv("TIMEOUT", 10))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "postgres")

REDIS_DB = os.getenv("REDIS_DB", 0)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

REDIS_QUEUE_HOST = os.getenv("REDIS_QUEUE_HOST", REDIS_HOST)
REDIS_QUEUE_PORT = os.getenv("REDIS_QUEUE_PORT", REDIS_PORT)
REDIS_QUEUE_PASSWORD = os.getenv("REDIS_QUEUE_PASSWORD", REDIS_PASSWORD)
REDIS_QUEUE_NAME = os.getenv("REDIS_QUEUE_NAME", "queue")

ASYNC_CRAWLERS = int(os.getenv("ASYNC_CRAWLERS", 8))
