### Create DB

```SQL
CREATE DATABASE site_index;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE websites (
    url VARCHAR(2048) PRIMARY KEY UNIQUE NOT NULL,
    word_count INT NOT NULL
);

CREATE TABLE keywords (
    word VARCHAR(45) PRIMARY KEY UNIQUE NOT NULL, 
    documents_containing_word BIGINT DEFAULT 0
);

CREATE TABLE website_keywords (
    id BIGSERIAL PRIMARY KEY,
    keyword_id VARCHAR(45) NOT NULL, website_id VARCHAR(2048) NOT NULL,
    occurrences INT NOT NULL,
    position INT NOT NULL
);

CREATE INDEX idx_keywords_name ON keywords (word);
CREATE INDEX website_keyword_id ON website_keywords (keyword_id);
```

```SQL
CREATE DATABASE sites;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE websites (
    url VARCHAR(2048) PRIMARY KEY UNIQUE NOT NULL,
    content TEXT NOT NULL
);
```

Taken from https://github.com/conaticus/search-engine-crawler/tree/dev