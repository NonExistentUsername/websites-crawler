import asyncio

import aiopg
import psycopg2  # type: ignore

from crawler.constants import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


class DB:
    def __init__(self, host, port, user, password, database):
        # disable autocommit
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )

    def insert(self, table, columns, values):
        with self.conn.cursor() as cursor:
            try:
                cursor.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(values))})",
                    values,
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise e

    def insertMany(self, table, columns, values):
        with self.conn.cursor() as cursor:
            try:
                cursor.executemany(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})",
                    values,
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise e

    def insertManyOrUpdate(
        self,
        table,
        columns,
        values,
        conflict_columns,
    ):
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) ON CONFLICT ({', '.join(conflict_columns)}) DO NOTHING"

        with self.conn.cursor() as cursor:
            try:
                cursor.executemany(query, values)
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                raise e

    def select(self, table, columns, where=None):
        query = f"SELECT {', '.join(columns)} FROM {table}"
        if where:
            query += f" WHERE {where}"

        with self.conn.cursor() as cursor:
            try:
                cursor.execute(query)
                return cursor.fetchall()
            except Exception as e:
                raise e

    def close(self):
        self.conn.close()


class AsyncDB:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

        self.dsn = (
            f"dbname={database} user={user} password={password} host={host} port={port}"
        )

    async def insert(self, table, columns, values):
        async with aiopg.create_pool(self.dsn) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(values))})",
                        values,
                    )

    def _insertMany(self, table, columns, values):
        db = DB(self.host, self.port, self.user, self.password, self.database)
        db.insertMany(table, columns, values)
        db.close()

    async def insertMany(self, table, columns, values):
        await asyncio.to_thread(self._insertMany, table, columns, values)

    def _insertManyOrUpdate(
        self,
        table,
        columns,
        values,
        conflict_columns,
    ):
        db = DB(self.host, self.port, self.user, self.password, self.database)
        db.insertManyOrUpdate(table, columns, values, conflict_columns)
        db.close()

    async def insertManyOrUpdate(
        self,
        table,
        columns,
        values,
        conflict_columns,
    ):
        await asyncio.to_thread(
            self._insertManyOrUpdate,
            table,
            columns,
            values,
            conflict_columns,
        )

    async def select(self, table, columns, where=None, limit=None, offset=None):
        query = f"SELECT {', '.join(columns)} FROM {table}"
        if where:
            query += f" WHERE {where}"
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        async with aiopg.create_pool(self.dsn) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    async with cursor.begin():
                        await cursor.execute(query)
                        return await cursor.fetchall()

    async def count(self, table):
        async with aiopg.create_pool(self.dsn) as pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    async with cursor.begin():
                        await cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        return await cursor.fetchone()

    def close(self):
        pass


def create_db() -> AsyncDB:
    return AsyncDB(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
