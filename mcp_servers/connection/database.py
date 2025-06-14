import sqlite3

import oracledb
import psycopg2
import sqlite_vec
from dotenv import dotenv_values
from psycopg2.extras import RealDictCursor


def get_env():
    return dotenv_values(".env")


def get_sqlite_client():
    """Get or create the global Sqlite3 client instance"""
    environment = get_env()
    sqlite_client = sqlite3.connect(environment.get('SQLITE_DATABASE'))
    sqlite_client.row_factory = sqlite3.Row
    sqlite_client.execute("PRAGMA journal_mode=WAL")
    sqlite_cursor = sqlite_client.cursor()
    sqlite_client.enable_load_extension(True)
    sqlite_vec.load(sqlite_client)
    sqlite_client.enable_load_extension(False)
    return sqlite_client, sqlite_cursor


def get_oracle_client():
    """Get or create the global Oracle client instance"""
    environment = get_env()
    oracle_client = oracledb.connect(
        user=environment.get('ORACLE_USER'),
        password=environment.get('ORACLE_PASSWORD'),
        dsn=environment.get('ORACLE_DSN'),  # e.g., "localhost/XEPDB1"
    )
    oracle_cursor = oracle_client.cursor()
    return oracle_client, oracle_cursor

def get_postgres_client():
    """Get or create the global Postgre client instance"""
    environment = get_env()
    postgres_client = psycopg2.connect(
        dbname=environment.get('POSTGRES_DB'),
        user=environment.get('POSTGRES_USER'),
        password=environment.get('POSTGRES_PASSWORD'),
        host=environment.get('POSTGRES_HOST'),
        port=environment.get('POSTGRES_PORT')
    )
    postgres_cursor = postgres_client.cursor(cursor_factory=RealDictCursor)
    return postgres_client, postgres_cursor