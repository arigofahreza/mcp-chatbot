import asyncio
import json
import re
from typing import List, Optional

from langchain_experimental.utilities import PythonREPL
from mcp.server.fastmcp import FastMCP

from config.configuration import Configuration
from config.database import get_sqlite_client, get_oracle_client, get_postgres_client
from models.base_model import Metadata
from utils.llm import create_llm_client
from utils.query import generate_sqlite_table, generate_sqlite_insert, generate_sqlite_select, generate_sqlite_update, \
    generate_sqlite_delete, generate_sqlite_vector, generate_sqlite_select_vector, generate_sqlite_select_by_id

mcp = FastMCP('unified')

# Global variables
_sqlite_client = None
_sqlite_cursor = None
_oracle_client = None
_oracle_cursor = None
_postgres_client = None
_postgres_cursor = None

def sqlite_client():
    """Get or create the global Sqlite3 client instance"""
    global _sqlite_client
    global _sqlite_cursor
    if _sqlite_client is None:
        _sqlite_client, _sqlite_cursor = get_sqlite_client()
        return _sqlite_client, _sqlite_cursor
    return _sqlite_client, _sqlite_cursor

def oracle_client():
    """Get or create the global Oracle client instance"""
    global _oracle_client
    global _oracle_cursor
    if _oracle_client is None:
        _oracle_client, _oracle_cursor = get_oracle_client()
    return _oracle_client, _oracle_cursor

def postgres_client():
    """Get or create the global Postgre client instance"""
    global _postgres_client
    global _postgres_cursor
    if _postgres_client is None:
        _postgres_client, _postgres_cursor = get_postgres_client()
    return _postgres_client, _postgres_cursor

def create_metadata_table():
    conn, cur = sqlite_client()
    try:
        query = generate_sqlite_table()
        cur.execute(query)
        conn.commit()
    except Exception as e:
        raise Exception(f"Failed to create metadatas table: {str(e)}") from e


def create_vector_table():
    conn, cur = sqlite_client()
    try:
        query = generate_sqlite_vector()
        cur.execute(query)
        conn.commit()
    except Exception as e:
        raise Exception(f"Failed to create vector table: {str(e)}") from e


##### Collection Tools #####

@mcp.tool()
async def metadata_create(
        table_name: str,
        database_type: str,
        description: str,
        metadatas: List[Metadata]
) -> str:
    """Create metadata table in the SQLite3 database.

        Args:
            table_name: Name of the table to create
            database_type: Type of database ("postgresql", "oracledb")
            description: A description of the table that provides an overview of the table itself.
            metadatas: List metadata which describes the schema and metadata of the table, including column names, data types, and descriptions of each column.

    """
    conn, cur = sqlite_client()
    try:
        query = generate_sqlite_insert()
        json_metadata = json.dumps([metadata.model_dump_json() for metadata in metadatas])
        cur.execute(query, (table_name, database_type, description, json_metadata))
        conn.commit()
        return f"Successfully insert data with name {table_name}"
    except Exception as e:
        raise Exception(f"Failed to insert metadata: {str(e)}") from e


async def metadata_get(
        limit: Optional[int] = 10,
        offset: Optional[int] = 0
) -> List[dict]:
    """List all metadata table names in the SQLite3 database with pagination support.

        Args:
            limit: Optional maximum number of metadata tables to return
            offset: Optional number of metadata tables to skip before returning results

        Returns:
            List of metadata tables
    """
    conn, cur = sqlite_client()
    try:
        query = generate_sqlite_select()
        cur.execute(query, (limit, offset))
        rows = cur.fetchall()
        print(rows)
        result = [dict(row) for row in rows]
        return result
    except Exception as e:
        raise Exception(f"Failed to select metadata: {str(e)}") from e


@mcp.tool()
async def metadata_update(
        table_name: str,
        database_type: str,
        description: str,
        metadata: Metadata
) -> str:
    """Create metadata table in the SQLite3 database.

        Args:
            table_name: Name of the table to update in
            database_type: Type of database ("postgresql", "oracledb")
            description: A new description of the table that provides an overview of the table itself. Provide previous value if no update
            metadata: A new metadata which describes the schema and metadata of the table, including column names, data types, and descriptions of each column. Provide previous value if no update

    """
    conn, cur = sqlite_client()
    try:
        query = generate_sqlite_update()
        json_metadata = metadata.model_dump_json()
        cur.execute(query, (table_name, database_type, description, json_metadata, table_name))
        conn.commit()
        return f"Successfully update data with name {table_name} | description {description} | metadata {json_metadata}"
    except Exception as e:
        raise Exception(f"Failed to update metadata: {str(e)}") from e


@mcp.tool()
async def metadata_delete(
        table_name: str
) -> str:
    """Delete metadata table names in the SQLite3 database .

        Args:
            table_name: Name of the table metadata to delete
    """
    conn, cur = sqlite_client()
    try:
        query = generate_sqlite_delete()
        cur.execute(query, table_name)
        conn.commit()
        return f"Successfully delete data with name {table_name}"
    except Exception as e:
        raise Exception(f"Failed to delete metadata: {str(e)}") from e

@mcp.tool()
async def relevant_table(prompt: str, provider: str) -> dict:
    """This tool is mandatory before executing *_data_get tool.
       Fetch the relevant table base on natural language query from user using vector

        Args:
            prompt: user input prompt as is, as a string
            provider: LLM provider type ("openai" or "ollama") auto generated from LLM

        Returns:
            a result metadata
    """
    client = create_llm_client(provider, Configuration())
    conn, cur = get_sqlite_client()
    try:
        query = generate_sqlite_select_vector()
        embedding_results = client.get_embedding_response([prompt])
        datas = embedding_results[0]
        rows = cur.execute(query, (datas[1],) ).fetchall()
        results = [dict(row) for row in rows]
        result = results[0]
        rowid = result.get('rowid')
        select_query = generate_sqlite_select_by_id()
        cur.execute(select_query, (rowid,))
        rows = cur.fetchall()
        metadata_results = [dict(row) for row in rows]
        return metadata_results[0]
    except Exception as e:
        raise Exception(f"Failed to get relevant table: {str(e)}") from e


@mcp.tool()
async def oracle_data_get(query: str) -> List[dict]:
    """Before executing this tool, need to execute relevant table tool.
       Fetch all the result from oracle database with provided query

        Args:
            query: SQL query that LLM generated from user input prompt

        Returns:
            List of result data
    """
    conn, cur = oracle_client()
    try:
        cur.execute(query.replace(';', ''))
        columns = [col[0] for col in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]
        return results
    except Exception as e:
        raise Exception(f"Failed to get data from oracle: {str(e)}") from e

@mcp.tool()
async def postgres_data_get(query: str) -> List[dict]:
    """Before executing this tool, need to execute relevant table tool.
       Fetch all the result from postgres database with provided query

        Args:
            query: SQL query that LLM generated from user input prompt

        Returns:
            List of result data
    """
    conn, cur = postgres_client()
    try:
        cur.execute(query.replace(';', ''))
        result = cur.fetchall()
        return [dict(row) for row in result]
    except Exception as e:
        raise Exception(f"Failed to get data from postgres: {str(e)}") from e

@mcp.tool()
async def chart_generator(python_code: str) -> str:
    """Creating chart based on python code and run into PythonREPL() function

        Args:
            python_code: Python code that contain matplotlib code generator and print(base64_string)

        Returns:
            base64_string of the image
    """
    try:
        python_repl = PythonREPL()
        cleaned_code = python_code.strip().replace('\\n', '\n')
        cleaned_code = re.sub(r'\\([\'"])', r'\1', cleaned_code)
        print(cleaned_code)
        output = await asyncio.to_thread(python_repl.run, cleaned_code)
        if output.endswith("\n"):
            output = output[:-1]
        return output
    except Exception as e:
        raise Exception(f"Failed to generate chart from code: {str(e)}") from e

def main():
    """Entry point for the Unified MCP server."""
    create_metadata_table()
    create_vector_table()
    print("Successfully Created all required tables")
    try:
        sqlite_client()
        oracle_client()
        postgres_client()
        print("Successfully initialized All clients")
    except Exception as e:
        print(f"Failed to initialize All client: {str(e)}")
        raise

    # Initialize and run the server
    print("Starting MCP server")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
