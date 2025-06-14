import json
import struct
from typing import List

from mcp_servers.connection.database import get_sqlite_client
from mcp_servers.utils.query import generate_sqlite_select_all, generate_sqlite_delete_vector, \
    generate_sqlite_insert_vector, generate_sqlite_select_vector, generate_sqlite_select_by_id


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

def sync_metadata(client):
    """Synchronize and re-create all metadata vector in sqlite database including delete previous metadata and add new metadata"""
    conn, cur = get_sqlite_client()
    try:
        query = generate_sqlite_select_all()
        cur.execute(query)
        rows = cur.fetchall()
        results = [json.dumps(dict(row)) for row in rows]
        embedding_results = client.get_embedding_response(results)
        delete_query = generate_sqlite_delete_vector()
        cur.execute(delete_query)
        conn.commit()
        insert_query = generate_sqlite_insert_vector()
        cur.executemany(insert_query, embedding_results)
        conn.commit()
        return f"Successfully sync metadata"
    except Exception as e:
        raise Exception(f"Failed to sync metadata: {str(e)}") from e

def get_relevant_tables(prompt: str, client):
    """Tool to get relevant table to get schema and metadata from vector db to improve context.

        Args:
            prompt: Prompt from user input as an identifier for metadata table
            client: LLMClient for generating embedding for vector query
        Returns:
            A metadata info from table
    """
    conn, cur = get_sqlite_client()
    try:
        query = generate_sqlite_select_vector()
        embedding_results = client.get_embedding_response([prompt])
        datas = embedding_results[0]
        rows = cur.execute(query, datas[1]).fetchall()
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