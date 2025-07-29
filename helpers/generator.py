import json
import struct
from typing import List

from config.database import get_sqlite_client
from utils.query import generate_sqlite_select_schema, generate_sqlite_delete_vector, generate_sqlite_insert_vector, \
    generate_sqlite_select_all


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

def sync_metadata(client):
    """Synchronize and re-create all metadata vector in sqlite database including delete previous metadata and add new metadata"""
    conn, cur = get_sqlite_client()
    try:
        query = generate_sqlite_select_schema()
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
    finally:
        cur.close()
        conn.close()

def all_used_table():
    """Get all tables used by this service """
    conn, cur = get_sqlite_client()
    try:
        query = generate_sqlite_select_all()
        cur.execute(query)
        rows = cur.fetchall()
        results = [dict(row) for row in rows]
        return results
    except Exception as e:
        raise Exception(f"Failed to sync metadata: {str(e)}") from e
    finally:
        cur.close()
        conn.close()

def klasifikasi_odometer(km):
    km = int(km)
    if km <= 10000:
        return 'Sangat Rendah'
    elif km <= 30000:
        return 'Rendah'
    elif km <= 60000:
        return 'Sedang'
    elif km <= 100000:
        return 'Tinggi'
    else:
        return 'Sangat Tinggi'
