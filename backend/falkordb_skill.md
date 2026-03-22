# FalkorDB Python Client — SKILL.md


***DISCLAIMER: THIS IS A SKILL MD FOR FALKORDB, BUT IF YOU NEED ANY DOUBLE CHECKING USE CONTEXT7 MCP FOR MORE INFORMATION ON WHAT YOU ARE SEEKING***

## Overview

FalkorDB is a property graph database powered by GraphBLAS sparse matrix linear algebra. It supports Cypher queries, native vector indexes (HNSW), full-text indexes (RediSearch), and async Python operation via `falkordb-py`. It is the primary graph store for FOSRA's codebase index.

---

## Installation

```bash
pip install falkordb
# or
uv add falkordb
```

Docker:
```bash
docker run --rm -p 6379:6379 falkordb/falkordb
```

---

## 1. Connection

### Synchronous

```python
from falkordb import FalkorDB

db = FalkorDB(host='localhost', port=6379)
# With auth:
db = FalkorDB(host='localhost', port=6379, username='default', password='secret')

g = db.select_graph('codebase')
```

### Asynchronous (preferred for FOSRA)

```python
import asyncio
from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

pool = BlockingConnectionPool(max_connections=16, timeout=None, decode_responses=True)
db = FalkorDB(connection_pool=pool)
g = db.select_graph('codebase')

result = await g.query('MATCH (n) RETURN n LIMIT 10')
await pool.aclose()
```

**CRITICAL**: Always use `falkordb.asyncio` in async contexts. Never mix sync and async clients.

---

## 2. Graph Operations

### Query (read/write)

```python
result = g.query("CREATE (:Function {name: 'parse_csv', language: 'python'})")
result = await g.query("CREATE (:Function {name: $name})", params={"name": "parse_csv"})
```

### Read-only Query

```python
result = g.ro_query("MATCH (f:Function) RETURN f.name LIMIT 10")
result = await g.ro_query("MATCH (f:Function) WHERE f.language = $lang RETURN f", params={"lang": "python"})
```

**RULE**: Always use `ro_query` for reads. It is safer and may use read replicas.

### Result Access

```python
result.result_set          # list[list[Any]] — row-major
result.result_set[0]       # first row
result.result_set[0][0]    # first cell of first row

# Iterating rows
for row in result.result_set:
    node = row[0]          # Node object
    print(node.properties) # dict of node properties
```

### Node/Relationship Objects

```python
node.id          # internal graph ID
node.labels      # list[str]
node.properties  # dict

rel.id
rel.type         # str, e.g. "CALLS"
rel.src_node
rel.dest_node
rel.properties
```

### Parameterized Queries (ALWAYS use params)

```python
# CORRECT — never interpolate user values into query strings
result = await g.query(
    "MATCH (f:Function {name: $name}) RETURN f",
    params={"name": function_name}
)

# WRONG — SQL injection equivalent
result = await g.query(f"MATCH (f:Function {{name: '{function_name}'}}) RETURN f")
```

### Graph Management

```python
g.delete()                     # delete entire graph
copy = g.copy('codebase_copy') # copy graph
db.list_graphs()               # list all graph names
```

---

## 3. Cypher Reference for FOSRA Schema

### MERGE (upsert — use for all ingestion)

```cypher
MERGE (f:Function {name: $name, language: $language})
ON CREATE SET
  f.signature = $signature,
  f.body = $body,
  f.docstring = $docstring,
  f.summary = $summary,
  f.chunk_id = $chunk_id
ON MATCH SET
  f.body = $body,
  f.summary = $summary
RETURN f
```

### CREATE relationship (after MERGE of both endpoints)

```cypher
MATCH (caller:Function {name: $caller_name})
MATCH (callee:Function {name: $callee_name})
MERGE (caller)-[:CALLS]->(callee)
```

### MATCH with WHERE

```cypher
MATCH (f:Function)<-[:CONTAINS]-(file:File)
WHERE f.language = $language
RETURN f, file.path AS file_path
LIMIT $limit
```

### OPTIONAL MATCH (for nullable relationships)

```cypher
MATCH (f:Function {name: $name})
OPTIONAL MATCH (f)-[:CALLS]->(dep:Function)
RETURN f, collect(dep.name) AS dependencies
```

### Variable-length paths

```cypher
MATCH (f:Function)-[:CALLS*1..2]->(dep:Function)
WHERE f.name = $name
RETURN dep.name, dep.summary
```

### DELETE node and relationships

```cypher
MATCH (f:Function {chunk_id: $chunk_id})
DETACH DELETE f
```

---

## 4. Vector Index

### Create

```python
# Nodes
g.query("""
    CREATE VECTOR INDEX FOR (f:Function) ON (f.summary_embedding)
    OPTIONS {dimension: 768, similarityFunction: 'cosine', M: 32, efConstruction: 200}
""")

g.query("""
    CREATE VECTOR INDEX FOR (f:Function) ON (f.signature_embedding)
    OPTIONS {dimension: 768, similarityFunction: 'cosine', M: 32, efConstruction: 200}
""")

g.query("""
    CREATE VECTOR INDEX FOR (f:File) ON (f.summary_embedding)
    OPTIONS {dimension: 768, similarityFunction: 'cosine', M: 16, efConstruction: 200}
""")
```

**Options:**
- `dimension` — REQUIRED. Must match embedding model output size exactly.
- `similarityFunction` — REQUIRED. `'cosine'` for normalized embeddings, `'euclidean'` for unnormalized.
- `M` — HNSW max connections per node. Default 16. Use 32 for higher recall.
- `efConstruction` — Candidates during build. Default 200. Higher = better quality, slower build.
- `efRuntime` — Candidates during query. Default 10. Increase if recall is low.

### Insert vectors

```python
embedding = encoder.encode("parse CSV file and return dataframe").tolist()

g.query(
    "CREATE (f:Function {name: $name, summary_embedding: vecf32($emb)})",
    params={"name": "process_csv", "emb": embedding}
)

# Or MERGE with SET:
g.query("""
    MERGE (f:Function {name: $name})
    SET f.summary_embedding = vecf32($emb)
""", params={"name": "process_csv", "emb": embedding})
```

**CRITICAL**: Always wrap float arrays in `vecf32()`. Never pass raw arrays without it.

### Query vector index

```python
query_embedding = encoder.encode("find function that parses csv").tolist()

result = await g.ro_query("""
    CALL db.idx.vector.queryNodes('Function', 'summary_embedding', $k, vecf32($emb))
    YIELD node, score
    WHERE score > $min_score
    MATCH (node)<-[:CONTAINS]-(file:File)
    OPTIONAL MATCH (node)-[:CALLS]->(dep:Function)
    RETURN node, file.path AS file_path, score, collect(dep.name) AS calls
    ORDER BY score DESC
    LIMIT $limit
""", params={
    "k": 10,
    "emb": query_embedding,
    "min_score": 0.75,
    "limit": 10,
})
```

Procedure signatures:
```
CALL db.idx.vector.queryNodes(label: STRING, attribute: STRING, k: INT, query: VECTOR)
YIELD node, score

CALL db.idx.vector.queryRelationships(relType: STRING, attribute: STRING, k: INT, query: VECTOR)
YIELD relationship, score
```

**NOTE**: Vector queries do NOT support additional property filters in the CALL clause itself. Apply `WHERE` after `YIELD`.

### Drop vector index

```python
g.query("DROP VECTOR INDEX FOR (f:Function) ON (f.summary_embedding)")
```

### List all indexes

```python
result = g.query("CALL db.indexes()")
for row in result.result_set:
    print(row)  # shows type: VECTOR | FULLTEXT | RANGE
```

---

## 5. Full-Text Index

### Create (node)

```python
# Single property
g.query("CALL db.idx.fulltext.createNodeIndex('Function', 'name')")

# Multiple properties
g.query("CALL db.idx.fulltext.createNodeIndex('Function', 'name', 'docstring')")

# With language options
g.query("CALL db.idx.fulltext.createNodeIndex({ label: 'Function', language: 'English', stopwords: [] }, 'name', 'docstring')")
```

### Query full-text index

```python
result = await g.ro_query("""
    CALL db.idx.fulltext.queryNodes('Function', $query_str)
    YIELD node, score
    RETURN node.name, node.docstring, score
    ORDER BY score DESC
    LIMIT 10
""", params={"query_str": "parse csv"})
```

**Query syntax:**
- `'word1 word2'` — AND (both must match)
- `'word1|word2'` — OR
- `'-word'` — NOT
- `'word*'` — prefix match
- `'%word%1'` — fuzzy (Levenshtein distance 1)

### Combine full-text + graph traversal

```python
result = await g.ro_query("""
    CALL db.idx.fulltext.queryNodes('Function', $query_str)
    YIELD node AS func, score
    WHERE func.language = $language
    MATCH (func)<-[:CONTAINS]-(file:File)
    RETURN func, file.path, score
    ORDER BY score DESC
    LIMIT $limit
""", params={"query_str": "authenticate user", "language": "python", "limit": 10})
```

### Drop full-text index

```python
g.query("CALL db.idx.fulltext.drop('Function')")
```

---

## 6. FOSRA-Specific Patterns

### Upsert File + Functions (ingestion)

```python
async def upsert_file(g, file_path: str, language: str, summary: str, summary_emb: list[float]):
    await g.query("""
        MERGE (f:File {path: $path})
        SET f.language = $language,
            f.summary = $summary,
            f.summary_embedding = vecf32($emb)
    """, params={"path": file_path, "language": language, "summary": summary, "emb": summary_emb})

async def upsert_function(g, func: dict, file_path: str):
    await g.query("""
        MERGE (fn:Function {name: $name, language: $language})
        ON CREATE SET
            fn.signature = $signature,
            fn.body = $body,
            fn.docstring = $docstring,
            fn.summary = $summary,
            fn.chunk_id = $chunk_id,
            fn.signature_embedding = vecf32($sig_emb),
            fn.summary_embedding = vecf32($sum_emb)
        ON MATCH SET
            fn.body = $body,
            fn.summary = $summary,
            fn.summary_embedding = vecf32($sum_emb)
        WITH fn
        MATCH (file:File {path: $file_path})
        MERGE (file)-[:CONTAINS]->(fn)
    """, params={**func, "file_path": file_path})
```

### Semantic entry + dependency traversal (retrieval)

```python
async def search_graph_semantic(g, query_embedding: list[float], k: int = 10, min_score: float = 0.75):
    return await g.ro_query("""
        CALL db.idx.vector.queryNodes('Function', 'summary_embedding', $k, vecf32($emb))
        YIELD node AS func, score
        WHERE score > $min_score
        MATCH (func)<-[:CONTAINS]-(file:File)
        OPTIONAL MATCH (func)-[:CALLS*1..2]->(dep:Function)
        OPTIONAL MATCH (dep)<-[:CONTAINS]-(dep_file:File)
        RETURN func, file.path AS file_path, score,
               collect(DISTINCT {name: dep.name, body: dep.body, file: dep_file.path}) AS dependencies
        ORDER BY score DESC
        LIMIT $limit
    """, params={"emb": query_embedding, "k": k, "min_score": min_score, "limit": k})
```

### Delete source from graph (on re-ingest or delete)

```python
async def delete_source(g, file_path: str):
    # Detach delete all nodes belonging to this file
    await g.query("""
        MATCH (file:File {path: $path})
        OPTIONAL MATCH (file)-[:CONTAINS]->(n)
        DETACH DELETE file, n
    """, params={"path": file_path})
```

### Concurrent queries (async)

```python
results = await asyncio.gather(
    g.ro_query("CALL db.idx.vector.queryNodes('Function', 'summary_embedding', 10, vecf32($emb)) YIELD node, score RETURN node, score", params={"emb": emb}),
    g.ro_query("CALL db.idx.fulltext.queryNodes('Function', $q) YIELD node, score RETURN node, score", params={"q": query_str}),
)
```

---

## 7. Known Limitations

- Vector queries do NOT support filtering inside `CALL db.idx.vector.queryNodes`. Apply `WHERE` after `YIELD`.
- No joins across separate graphs. Library codebases must be in the same graph with a `library` property if co-indexed.
- `vecf32()` dimension must match index dimension exactly. Mismatches raise an error at query time, not at index creation time.
- HNSW is approximate — recall is not 100%. Tune `efRuntime` (via OPTIONS) if precision is insufficient.
- Full-text index uses TF-IDF scoring, not BM25. Do not mix full-text scores with vector scores directly; re-rank with cross-encoder after merging.

---

## 8. Index Setup Script (FOSRA)

```python
async def setup_indexes(g):
    """Run once at project initialization."""
    index_queries = [
        # Vector indexes
        "CREATE VECTOR INDEX FOR (f:Function) ON (f.summary_embedding) OPTIONS {dimension: 768, similarityFunction: 'cosine', M: 32, efConstruction: 200}",
        "CREATE VECTOR INDEX FOR (f:Function) ON (f.signature_embedding) OPTIONS {dimension: 768, similarityFunction: 'cosine', M: 32, efConstruction: 200}",
        "CREATE VECTOR INDEX FOR (f:File) ON (f.summary_embedding) OPTIONS {dimension: 768, similarityFunction: 'cosine', M: 16, efConstruction: 200}",
        # Full-text indexes
        "CALL db.idx.fulltext.createNodeIndex('Function', 'name', 'docstring')",
        "CALL db.idx.fulltext.createNodeIndex('Class', 'name', 'docstring')",
        "CALL db.idx.fulltext.createNodeIndex('File', 'path')",
    ]
    for q in index_queries:
        try:
            await g.query(q)
        except Exception as e:
            if "already exists" in str(e).lower():
                pass  # idempotent
            else:
                raise
```
