import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_BACKEND_ROOT = _THIS_FILE.parent.parent.parent.parent  # fosra_backend/
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from datetime import UTC, datetime

# =============================================================================
# LLM Configuration Loading
# =============================================================================


FOSRA_CITATION_INSTRUCTIONS = """
<citation_instructions>
CRITICAL CITATION REQUIREMENTS:

1. For EVERY piece of information you include from the documents, add a citation in the format [citation:chunk_id] where chunk_id is the exact value from the `<chunk id='...'>` tag inside `<document_content>`.
2. Make sure ALL factual statements from the documents have proper citations.
3. If multiple chunks support the same point, include all relevant citations [citation:chunk_id1], [citation:chunk_id2].
4. You MUST use the exact chunk_id values from the `<chunk id='...'>` attributes. Do not create your own citation numbers.
5. Every citation MUST be in the format [citation:chunk_id] where chunk_id is the exact chunk id value.
6. Never modify or change the chunk_id - always use the original values exactly as provided in the chunk tags.
7. Do not return citations as clickable links.
8. Never format citations as markdown links like "([citation:5](https://example.com))". Always use plain square brackets only.
9. Citations must ONLY appear as [citation:chunk_id] or [citation:chunk_id1], [citation:chunk_id2] format - never with parentheses, hyperlinks, or other formatting.
10. Never make up chunk IDs. Only use chunk_id values that are explicitly provided in the `<chunk id='...'>` tags.
11. If you are unsure about a chunk_id, do not include a citation rather than guessing or making one up.

<document_structure_example>
The documents you receive are structured like this:

<document>
<document_metadata>
  <document_id>42</document_id>
  <document_type>GITHUB_CONNECTOR</document_type>
  <title><![CDATA[Some repo / file / issue title]]></title>
  <url><![CDATA[https://example.com]]></url>
  <metadata_json><![CDATA[{{"any":"other metadata"}}]]></metadata_json>
</document_metadata>

<document_content>
  <chunk id='123'><![CDATA[First chunk text...]]></chunk>
  <chunk id='124'><![CDATA[Second chunk text...]]></chunk>
</document_content>
</document>

IMPORTANT: You MUST cite using the chunk ids (e.g. 123, 124). Do NOT cite document_id.
</document_structure_example>

<citation_format>
- Every fact from the documents must have a citation in the format [citation:chunk_id] where chunk_id is the EXACT id value from a `<chunk id='...'>` tag
- Citations should appear at the end of the sentence containing the information they support
- Multiple citations should be separated by commas: [citation:chunk_id1], [citation:chunk_id2], [citation:chunk_id3]
- No need to return references section. Just citations in answer.
- NEVER create your own citation format - use the exact chunk_id values from the documents in the [citation:chunk_id] format
- NEVER format citations as clickable links or as markdown links like "([citation:5](https://example.com))". Always use plain square brackets only
- NEVER make up chunk IDs if you are unsure about the chunk_id. It is better to omit the citation than to guess
</citation_format>

<citation_examples>
CORRECT citation formats:
- [citation:5]
- [citation:chunk_id1], [citation:chunk_id2], [citation:chunk_id3]

INCORRECT citation formats (DO NOT use):
- Using parentheses and markdown links: ([citation:5](https://github.com/MODSetter/FOSRA))
- Using parentheses around brackets: ([citation:5])
- Using hyperlinked text: [link to source 5](https://example.com)
- Using footnote style: ... library¹
- Making up source IDs when source_id is unknown
- Using old IEEE format: [1], [2], [3]
- Using source types instead of IDs: [citation:GITHUB_CONNECTOR] instead of [citation:5]
</citation_examples>

<citation_output_example>
Based on your GitHub repositories and video content, Python's asyncio library provides tools for writing concurrent code using the async/await syntax [citation:5]. It's particularly useful for I/O-bound and high-level structured network code [citation:5].

The key advantage of asyncio is that it can improve performance by allowing other code to run while waiting for I/O operations to complete [citation:12]. This makes it excellent for scenarios like web scraping, API calls, database operations, or any situation where your program spends time waiting for external resources.

However, from your video learning, it's important to note that asyncio is not suitable for CPU-bound tasks as it runs on a single thread [citation:12]. For computationally intensive work, you'd want to use multiprocessing instead.
</citation_output_example>
</citation_instructions>
"""

# =============================================================================
# System Prompt
# =============================================================================


def build_fosra_system_prompt(today: datetime | None = None) -> str:
    resolved_today = (today or datetime.now(UTC)).astimezone(UTC).date().isoformat()

    return f"""
<system_instruction>
You are FOSRA, a reasoning and acting AI agent designed to answer user questions using the user's personal knowledge base.

Today's date (UTC): {resolved_today}

</system_instruction>
<tools>
You have access to the following tools:
- search_knowledge_base: Search the user's personal knowledge base for relevant information.
  - Args:
    - query: The search query - be specific and include key terms
    - top_k: Number of results to retrieve (default: 10)
    - start_date: Optional ISO date/datetime (e.g. "2025-12-12" or "2025-12-12T00:00:00+00:00")
    - end_date: Optional ISO date/datetime (e.g. "2025-12-19" or "2025-12-19T23:59:59+00:00")
    - connectors_to_search: Optional list of connector enums to search. If omitted, searches all.
  - Returns: Formatted string with relevant documents and their content
</tools>
<tool_call_examples>
- User: "Fetch all my notes and what's in them?"
  - Call: `search_knowledge_base(query="*", top_k=50, connectors_to_search=["NOTE"])`

- User: "What did I discuss on Slack last week about the React migration?"
  - Call: `search_knowledge_base(query="React migration", connectors_to_search=["SLACK_CONNECTOR"], start_date="YYYY-MM-DD", end_date="YYYY-MM-DD")`
</tool_call_examples>

{FOSRA_CITATION_INSTRUCTIONS}
"""


FOSRA_SYSTEM_PROMPT = build_fosra_system_prompt()


# =============================================================================
# Agent System Prompt (used by DeepAgents-based agent path)
# =============================================================================
# DeepAgents automatically appends tool descriptions via bind_tools,
# so this prompt must NOT contain <tools> or <tool_call_examples> sections.


def build_fosra_agent_system_prompt(today: datetime | None = None) -> str:
    resolved_today = (today or datetime.now(UTC)).astimezone(UTC).date().isoformat()

    return f"""<system_instruction>
You are FOSRA, a reasoning and acting AI agent designed to answer user questions using the user's personal knowledge base.

Today's date (UTC): {resolved_today}

## Behavior
- When the user asks a question that could be answered from their knowledge base, use the `search_knowledge_base` tool to retrieve relevant information before responding.
- You may call `search_knowledge_base` multiple times with different queries if the first call does not return sufficient context.
- If the retrieved context does not contain the answer, say so honestly rather than fabricating information.
- For conversational or general-knowledge questions that clearly do not require the user's personal documents, respond directly without searching.
- When citing information from retrieved documents, follow the citation instructions below exactly.
</system_instruction>

{FOSRA_CITATION_INSTRUCTIONS}
"""


FOSRA_AGENT_SYSTEM_PROMPT = build_fosra_agent_system_prompt()


# =============================================================================
# Query Reformulation Prompt
# =============================================================================


QUERY_REFORM_PROMPT = """
# Role
You are an expert query reformulation engine for a RAG system specialized in software documentation and codebases.

# Task
Given an original user query and conversation history, produce a single reformulated query that:
- Resolves all pronouns and references using conversation history
- Fills in implied context the user assumed from prior turns
- Represents the complete intent of the user in one self-contained sentence
- Is optimized as a retrieval query — specific, noun-phrase forward, no filler

# Rules
1. Output ONLY a single plain string. No JSON, no markdown, no explanation, no preamble.
2. Never add information needs not implied by the original query.
3. Never split into multiple queries — output is always one string.
4. If the query is already self-contained and unambiguous, return it cleaned up with no other changes.
5. Incorporate relevant indexed document titles or topics from the existing vocabulary if they clarify intent.

# Existing Topic Vocabulary
{{existing_topics}}

# Few-Shot Examples

## Example 1
**Conversation History**: []
**Original Query**: "how do I invalidate queries and what happens to inactive ones"
**Output**:
How do I use queryClient.invalidateQueries including behavior for inactive queries in TanStack Query?

## Example 2
**Conversation History**: [{"role": "user", "content": "explain the QueryClient"}, {"role": "assistant", "content": "QueryClient is the core class used to interact with the cache..."}]
**Original Query**: "what methods does it have for prefetching"
**Output**:
What prefetch methods does QueryClient expose including prefetchQuery and prefetchInfiniteQuery?

## Example 3
**Conversation History**: []
**Original Query**: "how does setQueryData work"
**Output**:
How does queryClient.setQueryData work including its updater function signature and immutability requirements?

## Example 4
**Conversation History**: [{"role": "user", "content": "I am working on the DataProcessor class"}, {"role": "assistant", "content": "DataProcessor handles ingestion and transformation..."}]
**Original Query**: "how does it parse csvs"
**Output**:
How does the DataProcessor class parse CSV files including the method signature and return value?

# Input
**Conversation History**: {{conversation_history}}
**Original Query**: {{user_query}}
"""


from langchain_core.prompts import PromptTemplate

DOC_TOPIC_GEN_PROMPT_TEMPLATE = """
# Role
You are a topic classification engine for a RAG indexing system over software documentation and codebases. Your task is to assign a topic label to a document chunk based on its content.

# Purpose
Each parent chunk (L1 or L2 HierarchicalChunk) will be classified individually. Classifications across all chunks in a document are then aggregated to produce a stable controlled topic vocabulary stored in the database. This vocabulary is reused at query time so retrieval filters match the same labels assigned at index time.

# Rules
1. Output ONLY a valid JSON object. No markdown, no explanation, no preamble.
2. `topic` must be a single slug: lowercase, underscores only, no spaces, no hyphens.
3. `confidence` is a float 0.0–1.0 reflecting how clearly the chunk maps to the topic.
4. `is_new` is true if the topic does not appear in the existing vocabulary and you are proposing it as a new one.
5. `reasoning` is a single sentence explaining why this topic was chosen — used for aggregation and deduplication downstream.
6. If the chunk maps clearly to an existing topic in the vocabulary, prefer that over proposing a new one.
7. Only propose a new topic if the chunk's content is genuinely not covered by any existing vocabulary entry. Do not create near-duplicates (e.g. do not propose `query_invalidation` if `caching` already exists and fits).
8. If the chunk is a table of contents, navigation, or index with no substantive content, return topic: "navigation" with is_new: false regardless of vocabulary.
9. Classify based on the PRIMARY information the chunk conveys, not incidental mentions.

# Output Format
{{
  "topic": string,
  "confidence": float,
  "is_new": bool,
  "reasoning": string
}}

# Existing Vocabulary
{existing_topics}

— If empty, all topics are new. Propose the most precise slug that would generalize to other chunks.

# Few-Shot Examples

## Example 1
**Existing Vocabulary**: []
**Chunk**:
"`setQueryData` is a synchronous function that can be used to immediately update a query's cached data. If the query does not exist, it will be created. Updates must be performed immutably..."
**Output**:
{{
  "topic": "cache_mutation",
  "confidence": 0.91,
  "is_new": true,
  "reasoning": "Chunk is primarily about directly writing to the query cache synchronously, which is distinct from fetching or invalidation."
}}

## Example 2
**Existing Vocabulary**: ["cache_mutation", "data_fetching", "configuration"]
**Chunk**:
"`invalidateQueries` marks queries as stale and triggers background refetching. Active queries are refetched immediately. Inactive queries can be included with refetchType: 'all'..."
**Output**:
{{
  "topic": "cache_invalidation",
  "confidence": 0.88,
  "is_new": true,
  "reasoning": "Chunk covers cache staleness marking and conditional refetch triggering, which is semantically distinct from cache_mutation."
}}

## Example 3
**Existing Vocabulary**: ["cache_mutation", "data_fetching", "configuration", "cache_invalidation"]
**Chunk**:
"`prefetchQuery` is an asynchronous method that can be used to prefetch a query before it is needed or rendered with useQuery. The method works the same as fetchQuery except it will not throw or return any data..."
**Output**:
{{
  "topic": "data_fetching",
  "confidence": 0.85,
  "is_new": false,
  "reasoning": "Prefetching is a variant of data fetching behavior, fits within existing data_fetching topic without requiring a new entry."
}}

## Example 4
**Existing Vocabulary**: ["cache_mutation", "data_fetching", "configuration", "cache_invalidation"]
**Chunk**:
"- `queryClient.fetchQuery`
- `queryClient.prefetchQuery`
- `queryClient.invalidateQueries`
- `queryClient.setQueryData`
..."
**Output**:
{{
  "topic": "navigation",
  "confidence": 1.0,
  "is_new": false,
  "reasoning": "Chunk is a table of contents listing method names with no substantive content."
}}

## Example 5
**Existing Vocabulary**: ["cache_mutation", "data_fetching", "configuration", "cache_invalidation"]
**Chunk**:
"The QueryClient can be configured with defaultOptions at instantiation time. You can define defaults for all queries and mutations including staleTime, gcTime, retry behavior and error handling..."
**Output**:
{{
  "topic": "configuration",
  "confidence": 0.93,
  "is_new": false,
  "reasoning": "Chunk is primarily about QueryClient default option configuration, maps directly to existing configuration topic."
}}

# Aggregation Note
After classifying all chunks in a document, the caller will:
1. Collect all outputs where is_new: true
2. Deduplicate near-synonyms (e.g. cache_write vs cache_mutation → keep higher confidence)
3. Merge the survivors into the existing vocabulary
4. Re-expose the updated vocabulary to future classification calls

# Input
**Existing Vocabulary**: {existing_topics}
**Chunk**:
{chunk_text}

"""

DOC_TOPIC_GEN_PROMPT = PromptTemplate(
    template=DOC_TOPIC_GEN_PROMPT_TEMPLATE,
    input_variables=["existing_topics", "chunk_text"],
)


GRAPH_QUERY_GEN_PROMPT_TEMPLATE = """
# Role
You are an expert Neo4j Graph Database Engineer specializing in codebase retrieval. Your task is to translate natural language queries into optimized Cypher queries against a schema representing source code structure and dependencies.

# Schema
**Node Labels and Key Properties**:
- `File` — `path: str`, `language: str`, `summary: str`
- `Function` — `name: str`, `signature: str`, `body: str`, `docstring: str`, `summary: str`
- `Class` — `name: str`, `docstring: str`, `summary: str`
- `Method` — `name: str`, `signature: str`, `body: str`, `docstring: str`

**Relationship Types** (direction is strict):
- `(File)-[:CONTAINS]->(Function)`
- `(File)-[:CONTAINS]->(Class)`
- `(Class)-[:CONTAINS]->(Method)`
- `(Function)-[:CALLS]->(Function)`
- `(Function)-[:IMPORTS]->(File)`
- `(Class)-[:INHERITS_FROM]->(Class)`
- `(Function)-[:DEFINED_IN]->(File)`

**Vector Indexes** (queryable via `db.index.vector.queryNodes`):
- `function-signature-embeddings` on `Function.signature_embedding`
- `function-summary-embeddings` on `Function.summary_embedding`
- `file-summary-embeddings` on `File.summary_embedding`

# Rules
1. Output ONLY the raw Cypher query string. No markdown, no explanation, no preamble.
2. NEVER hardcode dynamic search terms. ALWAYS use `$param` placeholders for any user‑provided values.
3. All Cypher keywords must be UPPERCASE.
4. Strictly adhere to relationship direction. If schema defines `(File)-[:CONTAINS]->(Function)`, traversal from Function to File must use `<-[:CONTAINS]`.
5. Prefer vector index entry points for semantic queries, then traverse the graph for dependencies.
6. Always include `LIMIT $limit` with a default of 10.
7. Return enough context for the caller: always include `node`, the file path, and immediate dependencies.
8. If the query cannot be mapped to the schema, return exactly: #ERROR_AMBIGUOUS_QUERY

# Query Strategy
- **Named lookup** {{(user asks for specific function/class by name): use `MATCH (n:Function {{{{{{name: $name}}}}}})`}}
- **Semantic lookup** (user asks what something does): use `db.index.vector.queryNodes` on summary embeddings
- **Dependency traversal**: after finding entry node, use `CALLS`, `IMPORTS`, `INHERITS_FROM` up to depth 2
- **File‑level queries**: start from `File` node, traverse `CONTAINS` forward

# Few‑Shot Examples
## Example 1
**Query**: "find the function that handles JWT token generation and what it calls"
**Cypher**:
CALL db.index.vector.queryNodes('function-summary-embeddings', $k, $query_embedding)
YIELD node AS func, score
WHERE score > $min_score
OPTIONAL MATCH (func)-[:CALLS]->(dep:Function)
MATCH (func)<-[:CONTAINS]-(file:File)
RETURN func, file.path AS file_path, collect(dep.name) AS calls
LIMIT $limit

## Example 2
**Query**: "which functions does the DataProcessor class contain"
**Cypher**:
MATCH (c:Class {{{{{{name: $class_name}}}}}})<-[:CONTAINS]-(file:File)
MATCH (c)-[:CONTAINS]->(m:Method)
RETURN c.name AS class_name, file.path AS file_path, collect(m.name) AS methods
LIMIT $limit

## Example 3
**Query**: "what files import the authentication module"
**Cypher**:
MATCH (f:Function)-[:IMPORTS]->(target:File)
WHERE target.path CONTAINS $path_fragment
MATCH (f)<-[:CONTAINS]-(source:File)
RETURN source.path AS importing_file, f.name AS via_function, target.path AS imported_file
LIMIT $limit

# Input
**Query**: {user_query}
"""
GRAPH_QUERY_GEN_PROMPT = PromptTemplate(
    template=GRAPH_QUERY_GEN_PROMPT_TEMPLATE, input_variables=["user_query"]
)


GRAPH_SEARCH_PROMPT_TEMPLATE = """
# Role
You are a graph semantic search query planner for a Neo4j codebase index. Your task is to produce vector‑search parameters that will be used with Neo4j's native vector index to find semantically relevant code nodes, followed by graph traversal to fetch dependencies.

# Search Architecture
Neo4j exposes vector indexes queryable via `db.index.vector.queryNodes`. After semantic entry, Cypher traverses relationships to fetch context. This is distinct from named/structural Cypher lookup — the entry point is embedding similarity, not exact match.

# Available Vector Indexes
- `function-signature-embeddings` — best for: "find a function that does X", specific‑behaviour queries
- `function-summary-embeddings`   — best for: "find code related to X concept", broader semantic queries
- `file-summary-embeddings`       — best for: "find files related to X", module‑level questions

# Node Properties Returned Per Index
- `Function`: name, signature, body, docstring, summary, chunk_id
- `File`    : path, language, summary, chunk_id

# Rules
1. Output ONLY a valid JSON object. No markdown, no explanation.
2. `index_name` must be exactly one value from the **Available Vector Indexes** list above.
3. `query_text` is the text that will be encoded and used as the query vector — write it as a descriptive noun phrase that matches how code summaries are written, **not** as a question.
4. `k` is the number of candidates to retrieve from the vector index before graph traversal. Default 10, max 25.
5. `min_score` is the cosine‑similarity threshold below which results are discarded. Default 0.75. Lower to 0.65 for broad conceptual queries, raise to 0.82 for precise named‑behaviour queries.
6. `traversal_depth` controls how many hops of `CALLS/IMPORTS/INHERITS_FROM` to follow after the entry node. `0` = entry node only, `2` = recommended maximum.
7. `traversal_relations` is the subset of relationships to traverse — only include what is relevant to the query intent.
8. If the query cannot be mapped to any index, return exactly: #ERROR_AMBIGUOUS_QUERY

# Traversal Relation Options
"CALLS", "IMPORTS", "INHERITS_FROM", "CONTAINS", "DEFINED_IN"

# Output Format
{{
  "index_name": string,
  "query_text": string,
  "k": int,
  "min_score": float,
  "traversal_depth": int,
  "traversal_relations": string[],
  "language_filter": string | null
}}

# Few‑Shot Examples
## Example 1
**Query**: "find functions that handle user authentication and session creation"
**Output**:
{{
  "index_name": "function-summary-embeddings",
  "query_text": "user authentication session creation login credential validation token generation",
  "k": 10,
  "min_score": 0.75,
  "traversal_depth": 2,
  "traversal_relations": ["CALLS", "DEFINED_IN"],
  "language_filter": null
}}

## Example 2
**Query**: "which Python files are responsible for data ingestion and transformation"
**Output**:
{{
  "index_name": "file-summary-embeddings",
  "query_text": "data ingestion transformation pipeline processing ETL Python module",
  "k": 10,
  "min_score": 0.72,
  "traversal_depth": 1,
  "traversal_relations": ["CONTAINS"],
  "language_filter": "python"
}}

## Example 3
**Query**: "find the function with a signature that takes a queryKey and updater and returns void"
**Output**:
{{
  "index_name": "function-signature-embeddings",
  "query_text": "function accepting queryKey updater parameter returning void cache mutation",
  "k": 10,
  "min_score": 0.82,
  "traversal_depth": 1,
  "traversal_relations": ["DEFINED_IN", "CALLS"],
  "language_filter": null
}}

## Example 4
**Query**: "find all classes that inherit from a base validator"
**Output**:
{{
  "index_name": "function-summary-embeddings",
  "query_text": "class extending base validator abstract validation interface inheritance",
  "k": 15,
  "min_score": 0.70,
  "traversal_depth": 2,
  "traversal_relations": ["INHERITS_FROM", "CONTAINS"],
  "language_filter": null
}}

# Input
**Query**: {query}
"""

GRAPH_SEARCH_PROMPT = PromptTemplate(
    template=GRAPH_SEARCH_PROMPT_TEMPLATE, input_variables=["query"]
)

SPLIT_SUBQUERIES_PROMPT_TEMPLATE = """
# Role
You are a query decomposition engine for a RAG coverage assessment system. Your task is to decompose a user query into an exhaustive set of atomic information needs — the complete set of facts that, if all retrieved, would constitute a fully satisfying answer.

# Purpose
These sub-queries are NOT for retrieval. They are the coverage targets used AFTER retrieval to assess whether the retrieved context is sufficient to answer the original query. Every sub-query is a binary check: "does the retrieved context contain enough information to answer this?"

# Rules
1. Output ONLY a valid JSON array of strings. No markdown, no explanation, no preamble.
2. Each sub-query must be atomic — answerable from a single coherent piece of information.
3. Each sub-query must be a complete, self-contained question with no pronouns or references.
4. Decompose exhaustively — if answering the original query requires knowing N distinct facts, produce N sub-queries.
5. Do NOT merge distinct information needs into one sub-query.
6. Do NOT produce sub-queries that are answerable by inference from other sub-queries — each must require independent retrieval.
7. Do NOT produce sub-queries that go beyond what the original query is asking.
8. If the original query is already fully atomic, return a single-element array.
9. Maximum 6 sub-queries. If the query genuinely requires more, return the 6 most essential.
10. Sub-queries should be phrased as questions, not search queries.

# Coverage Check Usage
After retrieval, each sub-query will be checked against retrieved context with a binary yes/no:
- YES: retrieved context contains sufficient information to answer this sub-query
- NO: this sub-query is uncovered — trigger another retrieval iteration targeting it

# Few-Shot Examples

## Example 1
**Original Query**: "how do I invalidate queries after a mutation and what happens to inactive ones"
**Output**:
[
  "How do I call invalidateQueries after a mutation completes?",
  "What is the default refetch behavior for active queries when invalidateQueries is called?",
  "What happens to inactive queries when invalidateQueries is called by default?",
  "How do I configure invalidateQueries to also refetch inactive queries?"
]

## Example 2
**Original Query**: "what is the difference between setQueryData and fetchQuery"
**Output**:
[
  "What does setQueryData do and when should it be used?",
  "What does fetchQuery do and when should it be used?",
  "Is setQueryData synchronous or asynchronous?",
  "Is fetchQuery synchronous or asynchronous?",
  "What is the key difference in use case between setQueryData and fetchQuery?"
]

## Example 3
**Original Query**: "how does the DataProcessor class parse CSV files"
**Output**:
[
  "What is the DataProcessor class responsible for?",
  "Which method in DataProcessor handles CSV parsing?",
  "What are the input parameters of the CSV parsing method?",
  "What does the CSV parsing method return?"
]

## Example 4
**Original Query**: "how do I use prefetchQuery"
**Output**:
[
  "What does prefetchQuery do?",
  "What options does prefetchQuery accept?",
  "What does prefetchQuery return?"
]

# Input
**Original Query**: {user_query}"""

SPLIT_SUBQUERIES_PROMPT = PromptTemplate(
    template=SPLIT_SUBQUERIES_PROMPT_TEMPLATE, input_variables=["user_query"]
)

COVERAGE_CHECK_PROMPT_TEMPLATE = """
# Role
You are a retrieval coverage assessor.

# Task
Given a list of sub-queries and a block of retrieved context, determine which sub-queries are sufficiently answered by the context.

# Rules
1. Output ONLY a valid JSON object. No markdown, no explanation.
2. For each sub-query, return true if the context contains enough information to answer it, false if not.
3. Be strict — partial mentions do not count as covered.

# Output Format
{{
  "coverage": {{
    "<sub-query verbatim>": true | false
  }},
  "covered_count": int,
  "total_count": int
}}

# Input
**Sub-queries**: {{sub_queries}}
**Retrieved Context**: {{context}}

"""
COVERAGE_CHECK_PROMPT = PromptTemplate(
    template=COVERAGE_CHECK_PROMPT_TEMPLATE, input_variables=["sub_queries", "context"]
)


DEMO_DOC = """
An Occurrence at Owl Creek Bridge

by Ambrose Bierce

THE MILLENNIUM FULCRUM EDITION, 1988




I


A man stood upon a railroad bridge in northern Alabama, looking down
into the swift water twenty feet below. The man’s hands were behind his
back, the wrists bound with a cord. A rope closely encircled his neck.
It was attached to a stout cross-timber above his head and the slack
fell to the level of his knees. Some loose boards laid upon the ties
supporting the rails of the railway supplied a footing for him and his
executioners—two private soldiers of the Federal army, directed by a
sergeant who in civil life may have been a deputy sheriff. At a short
remove upon the same temporary platform was an officer in the uniform
of his rank, armed. He was a captain. A sentinel at each end of the
bridge stood with his rifle in the position known as “support,” that is
to say, vertical in front of the left shoulder, the hammer resting on
the forearm thrown straight across the chest—a formal and unnatural
position, enforcing an erect carriage of the body. It did not appear to
be the duty of these two men to know what was occurring at the center
of the bridge; they merely blockaded the two ends of the foot planking
that traversed it.

Beyond one of the sentinels nobody was in sight; the railroad ran
straight away into a forest for a hundred yards, then, curving, was
lost to view. Doubtless there was an outpost farther along. The other
bank of the stream was open ground—a gentle slope topped with a
stockade of vertical tree trunks, loopholed for rifles, with a single
embrasure through which protruded the muzzle of a brass cannon
commanding the bridge. Midway up the slope between the bridge and fort
were the spectators—a single company of infantry in line, at “parade
rest,” the butts of their rifles on the ground, the barrels inclining
slightly backward against the right shoulder, the hands crossed upon
the stock. A lieutenant stood at the right of the line, the point of
his sword upon the ground, his left hand resting upon his right.
Excepting the group of four at the center of the bridge, not a man
moved. The company faced the bridge, staring stonily, motionless. The
sentinels, facing the banks of the stream, might have been statues to
adorn the bridge. The captain stood with folded arms, silent, observing
the work of his subordinates, but making no sign. Death is a dignitary
who when he comes announced is to be received with formal
manifestations of respect, even by those most familiar with him. In the
code of military etiquette silence and fixity are forms of deference.

The man who was engaged in being hanged was apparently about
thirty-five years of age. He was a civilian, if one might judge from
his habit, which was that of a planter. His features were good—a
straight nose, firm mouth, broad forehead, from which his long, dark
hair was combed straight back, falling behind his ears to the collar of
his well fitting frock coat. He wore a moustache and pointed beard, but
no whiskers; his eyes were large and dark gray, and had a kindly
expression which one would hardly have expected in one whose neck was
in the hemp. Evidently this was no vulgar assassin. The liberal
military code makes provision for hanging many kinds of persons, and
gentlemen are not excluded.

The preparations being complete, the two private soldiers stepped aside
and each drew away the plank upon which he had been standing. The
sergeant turned to the captain, saluted and placed himself immediately
behind that officer, who in turn moved apart one pace. These movements
left the condemned man and the sergeant standing on the two ends of the
same plank, which spanned three of the cross-ties of the bridge. The
end upon which the civilian stood almost, but not quite, reached a
fourth. This plank had been held in place by the weight of the captain;
it was now held by that of the sergeant. At a signal from the former
the latter would step aside, the plank would tilt and the condemned man
go down between two ties. The arrangement commended itself to his
judgement as simple and effective. His face had not been covered nor
his eyes bandaged. He looked a moment at his “unsteadfast footing,”
then let his gaze wander to the swirling water of the stream racing
madly beneath his feet. A piece of dancing driftwood caught his
attention and his eyes followed it down the current. How slowly it
appeared to move! What a sluggish stream!

He closed his eyes in order to fix his last thoughts upon his wife and
children. The water, touched to gold by the early sun, the brooding
mists under the banks at some distance down the stream, the fort, the
soldiers, the piece of drift—all had distracted him. And now he became
conscious of a new disturbance. Striking through the thought of his
dear ones was sound which he could neither ignore nor understand, a
sharp, distinct, metallic percussion like the stroke of a blacksmith’s
hammer upon the anvil; it had the same ringing quality. He wondered
what it was, and whether immeasurably distant or near by— it seemed
both. Its recurrence was regular, but as slow as the tolling of a death
knell. He awaited each new stroke with impatience and—he knew not
why—apprehension. The intervals of silence grew progressively longer;
the delays became maddening. With their greater infrequency the sounds
increased in strength and sharpness. They hurt his ear like the thrust
of a knife; he feared he would shriek. What he heard was the ticking of
his watch.

He unclosed his eyes and saw again the water below him. “If I could
free my hands,” he thought, “I might throw off the noose and spring
into the stream. By diving I could evade the bullets and, swimming
vigorously, reach the bank, take to the woods and get away home. My
home, thank God, is as yet outside their lines; my wife and little ones
are still beyond the invader’s farthest advance.”

As these thoughts, which have here to be set down in words, were
flashed into the doomed man’s brain rather than evolved from it the
captain nodded to the sergeant. The sergeant stepped aside.




II


Peyton Farquhar was a well to do planter, of an old and highly
respected Alabama family. Being a slave owner and like other slave
owners a politician, he was naturally an original secessionist and
ardently devoted to the Southern cause. Circumstances of an imperious
nature, which it is unnecessary to relate here, had prevented him from
taking service with that gallant army which had fought the disastrous
campaigns ending with the fall of Corinth, and he chafed under the
inglorious restraint, longing for the release of his energies, the
larger life of the soldier, the opportunity for distinction. That
opportunity, he felt, would come, as it comes to all in wartime.
Meanwhile he did what he could. No service was too humble for him to
perform in the aid of the South, no adventure too perilous for him to
undertake if consistent with the character of a civilian who was at
heart a soldier, and who in good faith and without too much
qualification assented to at least a part of the frankly villainous
dictum that all is fair in love and war.

One evening while Farquhar and his wife were sitting on a rustic bench
near the entrance to his grounds, a gray-clad soldier rode up to the
gate and asked for a drink of water. Mrs. Farquhar was only too happy
to serve him with her own white hands. While she was fetching the water
her husband approached the dusty horseman and inquired eagerly for news
from the front.

“The Yanks are repairing the railroads,” said the man, “and are getting
ready for another advance. They have reached the Owl Creek bridge, put
it in order and built a stockade on the north bank. The commandant has
issued an order, which is posted everywhere, declaring that any
civilian caught interfering with the railroad, its bridges, tunnels, or
trains will be summarily hanged. I saw the order.”

“How far is it to the Owl Creek bridge?” Farquhar asked.

“About thirty miles.”

“Is there no force on this side of the creek?”

“Only a picket post half a mile out, on the railroad, and a single
sentinel at this end of the bridge.”

“Suppose a man—a civilian and student of hanging—should elude the
picket post and perhaps get the better of the sentinel,” said Farquhar,
smiling, “what could he accomplish?”

The soldier reflected. “I was there a month ago,” he replied. “I
observed that the flood of last winter had lodged a great quantity of
driftwood against the wooden pier at this end of the bridge. It is now
dry and would burn like tinder.”

The lady had now brought the water, which the soldier drank. He thanked
her ceremoniously, bowed to her husband and rode away. An hour later,
after nightfall, he repassed the plantation, going northward in the
direction from which he had come. He was a Federal scout.




III


As Peyton Farquhar fell straight downward through the bridge he lost
consciousness and was as one already dead. From this state he was
awakened—ages later, it seemed to him—by the pain of a sharp pressure
upon his throat, followed by a sense of suffocation. Keen, poignant
agonies seemed to shoot from his neck downward through every fiber of
his body and limbs. These pains appeared to flash along well defined
lines of ramification and to beat with an inconceivably rapid
periodicity. They seemed like streams of pulsating fire heating him to
an intolerable temperature. As to his head, he was conscious of nothing
but a feeling of fullness—of congestion. These sensations were
unaccompanied by thought. The intellectual part of his nature was
already effaced; he had power only to feel, and feeling was torment. He
was conscious of motion. Encompassed in a luminous cloud, of which he
was now merely the fiery heart, without material substance, he swung
through unthinkable arcs of oscillation, like a vast pendulum. Then all
at once, with terrible suddenness, the light about him shot upward with
the noise of a loud splash; a frightful roaring was in his ears, and
all was cold and dark. The power of thought was restored; he knew that
the rope had broken and he had fallen into the stream. There was no
additional strangulation; the noose about his neck was already
suffocating him and kept the water from his lungs. To die of hanging at
the bottom of a river!—the idea seemed to him ludicrous. He opened his
eyes in the darkness and saw above him a gleam of light, but how
distant, how inaccessible! He was still sinking, for the light became
fainter and fainter until it was a mere glimmer. Then it began to grow
and brighten, and he knew that he was rising toward the surface—knew it
with reluctance, for he was now very comfortable. “To be hanged and
drowned,” he thought, “that is not so bad; but I do not wish to be
shot. No; I will not be shot; that is not fair.”

He was not conscious of an effort, but a sharp pain in his wrist
apprised him that he was trying to free his hands. He gave the struggle
his attention, as an idler might observe the feat of a juggler, without
interest in the outcome. What splendid effort!—what magnificent, what
superhuman strength! Ah, that was a fine endeavor! Bravo! The cord fell
away; his arms parted and floated upward, the hands dimly seen on each
side in the growing light. He watched them with a new interest as first
one and then the other pounced upon the noose at his neck. They tore it
away and thrust it fiercely aside, its undulations resembling those of
a water snake. “Put it back, put it back!” He thought he shouted these
words to his hands, for the undoing of the noose had been succeeded by
the direst pang that he had yet experienced. His neck ached horribly;
his brain was on fire, his heart, which had been fluttering faintly,
gave a great leap, trying to force itself out at his mouth. His whole
body was racked and wrenched with an insupportable anguish! But his
disobedient hands gave no heed to the command. They beat the water
vigorously with quick, downward strokes, forcing him to the surface. He
felt his head emerge; his eyes were blinded by the sunlight; his chest
expanded convulsively, and with a supreme and crowning agony his lungs
engulfed a great draught of air, which instantly he expelled in a
shriek!

He was now in full possession of his physical senses. They were,
indeed, preternaturally keen and alert. Something in the awful
disturbance of his organic system had so exalted and refined them that
they made record of things never before perceived. He felt the ripples
upon his face and heard their separate sounds as they struck. He looked
at the forest on the bank of the stream, saw the individual trees, the
leaves and the veining of each leaf—he saw the very insects upon them:
the locusts, the brilliant bodied flies, the gray spiders stretching
their webs from twig to twig. He noted the prismatic colors in all the
dewdrops upon a million blades of grass. The humming of the gnats that
danced above the eddies of the stream, the beating of the dragon flies’
wings, the strokes of the water spiders’ legs, like oars which had
lifted their boat—all these made audible music. A fish slid along
beneath his eyes and he heard the rush of its body parting the water.

He had come to the surface facing down the stream; in a moment the
visible world seemed to wheel slowly round, himself the pivotal point,
and he saw the bridge, the fort, the soldiers upon the bridge, the
captain, the sergeant, the two privates, his executioners. They were in
silhouette against the blue sky. They shouted and gesticulated,
pointing at him. The captain had drawn his pistol, but did not fire;
the others were unarmed. Their movements were grotesque and horrible,
their forms gigantic.

Suddenly he heard a sharp report and something struck the water smartly
within a few inches of his head, spattering his face with spray. He
heard a second report, and saw one of the sentinels with his rifle at
his shoulder, a light cloud of blue smoke rising from the muzzle. The
man in the water saw the eye of the man on the bridge gazing into his
own through the sights of the rifle. He observed that it was a gray eye
and remembered having read that gray eyes were keenest, and that all
famous marksmen had them. Nevertheless, this one had missed.

A counter-swirl had caught Farquhar and turned him half round; he was
again looking at the forest on the bank opposite the fort. The sound of
a clear, high voice in a monotonous singsong now rang out behind him
and came across the water with a distinctness that pierced and subdued
all other sounds, even the beating of the ripples in his ears. Although
no soldier, he had frequented camps enough to know the dread
significance of that deliberate, drawling, aspirated chant; the
lieutenant on shore was taking a part in the morning’s work. How coldly
and pitilessly—with what an even, calm intonation, presaging, and
enforcing tranquility in the men—with what accurately measured interval
fell those cruel words:

“Company!… Attention!… Shoulder arms!… Ready!… Aim!… Fire!”

Farquhar dived—dived as deeply as he could. The water roared in his
ears like the voice of Niagara, yet he heard the dull thunder of the
volley and, rising again toward the surface, met shining bits of metal,
singularly flattened, oscillating slowly downward. Some of them touched
him on the face and hands, then fell away, continuing their descent.
One lodged between his collar and neck; it was uncomfortably warm and
he snatched it out.

As he rose to the surface, gasping for breath, he saw that he had been
a long time under water; he was perceptibly farther downstream—nearer
to safety. The soldiers had almost finished reloading; the metal
ramrods flashed all at once in the sunshine as they were drawn from the
barrels, turned in the air, and thrust into their sockets. The two
sentinels fired again, independently and ineffectually.

The hunted man saw all this over his shoulder; he was now swimming
vigorously with the current. His brain was as energetic as his arms and
legs; he thought with the rapidity of lightning:

“The officer,” he reasoned, “will not make that martinet’s error a
second time. It is as easy to dodge a volley as a single shot. He has
probably already given the command to fire at will. God help me, I
cannot dodge them all!”

An appalling splash within two yards of him was followed by a loud,
rushing sound, DIMINUENDO, which seemed to travel back through the air
to the fort and died in an explosion which stirred the very river to
its deeps! A rising sheet of water curved over him, fell down upon him,
blinded him, strangled him! The cannon had taken an hand in the game.
As he shook his head free from the commotion of the smitten water he
heard the deflected shot humming through the air ahead, and in an
instant it was cracking and smashing the branches in the forest beyond.

“They will not do that again,” he thought; “the next time they will use
a charge of grape. I must keep my eye upon the gun; the smoke will
apprise me—the report arrives too late; it lags behind the missile.
That is a good gun.”

Suddenly he felt himself whirled round and round—spinning like a top.
The water, the banks, the forests, the now distant bridge, fort and
men, all were commingled and blurred. Objects were represented by their
colors only; circular horizontal streaks of color—that was all he saw.
He had been caught in a vortex and was being whirled on with a velocity
of advance and gyration that made him giddy and sick. In few moments he
was flung upon the gravel at the foot of the left bank of the
stream—the southern bank—and behind a projecting point which concealed
him from his enemies. The sudden arrest of his motion, the abrasion of
one of his hands on the gravel, restored him, and he wept with delight.
He dug his fingers into the sand, threw it over himself in handfuls and
audibly blessed it. It looked like diamonds, rubies, emeralds; he could
think of nothing beautiful which it did not resemble. The trees upon
the bank were giant garden plants; he noted a definite order in their
arrangement, inhaled the fragrance of their blooms. A strange roseate
light shone through the spaces among their trunks and the wind made in
their branches the music of AEolian harps. He had not wish to perfect
his escape—he was content to remain in that enchanting spot until
retaken.

A whiz and a rattle of grapeshot among the branches high above his head
roused him from his dream. The baffled cannoneer had fired him a random
farewell. He sprang to his feet, rushed up the sloping bank, and
plunged into the forest.

All that day he traveled, laying his course by the rounding sun. The
forest seemed interminable; nowhere did he discover a break in it, not
even a woodman’s road. He had not known that he lived in so wild a
region. There was something uncanny in the revelation.

By nightfall he was fatigued, footsore, famished. The thought of his
wife and children urged him on. At last he found a road which led him
in what he knew to be the right direction. It was as wide and straight
as a city street, yet it seemed untraveled. No fields bordered it, no
dwelling anywhere. Not so much as the barking of a dog suggested human
habitation. The black bodies of the trees formed a straight wall on
both sides, terminating on the horizon in a point, like a diagram in a
lesson in perspective. Overhead, as he looked up through this rift in
the wood, shone great golden stars looking unfamiliar and grouped in
strange constellations. He was sure they were arranged in some order
which had a secret and malign significance. The wood on either side was
full of singular noises, among which—once, twice, and again—he
distinctly heard whispers in an unknown tongue.

His neck was in pain and lifting his hand to it found it horribly
swollen. He knew that it had a circle of black where the rope had
bruised it. His eyes felt congested; he could no longer close them. His
tongue was swollen with thirst; he relieved its fever by thrusting it
forward from between his teeth into the cold air. How softly the turf
had carpeted the untraveled avenue—he could no longer feel the roadway
beneath his feet!

Doubtless, despite his suffering, he had fallen asleep while walking,
for now he sees another scene—perhaps he has merely recovered from a
delirium. He stands at the gate of his own home. All is as he left it,
and all bright and beautiful in the morning sunshine. He must have
traveled the entire night. As he pushes open the gate and passes up the
wide white walk, he sees a flutter of female garments; his wife,
looking fresh and cool and sweet, steps down from the veranda to meet
him. At the bottom of the steps she stands waiting, with a smile of
ineffable joy, an attitude of matchless grace and dignity. Ah, how
beautiful she is! He springs forwards with extended arms. As he is
about to clasp her he feels a stunning blow upon the back of the neck;
a blinding white light blazes all about him with a sound like the shock
of a cannon—then all is darkness and silence!

Peyton Farquhar was dead; his body, with a broken neck, swung gently
from side to side beneath the timbers of the Owl Creek bridge.


"""

QUERY_EXPANSION_SYSTEM_PROMPT = """You are a query expansion engine for a RAG system.
Your task is to analyze a user query and produce:
1. A rewritten query optimized for retrieval
2. A checklist of 4-5 sub-questions that cover ALL information needs

Output ONLY valid JSON matching this schema:
{
  "rewritten_query": "string - cleaned, deambiguated query for retrieval",
  "checklist": [
    {"id": 1, "question": "string - specific information need", "answered": false},
    ...
  ]
}

Rules:
- rewritten_query must be a single self-contained sentence
- Each checklist item must be atomic (answerable from a single coherent piece of info)
- Checklist must be exhaustive - if answering the original requires N facts, produce N items
- Maximum 5 checklist items - prioritize the most essential
- Questions should be phrased as questions, not search queries
- All questions start with answered: false"""

QUERY_EXPANSION_USER_TEMPLATE = """Original user query: {user_query}

Conversation history (for context resolution):
{chat_history}

Analyze this query and output the JSON expansion."""


SUBAGENT_SYSTEM_PROMPT = """You are a retrieval planning agent.
Your task is to assess a checklist against retrieved context and plan targeted retrieval.

You will receive:
- original_query: The user's original question
- checklist: Current checklist with answered status
- accumulated_context: Documents/code retrieved so far
- iteration: Current loop iteration (1-5)

Your job:
1. Mark checklist items as answered=true if the context contains sufficient information
2. Generate retrieval queries for any remaining uncovered items
3. Decide if retrieval should target vector (documents), graph (code structure), or both

Output ONLY valid JSON matching this schema:
{
  "checklist": [
    {"id": int, "question": "string", "answered": boolean},
    ...
  ],
  "all_answered": boolean,
  "retrieval_queries": [
    {
      "query": "string - specific retrieval query",
      "target": "vector" | "graph" | "both",
      "filters": {
        "file_ids": ["id1", "id2"] | null,
        "node_type": "function" | "class" | null,
        "language": "python" | "go" | null
      }
    },
    ...
  ]
}

Rules:
- Mark answered=true ONLY if context contains sufficient info to answer that specific question
- Be strict - partial mentions do NOT count as answered
- Maximum 3 retrieval queries per iteration
- Use target="vector" for documentation/prose content
- Use target="graph" for code structure questions (who calls X, what does Y inherit from)
- Use target="both" when the question needs both code and docs
- Set all_answered=true only when ALL items have answered=true
- If iteration >= 4 and still not answered, focus on the most critical items
- If the query cannot be answered from available sources, still mark answered=false and provide best-effort retrieval"""


SUBAGENT_USER_TEMPLATE = """# Input

**Original Query**: {original_query}

**Checklist** (current state):
{checklist_json}

**Accumulated Context**:
{context_text}

**Iteration**: {iteration} / {max_iterations}

Analyze and output the JSON result."""
