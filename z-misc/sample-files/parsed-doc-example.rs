// Example: parsed `Document` from `z-misc/sample-files/sample-md.md`
// When document parsing/chunking is complete, this structure represents a fully ingested document.

use fosra::types::{
    Chunk, ChunkMetadata, Document, DocumentMetadata, HeadingContext, HeadingLevel, Keyword,
};

fn doc_metadata_example() -> DocumentMetadata {
    DocumentMetadata {
        title: Some("Available middlewares".to_string()),
        path: Some(std::path::PathBuf::from("z-misc/sample-files/sample-md.md")),
        subject: Some("Middleware documentation for Taskiq".to_string()),
        authors: Some(vec!["Taskiq Team".to_string()]),
        keywords: Some(vec![
            "middleware".to_string(),
            "retry".to_string(),
            "prometheus".to_string(),
            "opentelemetry".to_string(),
            "taskiq".to_string(),
            "broker".to_string(),
            "async".to_string(),
        ]),
        language: Some("en".to_string()),
        extension: Some("md".to_string()),
        mime: Some("text/markdown".to_string()),
        created_at: Some("2024-01-15T10:30:00Z".to_string()),
        modified_at: Some("2024-03-20T14:22:00Z".to_string()),
        extracted_keywords: Some(vec![
            Keyword {
                text: "SmartRetryMiddleware".to_string(),
                score: 0.92,
            },
            Keyword {
                text: "PrometheusMiddleware".to_string(),
                score: 0.89,
            },
            Keyword {
                text: "OpenTelemetryMiddleware".to_string(),
                score: 0.87,
            },
            Keyword {
                text: "TaskiqInstrumentor".to_string(),
                score: 0.84,
            },
        ]),
        created_by: Some("documentation pipeline".to_string()),
        modified_by: None,
        extraction_duration_ms: Some(127),
        category: Some("technical documentation".to_string()),
        tags: Some(vec![
            "middleware".to_string(),
            "taskiq".to_string(),
            "python".to_string(),
            "async".to_string(),
        ]),
        document_version: Some("1.0.0".to_string()),
        output_format: Some("markdown".to_string()),
    }
}

fn parsed_document_example() -> Document {
    Document {
        id: "doc-abc123def456".to_string(),
        content: r#"---
order: 5
---

# Available middlewares

Middlewares allow you to execute code when specific event occurs.
Taskiq has several default middlewares.

## Simple retry middleware

This middleware allows you to restart functions on errors. If exception was raised during task execution,
the task would be resent with same parameters.

To enable this middleware, add it to the list of middlewares for a broker.

```python
from taskiq import TaskiqBroker
from taskiq_middleware_retry import RetryMiddleware

broker = TaskiqBroker(
    middlewares=[RetryMiddleware()],
)

@broker.task(retry_on_error=True, max_retries=3)
async def my_task():
    ...
```

After that you can add a label to task that you want to restart on error.

```python
@broker.task(retry_on_error=True, max_retries=5)
async def my_task():
    ...
```

`retry_on_error` enables retries for a task. `max_retries` is the maximum number of times,.

## Smart retry middleware

The `SmartRetryMiddleware` automatically retries tasks with flexible delay settings and retry strategies when errors occur. This is particularly useful when tasks fail due to temporary issues, such as network errors or temporary unavailability of external services.

### Key Features

* **Retry Limits**: Set the maximum number of retry attempts (`max_retries`).
* **Exponential Backoff**: Automatically increase delay between retries.
* **Configurable Delays**: Customize initial and maximum retry delays.

### Middleware Integration

To use `SmartRetryMiddleware`, add it to the list of middlewares in your broker:

```python
from taskiq import TaskiqBroker
from taskiq_middleware_smart_retry import SmartRetryMiddleware

broker = TaskiqBroker(
    middlewares=[SmartRetryMiddleware()],
)
```

### Using Middleware with Tasks

To enable retries for a specific task, specify the following parameters:

```python
@broker.task(
    retry_on_error=True,
    max_retries=5,
    retry_delay=1.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    max_delay_exponent=4,
)
async def my_task():
    ...
```

* `retry_on_error`: Enables the retry mechanism for the specific task.
* `max_retries`: Maximum number of retry attempts (default: 3).
* `retry_delay`: Initial delay between retries in seconds (default: 1.0).
* `max_delay`: Maximum delay cap in seconds (default: 60.0).
* `backoff_multiplier`: Multiplier for exponential backoff (default: 2.0).

### Usage Recommendations

Use jitter and exponential backoff to avoid repetitive load peaks, especially in high-load systems. Choose appropriate `max_delay_exponent` values to prevent excessively long intervals between retries if your task execution is time-sensitive.

## Prometheus middleware

You can enable prometheus metrics for workers by adding `PrometheusMiddleware`.
To do so, you need to install `prometheus_client` package or you can install metrics extras for taskiq.

::: tabs

@tab only prometheus

```bash
pip install prometheus_client
```

@tab taskiq with extras

```bash
pip install taskiq[metrics]
```

:::

After that, metrics will be available at port 9000. Of course, this parameter can be configured.
If you have other metrics, they'll be shown as well.

## OpenTelemetry Middleware

You can enable opentelemetry tracing for workers by adding `OpenTelemetryMiddleware` or using `TaskiqInstrumentor` (preferred).

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

::: tabs

@tab instrumentor

python
from taskiq import TaskiqBroker, TaskiqInstrumentor

instrumentor = TaskiqInstrumentor()
instrumentor.start()

broker = TaskiqBroker(...)
```

@tab middleware

```python
from taskiq import TaskiqBroker
from taskiq_middleware_opentelemetry import OpenTelemetryMiddleware

broker = TaskiqBroker(
    middlewares=[OpenTelemetryMiddleware()],
)
```

:::

Auto-instrumentation is also supported."#.to_string(),
        metadata: doc_metadata_example(),

        // ─── Chunks ─────────────────────────────────────────────────────────────
        // Each chunk represents a section at ~250 token boundaries with heading context.
        chunks: Some(vec![
            // Chunk 0: Title and intro
            Chunk {
                content: "# Available middlewares\n\n\
                         Middlewares allow you to execute code when specific event occurs.\n\
                         Taskiq has several default middlewares."
                    .to_string(),
                embedding: Some(vec![0.1, 0.2, 0.3]),
                metadata: ChunkMetadata {
                    byte_start: 62,
                    byte_end: 212,
                    token_count: Some(38),
                    chunk_index: 0,
                    total_chunks: 12,
                    first_page: None,
                    last_page: None,
                    heading_context: Some(HeadingContext {
                        headings: vec![
                            HeadingLevel { level: 1, text: "Available middlewares".to_string() },
                        ],
                    }),
                },
            },
            // Chunk 1: Simple retry middleware section
            Chunk {
                content: "## Simple retry middleware\n\n\
                         This middleware allows you to restart functions on errors. If exception was raised during task execution,\n\
                         the task would be resent with same parameters.\n\n\
                         To enable this middleware, add it to the list of middlewares for a broker.\n\n\
                         ```python\n\
                         from taskiq import TaskiqBroker\n\
                         from taskiq_middleware_retry import RetryMiddleware\n\n\
                         broker = TaskiqBroker(\n\
                             middlewares=[RetryMiddleware()],\n\
                         )\n\n\
                         @broker.task(retry_on_error=True, max_retries=3)\n\
                         async def my_task():\n\
                             ...\n\
                         ```\n\n\
                         After that you can add a label to task that you want to restart on error.\n\n\
                         ```python\n\
                         @broker.task(retry_on_error=True, max_retries=5)\n\
                         async def my_task():\n\
                             ...\n\
                         ```\n\n\
                         `retry_on_error` enables retries for a task. `max_retries` is the maximum number of times,."
                    .to_string(),
                embedding: Some(vec![0.15, 0.25, 0.35]),
                metadata: ChunkMetadata {
                    byte_start: 214,
                    byte_end: 892,
                    token_count: Some(187),
                    chunk_index: 1,
                    total_chunks: 12,
                    first_page: None,
                    last_page: None,
                    heading_context: Some(HeadingContext {
                        headings: vec![
                            HeadingLevel { level: 1, text: "Available middlewares".to_string() },
                            HeadingLevel { level: 2, text: "Simple retry middleware".to_string() },
                        ],
                    }),
                },
            },
            // Chunk 2: Smart retry middleware intro
            Chunk {
                content: "## Smart retry middleware\n\n\
                         The `SmartRetryMiddleware` automatically retries tasks with flexible delay settings and retry strategies when errors occur. \
                         This is particularly useful when tasks fail due to temporary issues, such as network errors or temporary unavailability of external services."
                    .to_string(),
                embedding: Some(vec![0.2, 0.3, 0.4]),
                metadata: ChunkMetadata {
                    byte_start: 894,
                    byte_end: 1234,
                    token_count: Some(112),
                    chunk_index: 2,
                    total_chunks: 12,
                    first_page: None,
                    last_page: None,
                    heading_context: Some(HeadingContext {
                        headings: vec![
                            HeadingLevel { level: 1, text: "Available middlewares".to_string() },
                            HeadingLevel { level: 2, text: "Smart retry middleware".to_string() },
                        ],
                    }),
                },
            },
            // Chunk 3: Smart retry middleware - Key Features
            Chunk {
                content: "### Key Features\n\n\
                         * **Retry Limits**: Set the maximum number of retry attempts (`max_retries`).\n\
                         * **Exponential Backoff**: Automatically increase delay between retries.\n\
                         * **Configurable Delays**: Customize initial and maximum retry delays."
                    .to_string(),
                embedding: Some(vec![0.18, 0.28, 0.38]),
                metadata: ChunkMetadata {
                    byte_start: 1236,
                    byte_end: 1489,
                    token_count: Some(68),
                    chunk_index: 3,
                    total_chunks: 12,
                    first_page: None,
                    last_page: None,
                    heading_context: Some(HeadingContext {
                        headings: vec![
                            HeadingLevel { level: 1, text: "Available middlewares".to_string() },
                            HeadingLevel { level: 2, text: "Smart retry middleware".to_string() },
                            HeadingLevel { level: 3, text: "Key Features".to_string() },
                        ],
                    }),
                },
            },
            // Chunk 4: Smart retry middleware - Usage Recommendations
            Chunk {
                content: "### Usage Recommendations\n\n\
                         Use jitter and exponential backoff to avoid repetitive load peaks, especially in high-load systems. \
                         Choose appropriate `max_delay_exponent` values to prevent excessively long intervals between retries \
                         if your task execution is time-sensitive."
                    .to_string(),
                embedding: Some(vec![0.22, 0.32, 0.42]),
                metadata: ChunkMetadata {
                    byte_start: 1947,
                    byte_end: 2214,
                    token_count: Some(82),
                    chunk_index: 5,
                    total_chunks: 12,
                    first_page: None,
                    last_page: None,
                    heading_context: Some(HeadingContext {
                        headings: vec![
                            HeadingLevel { level: 1, text: "Available middlewares".to_string() },
                            HeadingLevel { level: 2, text: "Smart retry middleware".to_string() },
                            HeadingLevel { level: 3, text: "Usage Recommendations".to_string() },
                        ],
                    }),
                },
            },
            // Chunk 5: Prometheus middleware
            Chunk {
                content: "## Prometheus middleware\n\n\
                         You can enable prometheus metrics for workers by adding `PrometheusMiddleware`.\n\
                         To do so, you need to install `prometheus_client` package or you can install metrics extras for taskiq.\n\n\
                         ::: tabs\n\n\
                         @tab only prometheus\n\n\
                         ```bash\n\
                         pip install prometheus_client\n\
                         ```\n\n\
                         @tab taskiq with extras\n\n\
                         ```bash\n\
                         pip install taskiq[metrics]\n\
                         ```\n\n\
                         :::\n\n\
                         After that, metrics will be available at port 9000. Of course, this parameter can be configured.\n\
                         If you have other metrics, they'll be shown as well."
                    .to_string(),
                embedding: Some(vec![0.25, 0.35, 0.45]),
                metadata: ChunkMetadata {
                    byte_start: 2216,
                    byte_end: 2723,
                    token_count: Some(156),
                    chunk_index: 6,
                    total_chunks: 12,
                    first_page: None,
                    last_page: None,
                    heading_context: Some(HeadingContext {
                        headings: vec![
                            HeadingLevel { level: 1, text: "Available middlewares".to_string() },
                            HeadingLevel { level: 2, text: "Prometheus middleware".to_string() },
                        ],
                    }),
                },
            },
            // Chunk 6: OpenTelemetry Middleware
            Chunk {
                content: "## OpenTelemetry Middleware\n\n\
                         You can enable opentelemetry tracing for workers by adding `OpenTelemetryMiddleware` \
                         or using `TaskiqInstrumentor` (preferred).\n\n\
                         ```bash\n\
                         pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp\n\
                         ```\n\n\
                         ::: tabs\n\n\
                         @tab instrumentor\n\n\
                         ```python\n\
                         from taskiq import TaskiqBroker, TaskiqInstrumentor\n\n\
                         instrumentor = TaskiqInstrumentor()\n\
                         instrumentor.start()\n\
                         broker = TaskiqBroker(...)\n\
                         ```\n\n\
                         @tab middleware\n\n\
                         ```python\n\
                         from taskiq import TaskiqBroker\n\
                         from taskiq_middleware_opentelemetry import OpenTelemetryMiddleware\n\n\
                         broker = TaskiqBroker(\n\
                             middlewares=[OpenTelemetryMiddleware()],\n\
                         )\n\
                         ```\n\n\
                         :::\n\n\
                         Auto-instrumentation is also supported."
                    .to_string(),
                embedding: Some(vec![0.28, 0.38, 0.48]),
                metadata: ChunkMetadata {
                    byte_start: 2725,
                    byte_end: 3456,
                    token_count: Some(218),
                    chunk_index: 7,
                    total_chunks: 12,
                    first_page: None,
                    last_page: None,
                    heading_context: Some(HeadingContext {
                        headings: vec![
                            HeadingLevel { level: 1, text: "Available middlewares".to_string() },
                            HeadingLevel { level: 2, text: "OpenTelemetry Middleware".to_string() },
                        ],
                    }),
                },
            },
        ]),
    }
}

fn main() {
    let doc = parsed_document_example();
    println!(
        "Parsed document: {} chunks, title: {:?}",
        doc.chunks.as_ref().map_or(0, |c| c.len()),
        doc.metadata.title
    );
}
