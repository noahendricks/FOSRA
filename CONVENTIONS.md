📝 CONVENTIONS.md

1. Core Philosophy

    Readability First: Code must prioritize clarity and maintainability. Avoid excessive "cleverness" that sacrifices immediate understanding.

2. Python & Type Hinting Standards

    Mandatory Type Hints: All functions, methods, and public variables must be fully annotated with type hints. This includes arguments, return values, and local variables where clarity is improved.

    AnyIO Typing: Use Callable[..., Awaitable[T]] for functions that accept or return an asynchronous function, where T is the expected return type.

        Example: Annotate asynchronous database transaction managers using types from anyio where applicable (e.g., anyio.to_thread.run_sync).

    SQLAlchemy / Pydantic: Use pydantic.BaseModel for all data validation and schema definitions. Database models should be correctly typed using sqlalchemy.orm.Mapped and related types.

3. Asynchronous I/O (Async/Await)

    Async Primitives: Always use async and await keywords for any I/O operation.

    Networking: Prefer httpx for all HTTP requests over older or synchronous libraries. All HTTP functions must be async def.

    Database: Use aiosqlite for SQLite operations and asyncpg for PostgreSQL, ensuring all database interactions are correctly awaited.

    Threading (Blocking Ops): If interacting with a blocking I/O library or synchronous code is unavoidable (e.g., file system access via python-magic, pypdf, or unstructured), wrap the call using await. Do not manually manage threads. Use asyncio.

4. Logging and Debugging

    Loguru Mandatory: All logging must use the loguru library. Avoid Python's built-in logging module.

    Structured Logging: Pass relevant data as keyword arguments to loguru (e.g., logger.info("User created", user_id=user.id, email=user.email)).

    Log Levels:

        DEBUG: Detailed diagnostic information (e.g., function entry/exit, full API request/response payloads).

        INFO: Standard confirmation that things are working as expected (e.g., "Service started," "Database connection established").

        WARNING: Unexpected events that do not stop execution (e.g., "Cache miss," "Configuration not found, using default").

        ERROR: Serious issues that prevent a function from completing successfully (e.g., "Failed to connect to Elasticsearch").

    Pysnooper: When debugging complex flows, use the pysnooper decorator temporarily for trace logging, and ensure it is removed before committing.

5. Comments and Documentation

    Sparse but Insightful Comments: Comments must be concise and explain the "why" behind a design choice, not the "what" (which should be clear from the code).

        Anti-Pattern: # Increment counter

        Preferred: # Use a lock-free counter to avoid contention under high load.

    Docstrings: Use Google-style docstrings for all public functions, classes, and methods, including Args: and Returns: sections, even when type hints are present.

6. Git and Style (Ruff)

    Formatters: The code must be compliant with the ruff configuration specified in the project.

    Imports: All imports must be organized alphabetically and separated into standard library, third-party, and local application imports.

7. Boundaries and Anti-Patterns

    Never Edit: Do not modify the project.toml file unless explicitly instructed.

    Avoid: Never use blocking I/O inside an async def function without wrapping it in anyio.to_thread.run_sync.

    Prefer: Use the aioresult library for all functions that might return a success or failure state instead of relying purely on exceptions for flow control.
