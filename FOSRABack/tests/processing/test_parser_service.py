"""
Tests for parser_service.py - Document parsing with registry pattern.

Covers:
- Enums and configuration dataclasses
- BaseParser registry and factory methods
- LocalParser implementations (Docling, Unstructured, Markdown, PyPDF)
- APIParser implementations (LlamaParse)
- ParserService orchestration and fallback mechanisms
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from FOSRABack.src.config.request_context import ParserConfig
from FOSRABack.src.processing.services.parser_service import (
    ParserType,
    DocumentType,
    ParsedDocument,
    BaseParser,
    LocalParser,
    APIParser,
    DoclingParser,
    UnstructuredParser,
    MarkdownParser,
    PyPDFParser,
    LlamaParseParser,
    ParserService,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def parser_config() -> ParserConfig:
    """Default parser configuration."""
    return ParserConfig(
        max_pages=10,
        extract_tables=True,
        extract_images=False,
        ocr_enabled=True,
        language="eng",
        timeout_seconds=60,
        fallback_parsers=[ParserType.PYPDF, ParserType.MARKDOWN],
    )


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Create a temporary PDF path."""
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.touch()
    return str(pdf_path)


@pytest.fixture
def sample_markdown_path(tmp_path: Path) -> str:
    """Create a temporary markdown file with content."""
    md_path = tmp_path / "test_document.md"
    md_path.write_text(
        "# Test Document\n\nThis is test content.\n\n## Section 1\n\nMore content here."
    )
    return str(md_path)


@pytest.fixture
def sample_text_path(tmp_path: Path) -> str:
    """Create a temporary text file."""
    txt_path = tmp_path / "test_document.txt"
    txt_path.write_text("This is plain text content for testing.")
    return str(txt_path)


@pytest.fixture
def sample_docx_path(tmp_path: Path) -> str:
    """Create a temporary docx path."""
    docx_path = tmp_path / "test_document.docx"
    docx_path.touch()
    return str(docx_path)


@pytest.fixture
def mock_session() -> AsyncMock:
    """Mock database session."""
    return AsyncMock()


# ============================================================================
# Enum Tests
# ============================================================================


class TestParserType:
    """Tests for ParserType enum."""

    def test_local_parser_types(self):
        """Test local parser type values."""
        assert ParserType.DOCLING == "DOCLING"
        assert ParserType.UNSTRUCTURED == "UNSTRUCTURED"
        assert ParserType.MARKDOWN == "MARKDOWN"
        assert ParserType.PYPDF == "PYPDF"

    def test_api_parser_types(self):
        """Test API parser type values."""
        assert ParserType.LLAMA_PARSE == "LLAMA_PARSE"
        assert ParserType.AZURE_DOCUMENT_INTELLIGENCE == "AZURE_DOCUMENT_INTELLIGENCE"
        assert ParserType.GOOGLE_DOCUMENT_AI == "GOOGLE_DOCUMENT_AI"
        assert ParserType.AWS_TEXTRACT == "AWS_TEXTRACT"


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_document_types(self):
        """Test document type values."""
        assert DocumentType.PDF == "pdf"
        assert DocumentType.DOCX == "docx"
        assert DocumentType.DOC == "doc"
        assert DocumentType.MARKDOWN == "markdown"
        assert DocumentType.TEXT == "text"
        assert DocumentType.UNKNOWN == "unknown"


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_default_values(self):
        """Test default values for ParsedDocument."""
        doc = ParsedDocument(content="Test content")

        assert doc.content == "Test content"
        assert doc.metadata == {}
        assert doc.page_count == 1
        assert doc.parser_used == ""
        assert doc.parse_time_ms == 0.0
        assert doc.tables == []
        assert doc.images == []
        assert doc.errors == []

    def test_full_initialization(self):
        """Test full initialization of ParsedDocument."""
        doc = ParsedDocument(
            content="Full content",
            metadata={"author": "Test"},
            page_count=5,
            parser_used="DOCLING",
            parse_time_ms=150.5,
            tables=[{"id": 1}],
            images=[{"id": 2}],
            errors=["Warning: OCR quality low"],
        )

        assert doc.content == "Full content"
        assert doc.metadata == {"author": "Test"}
        assert doc.page_count == 5
        assert doc.parser_used == "DOCLING"
        assert doc.parse_time_ms == 150.5
        assert len(doc.tables) == 1
        assert len(doc.images) == 1
        assert len(doc.errors) == 1


class TestParserConfig:
    """Tests for ParserConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ParserConfig()

        assert config.max_pages is None
        assert config.extract_tables is True
        assert config.extract_images is False
        assert config.ocr_enabled is True
        assert config.language == "eng"
        assert config.timeout_seconds == 300
        assert config.fallback_parsers == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ParserConfig(
            max_pages=50,
            extract_tables=False,
            extract_images=True,
            ocr_enabled=False,
            language="deu",
            timeout_seconds=600,
            fallback_parsers=[ParserType.PYPDF],
        )

        assert config.max_pages == 50
        assert config.extract_tables is False
        assert config.extract_images is True
        assert config.ocr_enabled is False
        assert config.language == "deu"
        assert config.timeout_seconds == 600
        assert config.fallback_parsers == [ParserType.PYPDF]


# ============================================================================
# BaseParser Tests
# ============================================================================


class TestBaseParser:
    """Tests for BaseParser abstract class and registry."""

    def test_registry_contains_parsers(self):
        """Test that parsers are registered."""
        available = BaseParser.get_available_parsers()

        assert "DOCLING" in available
        assert "UNSTRUCTURED" in available
        assert "MARKDOWN" in available
        assert "PYPDF" in available
        assert "LLAMA_PARSE" in available

    def test_get_parser_docling(self):
        """Test getting Docling parser."""
        parser = BaseParser.get_parser(ParserType.DOCLING)

        assert parser is not None
        assert isinstance(parser, DoclingParser)

    def test_get_parser_unstructured(self):
        """Test getting Unstructured parser."""
        parser = BaseParser.get_parser(ParserType.UNSTRUCTURED)

        assert parser is not None
        assert isinstance(parser, UnstructuredParser)

    def test_get_parser_markdown(self):
        """Test getting Markdown parser."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)

        assert parser is not None
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_pypdf(self):
        """Test getting PyPDF parser."""
        parser = BaseParser.get_parser(ParserType.PYPDF)

        assert parser is not None
        assert isinstance(parser, PyPDFParser)

    def test_get_parser_llama_parse(self):
        """Test getting LlamaParse parser."""
        parser = BaseParser.get_parser(ParserType.LLAMA_PARSE)

        assert parser is not None
        assert isinstance(parser, LlamaParseParser)

    def test_get_parser_with_session(self, mock_session: AsyncMock):
        """Test getting parser with database session."""
        parser = BaseParser.get_parser(ParserType.DOCLING, mock_session)

        assert parser is not None
        assert parser.session == mock_session

    def test_detect_document_type_pdf(self):
        """Test document type detection for PDF."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)
        doc_type = parser._detect_document_type("/path/to/file.pdf")

        assert doc_type == DocumentType.PDF

    def test_detect_document_type_markdown(self):
        """Test document type detection for markdown."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)

        assert parser._detect_document_type("/path/to/file.md") == DocumentType.MARKDOWN
        assert (
            parser._detect_document_type("/path/to/file.markdown")
            == DocumentType.MARKDOWN
        )

    def test_detect_document_type_text(self):
        """Test document type detection for text files."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)

        assert parser._detect_document_type("/path/to/file.txt") == DocumentType.TEXT
        assert parser._detect_document_type("/path/to/file.text") == DocumentType.TEXT

    def test_detect_document_type_html(self):
        """Test document type detection for HTML."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)

        assert parser._detect_document_type("/path/to/file.html") == DocumentType.HTML
        assert parser._detect_document_type("/path/to/file.htm") == DocumentType.HTML

    def test_detect_document_type_images(self):
        """Test document type detection for images."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)

        assert parser._detect_document_type("/path/to/file.png") == DocumentType.IMAGE
        assert parser._detect_document_type("/path/to/file.jpg") == DocumentType.IMAGE
        assert parser._detect_document_type("/path/to/file.jpeg") == DocumentType.IMAGE
        assert parser._detect_document_type("/path/to/file.tiff") == DocumentType.IMAGE
        assert parser._detect_document_type("/path/to/file.webp") == DocumentType.IMAGE

    def test_detect_document_type_office(self):
        """Test document type detection for office documents."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)

        assert parser._detect_document_type("/path/to/file.docx") == DocumentType.DOCX
        assert parser._detect_document_type("/path/to/file.doc") == DocumentType.DOC
        assert parser._detect_document_type("/path/to/file.pptx") == DocumentType.PPTX
        assert parser._detect_document_type("/path/to/file.xlsx") == DocumentType.XLSX

    def test_detect_document_type_unknown(self):
        """Test document type detection for unknown extensions."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)

        assert parser._detect_document_type("/path/to/file.xyz") == DocumentType.UNKNOWN
        assert parser._detect_document_type("/path/to/file") == DocumentType.UNKNOWN

    def test_create_error_result(self):
        """Test error result creation."""
        parser = BaseParser.get_parser(ParserType.MARKDOWN)
        result = parser._create_error_result("Test error message")

        assert result.content == ""
        assert result.parser_used == "MARKDOWN"
        assert "Test error message" in result.errors


# ============================================================================
# DoclingParser Tests
# ============================================================================


class TestDoclingParser:
    """Tests for DoclingParser."""

    def test_parser_attributes(self):
        """Test parser class attributes."""
        parser = DoclingParser()

        assert parser.parser_type == ParserType.DOCLING
        assert parser.display_name == "Docling"
        assert DocumentType.PDF in parser.supported_types
        assert DocumentType.DOCX in parser.supported_types

    def test_supports_document_type_pdf(self):
        """Test PDF support."""
        parser = DoclingParser()
        assert parser.supports_document_type(DocumentType.PDF) is True

    def test_supports_document_type_docx(self):
        """Test DOCX support."""
        parser = DoclingParser()
        assert parser.supports_document_type(DocumentType.DOCX) is True

    def test_supports_document_type_text(self):
        """Test that TEXT is not supported."""
        parser = DoclingParser()
        assert parser.supports_document_type(DocumentType.TEXT) is False

    @pytest.mark.asyncio
    async def test_parse_success(self, sample_pdf_path: str):
        """Test successful document parsing."""
        parser = DoclingParser()

        mock_docling_service = MagicMock()
        mock_docling_service.process_document = AsyncMock(
            return_value={
                "content": "Parsed content from Docling",
                "metadata": {"author": "Test Author"},
                "page_count": 3,
                "tables": [{"id": 1, "data": []}],
                "images": [],
            }
        )

        with patch(
            "FOSRABack.src.convo.utils.docling_helper.create_docling_service",
            return_value=mock_docling_service,
        ):
            result = await parser.parse(sample_pdf_path)

        assert result.content == "Parsed content from Docling"
        assert result.page_count == 3
        assert result.parser_used == "DOCLING"
        assert result.parse_time_ms > 0
        assert len(result.tables) == 1
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_parse_initialization_failure(self, sample_pdf_path: str):
        """Test handling of initialization failure."""
        parser = DoclingParser()

        with patch(
            "FOSRABack.src.convo.utils.docling_helper.create_docling_service",
            return_value=None,
        ):
            result = await parser.parse(sample_pdf_path)

        assert result.content == ""
        assert "Failed to initialize Docling" in result.errors

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, sample_pdf_path: str):
        """Test exception handling during parsing."""
        parser = DoclingParser()

        mock_docling_service = MagicMock()
        mock_docling_service.process_document = AsyncMock(
            side_effect=Exception("Docling processing error")
        )

        with patch(
            "FOSRABack.src.convo.utils.docling_helper.create_docling_service",
            return_value=mock_docling_service,
        ):
            result = await parser.parse(sample_pdf_path)

        assert result.content == ""
        assert "Docling processing error" in result.errors[0]


# ============================================================================
# UnstructuredParser Tests
# ============================================================================


class TestUnstructuredParser:
    """Tests for UnstructuredParser."""

    def test_parser_attributes(self):
        """Test parser class attributes."""
        parser = UnstructuredParser()

        assert parser.parser_type == ParserType.UNSTRUCTURED
        assert parser.display_name == "Unstructured"
        assert DocumentType.PDF in parser.supported_types
        assert DocumentType.TEXT in parser.supported_types
        assert DocumentType.MARKDOWN in parser.supported_types

    def test_supports_document_type(self):
        """Test document type support."""
        parser = UnstructuredParser()

        assert parser.supports_document_type(DocumentType.PDF) is True
        assert parser.supports_document_type(DocumentType.DOCX) is True
        assert parser.supports_document_type(DocumentType.TEXT) is True
        assert parser.supports_document_type(DocumentType.UNKNOWN) is False

    @pytest.mark.asyncio
    async def test_initialize_parser(self):
        """Test that initialization is a no-op."""
        parser = UnstructuredParser()
        await parser._initialize_parser()  # type: ignore[attr-defined]
        # Should complete without error

    @pytest.mark.asyncio
    async def test_parse_success(self, sample_text_path: str):
        """Test successful parsing with Unstructured."""
        parser = UnstructuredParser()

        # Mock the loader and transformer
        mock_doc = MagicMock()
        mock_doc.page_content = "Parsed content"
        mock_doc.metadata = {"page_number": 1}

        mock_loader = MagicMock()
        mock_loader.aload = AsyncMock(return_value=[mock_doc])

        mock_transformed_doc = MagicMock()
        mock_transformed_doc.page_content = "# Transformed content"

        with patch(
            "langchain_community.document_loaders.UnstructuredFileLoader",
            return_value=mock_loader,
        ):
            with patch(
                "langchain_community.document_transformers.MarkdownifyTransformer"
            ) as mock_transformer_cls:
                mock_transformer = MagicMock()
                mock_transformer.transform_documents.return_value = [
                    mock_transformed_doc
                ]
                mock_transformer_cls.return_value = mock_transformer

                result = await parser.parse(sample_text_path)

        assert "Transformed content" in result.content
        assert result.parser_used == "UNSTRUCTURED"
        assert result.parse_time_ms > 0

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, sample_text_path: str):
        """Test exception handling during parsing."""
        parser = UnstructuredParser()

        with patch(
            "langchain_community.document_loaders.UnstructuredFileLoader",
            side_effect=Exception("Unstructured error"),
        ):
            result = await parser.parse(sample_text_path)

        assert result.content == ""
        assert "Unstructured error" in result.errors[0]


# ============================================================================
# MarkdownParser Tests
# ============================================================================


class TestMarkdownParser:
    """Tests for MarkdownParser."""

    def test_parser_attributes(self):
        """Test parser class attributes."""
        parser = MarkdownParser()

        assert parser.parser_type == ParserType.MARKDOWN
        assert parser.display_name == "Markdown"
        assert DocumentType.MARKDOWN in parser.supported_types
        assert DocumentType.TEXT in parser.supported_types

    def test_supports_document_type(self):
        """Test document type support."""
        parser = MarkdownParser()

        assert parser.supports_document_type(DocumentType.MARKDOWN) is True
        assert parser.supports_document_type(DocumentType.TEXT) is True
        assert parser.supports_document_type(DocumentType.PDF) is False

    @pytest.mark.asyncio
    async def test_initialize_parser(self):
        """Test that initialization is a no-op."""
        parser = MarkdownParser()
        await parser._initialize_parser()  # type: ignore[attr-defined]
        # Should complete without error

    @pytest.mark.asyncio
    async def test_parse_markdown_file(self, sample_markdown_path: str):
        """Test parsing a markdown file."""
        parser = MarkdownParser()

        result = await parser.parse(sample_markdown_path)

        assert "# Test Document" in result.content
        assert "This is test content" in result.content
        assert result.parser_used == "MARKDOWN"
        assert result.parse_time_ms > 0
        assert result.page_count >= 1
        assert "file_size" in result.metadata

    @pytest.mark.asyncio
    async def test_parse_text_file(self, sample_text_path: str):
        """Test parsing a text file."""
        parser = MarkdownParser()

        result = await parser.parse(sample_text_path)

        assert "plain text content" in result.content
        assert result.parser_used == "MARKDOWN"
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self):
        """Test handling of missing file."""
        parser = MarkdownParser()

        result = await parser.parse("/nonexistent/path/file.md")

        assert result.content == ""
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_parse_page_count_calculation(self, tmp_path: Path):
        """Test page count calculation based on content length."""
        parser = MarkdownParser()

        # Create a large file (>6000 chars = 2 pages at 3000 chars/page)
        large_content = "A" * 6500
        large_file = tmp_path / "large.md"
        large_file.write_text(large_content)

        result = await parser.parse(str(large_file))

        assert result.page_count >= 2


# ============================================================================
# PyPDFParser Tests
# ============================================================================


class TestPyPDFParser:
    """Tests for PyPDFParser."""

    def test_parser_attributes(self):
        """Test parser class attributes."""
        parser = PyPDFParser()

        assert parser.parser_type == ParserType.PYPDF
        assert parser.display_name == "PyPDF"
        assert parser.supported_types == [DocumentType.PDF]

    def test_supports_document_type(self):
        """Test document type support."""
        parser = PyPDFParser()

        assert parser.supports_document_type(DocumentType.PDF) is True
        assert parser.supports_document_type(DocumentType.DOCX) is False

    @pytest.mark.asyncio
    async def test_initialize_parser(self):
        """Test that initialization is a no-op."""
        parser = PyPDFParser()
        await parser._initialize_parser()  # type: ignore[attr-defined]
        # Should complete without error

    @pytest.mark.asyncio
    async def test_parse_success(self, sample_pdf_path: str):
        """Test successful PDF parsing."""
        parser = PyPDFParser()

        # Mock pypdf
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 content"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page, mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            with patch("builtins.open", mock_open(read_data=b"fake pdf")):
                result = await parser.parse(sample_pdf_path)

        assert "Page 1 content" in result.content
        assert result.page_count == 2
        assert result.parser_used == "PYPDF"
        assert result.parse_time_ms > 0

    @pytest.mark.asyncio
    async def test_parse_with_max_pages(self, sample_pdf_path: str):
        """Test parsing with max_pages config."""
        parser = PyPDFParser()
        config = ParserConfig(max_pages=1)

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2, mock_page1]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            with patch("builtins.open", mock_open(read_data=b"fake pdf")):
                result = await parser.parse(sample_pdf_path, config)

        # Should only have content from page 1
        assert "Page 1" in result.content
        assert "Page 2" not in result.content

    @pytest.mark.asyncio
    async def test_parse_exception_handling(self, sample_pdf_path: str):
        """Test exception handling during parsing."""
        parser = PyPDFParser()

        with patch("pypdf.PdfReader", side_effect=Exception("PDF read error")):
            with patch("builtins.open", mock_open(read_data=b"fake pdf")):
                result = await parser.parse(sample_pdf_path)

        assert result.content == ""
        assert "PDF read error" in result.errors[0]


# ============================================================================
# LlamaParseParser Tests
# ============================================================================


class TestLlamaParseParser:
    """Tests for LlamaParseParser (API-based)."""

    def test_parser_attributes(self):
        """Test parser class attributes."""
        parser = LlamaParseParser()

        assert parser.parser_type == ParserType.LLAMA_PARSE
        assert parser.display_name == "LlamaParse"
        assert DocumentType.PDF in parser.supported_types

    def test_supports_document_type(self):
        """Test document type support."""
        parser = LlamaParseParser()

        assert parser.supports_document_type(DocumentType.PDF) is True
        assert parser.supports_document_type(DocumentType.DOCX) is True
        assert parser.supports_document_type(DocumentType.TEXT) is False

    @pytest.mark.asyncio
    async def test_parse_without_context(self, sample_pdf_path: str):
        """Test that parse fails without RequestContext."""
        parser = LlamaParseParser()

        result = await parser.parse(sample_pdf_path, ctx=None)

        assert result.content == ""
        assert "RequestContext required" in result.errors[0]

    @pytest.mark.asyncio
    async def test_parse_no_api_config(self, sample_pdf_path: str):
        """Test parsing without API configuration in context."""
        from FOSRABack.src.config.request_context import RequestContext, UserPreferences

        parser = LlamaParseParser()

        # Create context with default (empty) preferences
        ctx = RequestContext.create_simple(user_id="test", workspace_id=1)
        ctx.preferences.parser = None  # type: ignore

        result = await parser.parse(sample_pdf_path, ctx=ctx)

        assert result.content == ""
        assert "No API configuration" in result.errors[0]

    @pytest.mark.asyncio
    async def test_execute_api_parse_success(self, sample_pdf_path: str):
        """Test successful API parsing."""
        parser = LlamaParseParser()
        config = ParserConfig()
        api_config = {"LLAMA_PARSE_API_KEY": "test-key"}

        mock_doc = MagicMock()
        mock_doc.text = "Parsed content from LlamaParse"

        mock_llama_parse = MagicMock()
        mock_llama_parse.aload_data = AsyncMock(return_value=[mock_doc, mock_doc])

        with patch.dict("sys.modules", {"llama_parse": MagicMock()}):
            from llama_parse import LlamaParse

            with patch(
                "llama_parse.LlamaParse",
                return_value=mock_llama_parse,
            ):
                result = await parser._execute_api_parse(
                    sample_pdf_path, config, api_config
                )

        assert "Parsed content from LlamaParse" in result.content
        assert result.page_count == 2
        assert result.parser_used == "LLAMA_PARSE"

    @pytest.mark.asyncio
    async def test_execute_api_parse_no_api_key(self, sample_pdf_path: str):
        """Test API parse without API key."""
        parser = LlamaParseParser()
        config = ParserConfig()
        api_config = {}  # No API key

        with patch.dict("sys.modules", {"llama_parse": MagicMock()}):
            result = await parser._execute_api_parse(
                sample_pdf_path, config, api_config
            )

        assert result.content == ""
        assert "API key not configured" in result.errors[0]

    # @pytest.mark.asyncio
    # async def test_execute_api_parse_import_error(self, sample_pdf_path: str):
    #     """Test handling of missing llama-parse package."""
    #     parser = LlamaParseParser()
    #     config = ParserConfig()
    #     api_config = {"LLAMA_PARSE_API_KEY": "test-key"}
    #
    #     with patch(
    #         "builtins.__import__",
    #         side_effect=ModuleNotFoundError("No module named 'llama_parse'"),
    #     ):
    #         result = await parser._execute_api_parse(
    #             sample_pdf_path, config, api_config
    #         )
    #
    #     assert result.content == ""
    #     assert "not installed" in result.errors[0]


# ============================================================================
# ParserService Tests
# ============================================================================


class TestParserService:
    """Tests for ParserService orchestration."""

    def test_initialization(self, mock_session: AsyncMock):
        """Test service initialization."""
        service = ParserService(mock_session)

        assert service.session == mock_session
        assert len(service._default_parser_order) > 0

    def test_get_parser(self):
        """Test getting a parser by type."""
        service = ParserService()

        parser = service.get_parser(ParserType.MARKDOWN)

        assert parser is not None
        assert isinstance(parser, MarkdownParser)

    def test_get_available_parsers(self):
        """Test getting list of available parsers."""
        service = ParserService()

        parsers = service.get_available_parsers()

        assert "DOCLING" in parsers
        assert "MARKDOWN" in parsers
        assert "PYPDF" in parsers

    def test_get_best_parser_for_pdf(self, sample_pdf_path: str):
        """Test getting best parser for PDF."""
        service = ParserService()

        parser = service.get_best_parser_for_document(sample_pdf_path)

        # Should get Docling as it's first in order and supports PDF
        assert parser is not None
        assert parser.supports_document_type(DocumentType.PDF)

    def test_get_best_parser_for_markdown(self, sample_markdown_path: str):
        """Test getting best parser for markdown."""
        service = ParserService()

        parser = service.get_best_parser_for_document(sample_markdown_path)

        assert parser is not None
        assert parser.supports_document_type(DocumentType.MARKDOWN)

    def test_get_best_parser_with_preferred(self, sample_pdf_path: str):
        """Test getting parser with preference."""
        service = ParserService()

        parser = service.get_best_parser_for_document(
            sample_pdf_path, preferred_parser=ParserType.PYPDF
        )

        assert parser is not None
        assert isinstance(parser, PyPDFParser)

    def test_get_best_parser_preferred_not_supported(self, sample_markdown_path: str):
        """Test fallback when preferred parser doesn't support document type."""
        service = ParserService()

        # PyPDF doesn't support markdown
        parser = service.get_best_parser_for_document(
            sample_markdown_path, preferred_parser=ParserType.PYPDF
        )

        # Should fall back to a parser that supports markdown
        assert parser is not None
        assert parser.supports_document_type(DocumentType.MARKDOWN)

    def test_detect_document_type(self):
        """Test internal document type detection."""
        service = ParserService()

        assert service._detect_document_type("pdf") == DocumentType.PDF
        assert service._detect_document_type("docx") == DocumentType.DOCX
        assert service._detect_document_type("md") == DocumentType.MARKDOWN
        assert service._detect_document_type("txt") == DocumentType.TEXT
        assert service._detect_document_type("xyz") == DocumentType.UNKNOWN

    @pytest.mark.asyncio
    async def test_parse_document_success(self, sample_markdown_path: str):
        """Test successful document parsing."""
        service = ParserService()

        result = await service.parse_document(sample_markdown_path)

        assert result.content != ""
        assert result.errors == []
        # ParserService uses _default_parser_order, UNSTRUCTURED is before MARKDOWN
        assert result.parser_used == "UNSTRUCTURED"

    @pytest.mark.asyncio
    async def test_parse_document_no_parser_available(self, tmp_path: Path):
        """Test parsing when no parser is available."""
        service = ParserService()

        # Create file with unsupported extension
        unknown_file = tmp_path / "file.xyz123"
        unknown_file.touch()

        result = await service.parse_document(str(unknown_file))

        assert result.content == ""
        assert "No parser available" in result.errors[0]

    @pytest.mark.asyncio
    async def test_parse_document_with_fallback(self, sample_pdf_path: str):
        """Test fallback parsing when primary fails."""
        service = ParserService()
        config = ParserConfig(fallback_parsers=[ParserType.PYPDF])

        # Mock Docling to fail, PyPDF to succeed
        mock_docling_service = MagicMock()
        mock_docling_service.process_document = AsyncMock(
            side_effect=Exception("Docling failed")
        )

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Fallback content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch(
            "FOSRABack.src.convo.utils.docling_helper.create_docling_service",
            return_value=mock_docling_service,
        ):
            with patch("pypdf.PdfReader", return_value=mock_reader):
                with patch("builtins.open", mock_open(read_data=b"fake pdf")):
                    result = await service.parse_document(
                        sample_pdf_path, config=config
                    )

        assert "Fallback content" in result.content
        assert result.parser_used == "PYPDF"

    @pytest.mark.asyncio
    async def test_parse_document_timeout(self, sample_markdown_path: str):
        """Test timeout handling during parsing."""
        service = ParserService()
        config = ParserConfig(timeout_seconds=0)  # Immediate timeout

        # Create a slow parser mock
        async def slow_parse(*args, **kwargs):
            await asyncio.sleep(10)
            return ParsedDocument(content="Should not reach here")

        with patch.object(MarkdownParser, "parse", slow_parse):
            result = await service.parse_document(sample_markdown_path, config=config)

        assert result.content == ""
        assert "timeout" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_parse_document_with_preferred_parser(self, sample_pdf_path: str):
        """Test parsing with preferred parser."""
        service = ParserService()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PyPDF parsed content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            with patch("builtins.open", mock_open(read_data=b"fake pdf")):
                result = await service.parse_document(
                    sample_pdf_path, preferred_parser=ParserType.PYPDF
                )

        assert result.parser_used == "PYPDF"
        assert "PyPDF parsed content" in result.content

    @pytest.mark.asyncio
    async def test_execute_parse_exception(self, sample_pdf_path: str):
        """Test exception handling in _execute_parse."""
        service = ParserService()
        parser = service.get_parser(ParserType.PYPDF)
        config = ParserConfig()

        assert parser is not None  # Type guard
        with patch.object(
            parser, "parse", side_effect=RuntimeError("Unexpected error")
        ):
            result = await service._execute_parse(parser, sample_pdf_path, config, None)

        assert result.content == ""
        assert "Unexpected error" in result.errors[0]


# ============================================================================
# Integration Tests
# ============================================================================


class TestParserServiceIntegration:
    """Integration tests for the parser service."""

    @pytest.mark.asyncio
    async def test_full_markdown_parsing_flow(self, sample_markdown_path: str):
        """Test complete markdown parsing workflow."""
        service = ParserService()
        config = ParserConfig(extract_tables=True)

        result = await service.parse_document(sample_markdown_path, config=config)

        assert result.content != ""
        assert result.parser_used == "MARKDOWN"
        assert result.parse_time_ms > 0
        assert result.page_count >= 1
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_parser_registry_consistency(self):
        """Test that all registered parsers are properly configured."""
        available_parsers = BaseParser.get_available_parsers()

        for parser_type_str in available_parsers:
            parser_type = ParserType(parser_type_str)
            parser = BaseParser.get_parser(parser_type)

            assert parser is not None
            assert hasattr(parser, "parser_type")
            assert hasattr(parser, "display_name")
            assert hasattr(parser, "supported_types")
            assert hasattr(parser, "parse")
            assert hasattr(parser, "supports_document_type")

    @pytest.mark.asyncio
    async def test_multiple_parsers_same_document(self, sample_markdown_path: str):
        """Test parsing same document with multiple parsers."""
        service = ParserService()

        # Parse with Markdown parser
        markdown_parser = service.get_parser(ParserType.MARKDOWN)
        result1 = await markdown_parser.parse(sample_markdown_path)

        # Parse with Unstructured (which also supports markdown)
        # Mock it to avoid actual unstructured dependency
        mock_doc = MagicMock()
        mock_doc.page_content = "Unstructured content"
        mock_doc.metadata = {}

        mock_loader = MagicMock()
        mock_loader.aload = AsyncMock(return_value=[mock_doc])

        mock_transformed = MagicMock()
        mock_transformed.page_content = "Transformed"

        with patch(
            "langchain_community.document_loaders.UnstructuredFileLoader",
            return_value=mock_loader,
        ):
            with patch(
                "langchain_community.document_transformers.MarkdownifyTransformer"
            ) as mock_cls:
                mock_transformer = MagicMock()
                mock_transformer.transform_documents.return_value = [mock_transformed]
                mock_cls.return_value = mock_transformer

                unstructured_parser = service.get_parser(ParserType.UNSTRUCTURED)
                result2 = await unstructured_parser.parse(sample_markdown_path)

        # Both should succeed
        assert result1.errors == []
        assert result2.errors == []
        assert result1.parser_used in ["MARKDOWN", "UNSTRUCTURED"]
        assert result2.parser_used in ["MARKDOWN", "UNSTRUCTURED"]
