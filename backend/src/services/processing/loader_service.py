from __future__ import annotations

from pathlib import Path

from loguru import logger
from ulid import ULID

from backend.src.domain.schemas.doc import Doc, MDNFile
from backend.src.services.processing.utils.parse_utils import code_mimes, text_mimes


def to_bytes(path: str):
    with open(path, "rb") as f:
        return f.read()


def ulid_factory() -> str:
    """Generate a new ULID string."""
    return str(ULID())


class LoaderService:
    @staticmethod
    def _parse_files(files: list[str | Path | MDNFile]) -> list[Doc]:
        from content_types import get_content_type as get_mime

        docs = []

        for file in files:
            if isinstance(file, Path):
                file = file.as_posix()

            if isinstance(file, str):
                mime_type = get_mime(file)
                logger.debug("Detected mime type: {}", mime_type)
            elif isinstance(file, MDNFile):
                raise NotImplementedError("MDNFiles cant be ingested yet")
            else:
                raise RuntimeError(f"Incorrect File Type at Ingestion: {type(file)}")

            match mime_type:
                case "application/pdf":
                    from backend.src.services.processing.loaders.docling_loader import (
                        DoclingLoader,
                        DoclingParseError,
                    )

                    try:
                        d = DoclingLoader.parse_file_sync(file, mime_type=mime_type)

                        docs.append(d)
                        logger.debug(
                            "created (docling) {}: {}: {}",
                            d.metadata.mime_type,
                            d.metadata.source,
                            d.id,
                        )
                        continue
                    except DoclingParseError:
                        raise
                    except Exception as ex:
                        logger.warning(
                            "Docling failed for {}, falling back to PyMuPDFParser: {}",
                            file,
                            ex,
                        )

                        # removed fallback, may replace or
                        pass

                case "text/plain":
                    import tempfile

                    from backend.src.services.processing.loaders.docling_loader import (
                        DoclingLoader,
                        DoclingParseError,
                    )

                    from backend.src.services.processing.utils.docling_regex import (
                        _clean_md,
                        _infer_chapters,
                    )

                    try:
                        text = Path(file).read_text(encoding="utf-8", errors="replace")

                        text = _infer_chapters(text)

                        text = _clean_md(text)

                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".md", delete=False
                        ) as tmp:
                            tmp.write(text)
                            tmp_path = tmp.name

                        try:
                            d = DoclingLoader.parse_file_sync(
                                tmp_path, mime_type="text/markdown"
                            )
                            # update source to original file
                            d.metadata.source = file
                            docs.append(d)
                            logger.debug(
                                "created (fast path) {}: {}",
                                d.metadata.source,
                                d.id,
                            )
                        finally:
                            Path(tmp_path).unlink(missing_ok=True)
                        continue
                    except Exception as ex:
                        logger.warning(
                            "Fast path failed for {}, falling back to docling: {}",
                            file,
                            ex,
                        )
                        # Fall through to docling

                case "text/markdown":
                    from backend.src.services.processing.loaders.docling_loader import (
                        DoclingLoader,
                        DoclingParseError,
                    )

                    try:
                        d = DoclingLoader.parse_file_sync(file, mime_type=mime_type)
                        docs.append(d)
                        logger.debug(
                            "created (docling) {}: {}: {}",
                            d.metadata.mime_type,
                            d.metadata.source,
                            d.id,
                        )
                        continue
                    except DoclingParseError:
                        raise
                    except Exception as ex:
                        logger.warning(
                            "Docling failed for {}, file: {}",
                            file,
                            ex,
                        )

                case _ if mime_type in code_mimes:
                    # NOTE: TextParser undefined - placeholder for future implementation
                    pass

                case _:
                    pass

        return docs

    @staticmethod
    def _parse_directory(dir_path: str | Path) -> list[Doc]:
        dir = Path(dir_path)
        file_limit = 100
        all_files_count = sum(1 for _ in dir.rglob("*") if _.is_file())

        files_list = []

        if all_files_count > file_limit:
            raise ValueError(
                f"Loader Error [Too Many Files]: The path [{dir_path}] has {all_files_count}, {all_files_count - file_limit} over the limit of {file_limit} "
            )

        else:
            for file_path in dir.rglob("*"):
                logger.debug("Found file: {}", file_path.as_posix())

                if file_path.is_file():
                    files_list.append(file_path)

            if files_list:
                logger.info("Running parse on {} files", len(files_list))
                all_files = LoaderService()._parse_files(files_list)
            else:
                all_files = []

            return all_files

    # public method
    @staticmethod
    def parse_user_paths(user_paths: list[str]) -> list[Doc]:
        files_as_docs = []
        loose_files = []
        for path in user_paths:
            p = Path(path)
            if p.is_dir():
                logger.info("Processing directory: {}", p.as_posix())

                dir_files_list: list[Doc] = LoaderService()._parse_directory(dir_path=p)

                files_as_docs.append(dir_files_list)
            if p.is_file():
                loose_files.append(p)

        if loose_files:
            files_as_docs.append(LoaderService()._parse_files(loose_files))

        logger.debug("Returning {} docs", len(files_as_docs))
        return files_as_docs
