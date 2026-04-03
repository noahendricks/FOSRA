import pytest
import sys

sys.path.insert(0, "backend/src")
from logging_config import _process_record


class TestShortName:
    def test_short_name_two_parts(self):
        record = {"name": "api.routes.oc.state", "extra": {}}
        _process_record(record)
        assert record["extra"]["short_name"] == "oc.state"

    def test_short_name_single_part(self):
        record = {"name": "main", "extra": {}}
        _process_record(record)
        assert record["extra"]["short_name"] == "main"

    def test_short_name_deep_module(self):
        record = {"name": "a.b.c.d.e", "extra": {}}
        _process_record(record)
        assert record["extra"]["short_name"] == "d.e"

    def test_short_name_three_parts(self):
        record = {"name": "api.routes.oc.session_ops", "extra": {}}
        _process_record(record)
        assert record["extra"]["short_name"] == "oc.session_ops"


class TestStructuredRendering:
    def test_structured_renders_rich_data(self):
        record = {
            "name": "test.module",
            "extra": {"_structured": {"key": {"nested": 1}}},
        }
        _process_record(record)
        assert "_rich_data" in record["extra"]
        assert "key" in record["extra"]["_rich_data"]
        assert "nested" in record["extra"]["_rich_data"]

    def test_structured_primitives_use_repr(self):
        record = {"name": "test.module", "extra": {"_structured": {"count": 42}}}
        _process_record(record)
        assert "_rich_data" in record["extra"]
        assert "count" in record["extra"]["_rich_data"]

    def test_structured_is_popped(self):
        record = {"name": "test.module", "extra": {"_structured": {"key": "value"}}}
        _process_record(record)
        assert "_structured" not in record["extra"]

    def test_no_structured_no_rich_data(self):
        record = {"name": "test.module", "extra": {}}
        _process_record(record)
        assert record["extra"]["_rich_data"] == ""
