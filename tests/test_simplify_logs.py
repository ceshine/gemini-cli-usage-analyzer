import logging
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from typing import Any

import pytest
import orjson

from gemini_cli_usage_analyzer.simplify_logs import simplify_record, run_log_simplification


@pytest.fixture
def sample_log_records():
    """Provides a set of diverse sample log records for testing."""
    return [
        # api_response - with all fields for level 3 testing
        {
            "attributes": {
                "event.name": "gemini_cli.api_response",
                "event.timestamp": "2023-01-01T10:00:00Z",
                "duration_ms": 100,
                "input_token_count": 10,
                "output_token_count": 20,
                "cached_content_token_count": 5,
                "thoughts_token_count": 2,
                "total_token_count": 37,
                "tool_token_count": 0,
                "model": "gemini-pro",
                "session.id": "session123",
                "extra_attr_response": "value1",
            },
            "_body": {"response_body": "data"},
            "other_field": "ignore_me",
        },
        # api_request
        {
            "attributes": {
                "event.name": "gemini_cli.api_request",
                "event.timestamp": "2023-01-01T10:01:00Z",
                "request_id": "req456",
                "model": "gemini-pro",
                "extra_attr_request": "value2",
            },
            "_body": {"request_body": "data"},
        },
        # non-api event
        {
            "attributes": {
                "event.name": "gemini_cli.start_up",
                "event.timestamp": "2023-01-01T09:59:00Z",
            },
            "_body": {"startup_info": "data"},
        },
    ]


@pytest.fixture
def create_jsonl_file(tmp_path, sample_log_records):
    """Fixture to create a JSONL file with sample records."""

    def _create(filename="test.jsonl", records=None):
        file_path = tmp_path / filename
        records_to_write = records if records is not None else sample_log_records
        with open(file_path, "wb") as f:
            for record in records_to_write:
                # Ensure only dicts are passed for general creation
                assert isinstance(record, dict), "create_jsonl_file expects dictionary records"
                f.write(orjson.dumps(record) + b"\n")
        return file_path

    return _create


# Tests for simplify_record (existing tests, kept for completeness)
def test_simplify_record_none():
    record = {"some": "data", "attributes": {"event.name": "irrelevant"}}
    result = simplify_record(record, 0)
    assert result == record


def test_simplify_record_events_only_keep_response():
    record = {"attributes": {"event.name": "gemini_cli.api_response"}}
    result = simplify_record(record, 1)
    assert result == record


def test_simplify_record_events_only_keep_request():
    record = {"attributes": {"event.name": "gemini_cli.api_request"}}
    result = simplify_record(record, 1)
    assert result == record


def test_simplify_record_events_only_discard_other():
    record = {"attributes": {"event.name": "other_event"}}
    result = simplify_record(record, 1)
    assert result is None


def test_simplify_record_events_only_no_attributes():
    record: dict[str, Any] = {"other": "data"}
    result = simplify_record(record, 1)
    assert result is None


def test_simplify_record_level_2():
    record = {
        "attributes": {"event.name": "gemini_cli.api_response", "other_attr": "keep_me"},
        "_body": "keep_this_body",
        "extra_field": "discard_me",
    }
    result = simplify_record(record, 2)
    assert result == {
        "attributes": {"event.name": "gemini_cli.api_response", "other_attr": "keep_me"},
        "_body": "keep_this_body",
    }


def test_simplify_record_level_3_happy_path():
    attributes = {
        "event.name": "gemini_cli.api_response",
        "event.timestamp": "2023-10-26T12:00:00Z",
        "duration_ms": 100,
        "input_token_count": 10,
        "output_token_count": 20,
        "cached_content_token_count": 0,
        "thoughts_token_count": 5,
        "total_token_count": 30,
        "tool_token_count": 0,
        "model": "gemini-pro",
        "session.id": "123",
        "extra_attr": "discard_me",
    }
    record = {
        "attributes": attributes,
        "_body": "keep_this_body",
        "extra_field": "discard_me",
    }
    result = simplify_record(record, 3)

    expected_attributes = {
        "event.name": "gemini_cli.api_response",
        "event.timestamp": "2023-10-26T12:00:00Z",
        "duration_ms": 100,
        "input_token_count": 10,
        "output_token_count": 20,
        "cached_content_token_count": 0,
        "thoughts_token_count": 5,
        "total_token_count": 30,
        "tool_token_count": 0,
        "model": "gemini-pro",
        "session.id": "123",
    }

    assert result is not None
    assert result["_body"] == "keep_this_body"
    assert result["attributes"] == expected_attributes
    assert "extra_field" not in result


def test_simplify_record_level_3_filter_event():
    record = {
        "attributes": {"event.name": "gemini_cli.api_request"},
        "_body": "body",
    }
    result = simplify_record(record, 3)
    assert result is None


# New tests for run_log_simplification
def test_run_log_simplification_level0(create_jsonl_file, sample_log_records, tmp_path, caplog):
    input_file = create_jsonl_file()  # Use fixture default records
    with open(input_file, "rb") as f_orig:
        output_file_original_content = [orjson.loads(line) for line in f_orig]

    with caplog.at_level(logging.WARNING):
        exit_code = run_log_simplification(input_file, level=0, disable_archiving=True)

    assert exit_code == 0
    assert "Level 0 is a no-op. Exiting." in caplog.text
    with open(input_file, "rb") as f:
        simplified_records = [orjson.loads(line) for line in f]
    assert simplified_records == output_file_original_content


def test_run_log_simplification_level1(create_jsonl_file, sample_log_records, tmp_path):
    input_file = create_jsonl_file()
    exit_code = run_log_simplification(input_file, level=1, disable_archiving=True)
    assert exit_code == 0
    with open(input_file, "rb") as f:
        simplified_records = [orjson.loads(line) for line in f]

    expected_records = [
        r
        for r in sample_log_records
        if isinstance(r, dict)
        and r.get("attributes", {}).get("event.name") in ("gemini_cli.api_response", "gemini_cli.api_request")
    ]
    assert simplified_records == expected_records


def test_run_log_simplification_level2(create_jsonl_file, sample_log_records, tmp_path):
    input_file = create_jsonl_file()
    exit_code = run_log_simplification(input_file, level=2, disable_archiving=True)
    assert exit_code == 0
    with open(input_file, "rb") as f:
        simplified_records = [orjson.loads(line) for line in f]

    expected_records = []
    for r in sample_log_records:
        if isinstance(r, dict) and r.get("attributes", {}).get("event.name") in (
            "gemini_cli.api_response",
            "gemini_cli.api_request",
        ):
            expected_records.append({"attributes": r["attributes"], "_body": r["_body"]})
    assert simplified_records == expected_records


def test_run_log_simplification_level3(create_jsonl_file, sample_log_records, tmp_path):
    input_file = create_jsonl_file()
    exit_code = run_log_simplification(input_file, level=3, disable_archiving=True)
    assert exit_code == 0
    with open(input_file, "rb") as f:
        simplified_records = [orjson.loads(line) for line in f]

    expected_records = []
    for r in sample_log_records:
        if isinstance(r, dict) and r.get("attributes", {}).get("event.name") == "gemini_cli.api_response":
            expected_attributes = {
                "event.timestamp": r["attributes"]["event.timestamp"],
                "duration_ms": r["attributes"]["duration_ms"],
                "input_token_count": r["attributes"]["input_token_count"],
                "output_token_count": r["attributes"]["output_token_count"],
                "cached_content_token_count": r["attributes"]["cached_content_token_count"],
                "thoughts_token_count": r["attributes"]["thoughts_token_count"],
                "total_token_count": r["attributes"]["total_token_count"],
                "tool_token_count": r["attributes"]["tool_token_count"],
                "model": r["attributes"]["model"],
                "session.id": r["attributes"]["session.id"],
                "event.name": r["attributes"]["event.name"],
            }
            expected_records.append({"attributes": expected_attributes, "_body": r["_body"]})
    assert simplified_records == expected_records


def test_run_log_simplification_file_not_found(tmp_path):
    non_existent_file = tmp_path / "non_existent.jsonl"
    with pytest.raises(FileNotFoundError):
        run_log_simplification(non_existent_file, level=1)


def test_run_log_simplification_not_jsonl_file(tmp_path):
    text_file = tmp_path / "test.txt"
    text_file.write_text("some text")
    with pytest.raises(ValueError, match="Input file must be a .jsonl file"):
        run_log_simplification(text_file, level=1)


def test_run_log_simplification_temp_file_exists(create_jsonl_file, tmp_path):
    input_file = create_jsonl_file()
    temp_file = input_file.with_suffix(".jsonl.tmp")
    temp_file.touch()  # Create an empty temp file

    with pytest.raises(ValueError, match="Temp file .* already exists"):
        run_log_simplification(input_file, level=1)


@patch("gemini_cli_usage_analyzer.simplify_logs.datetime")
def test_run_log_simplification_archive_file_exists(mock_datetime, create_jsonl_file, tmp_path):
    mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
    mock_datetime.timestamp.return_value = 1672567200.0

    input_file = create_jsonl_file()
    archive_folder = tmp_path / "archive"
    archive_folder.mkdir()
    # Ensure the timestamped archive file already exists
    (archive_folder / f"test.{int(mock_datetime.now().timestamp())}.jsonl").touch()

    with pytest.raises(ValueError, match="Archive file .* already exists"):
        run_log_simplification(input_file, level=1, archive_folder=archive_folder, disable_archiving=False)


@patch("gemini_cli_usage_analyzer.simplify_logs.datetime")
def test_run_log_simplification_archiving_enabled(mock_datetime, create_jsonl_file, sample_log_records, tmp_path):
    mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
    mock_datetime.timestamp.return_value = 1672567200.0

    input_file = create_jsonl_file()
    original_input_content = Path(input_file).read_bytes()  # Capture original content

    archive_folder = tmp_path / "archive"
    archive_folder.mkdir()

    exit_code = run_log_simplification(input_file, level=1, disable_archiving=False, archive_folder=archive_folder)
    assert exit_code == 0

    # The original content should be in the archive folder
    archive_file = archive_folder / f"test.{int(mock_datetime.now().timestamp())}.jsonl"
    assert archive_file.exists()
    assert archive_file.read_bytes() == original_input_content

    # The input_file now contains the simplified version (Level 1)
    assert input_file.exists()
    with open(input_file, "rb") as f:
        simplified_records = [orjson.loads(line) for line in f]

    expected_records = [
        r
        for r in sample_log_records
        if isinstance(r, dict)
        and r.get("attributes", {}).get("event.name") in ("gemini_cli.api_response", "gemini_cli.api_request")
    ]
    assert simplified_records == expected_records


def test_run_log_simplification_archiving_disabled(create_jsonl_file, sample_log_records, tmp_path):
    input_file = create_jsonl_file()

    exit_code = run_log_simplification(input_file, level=1, disable_archiving=True)
    assert exit_code == 0
    # input_file should exist and contain the simplified content
    assert input_file.exists()

    with open(input_file, "rb") as f:
        simplified_records = [orjson.loads(line) for line in f]

    expected_records = [
        r
        for r in sample_log_records
        if isinstance(r, dict)
        and r.get("attributes", {}).get("event.name") in ("gemini_cli.api_response", "gemini_cli.api_request")
    ]
    assert simplified_records == expected_records


def test_run_log_simplification_malformed_records(tmp_path, caplog):
    # Create a file with malformed JSONL content directly

    input_file = tmp_path / "malformed.jsonl"

    malformed_lines = [
        orjson.dumps({"attributes": {"event.name": "gemini_cli.api_response"}}),
        orjson.dumps({"attributes": {"event.name": "gemini_cli.api_request"}}),
        orjson.dumps(
            {"_body": "no attributes"}
        ),  # Malformed for simplify_record filter (missing attributes.event.name)
    ]

    with open(input_file, "wb") as f:
        _ = f.write(b"\n".join(malformed_lines))

    with caplog.at_level(logging.WARNING):
        # Now running with level=1, assuming graceful handling of malformed JSON
        exit_code = run_log_simplification(input_file, level=1, disable_archiving=True)

    # Expecting successful completion (exit_code 0) if malformed lines are skipped gracefully
    assert exit_code == 0

    # Expecting warnings for the malformed lines and for the record filtered by simplify_record
    assert (
        "Skipping record with no `attributes` field" in caplog.text
    )  # This needs to match the exact log from run_log_simplification.

    # Read the processed file to check its content
    with open(input_file, "rb") as f:
        simplified_records = [orjson.loads(line) for line in f]

    # Expecting only the valid and kept records (Level 1 keeps api_response and api_request)
    expected_records = [
        orjson.loads(malformed_lines[0]),  # api_response
        orjson.loads(malformed_lines[1]),  # api_request
    ]

    assert simplified_records == expected_records
