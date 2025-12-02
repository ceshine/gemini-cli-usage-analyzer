from typing import Any
from gemini_cli_usage_analyzer.simplify_logs import simplify_record


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
