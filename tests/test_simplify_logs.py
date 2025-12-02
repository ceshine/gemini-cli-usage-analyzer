from typing import Any
from gemini_cli_usage_analyzer.simplify_logs import SimplificationLevel, simplify_record

def test_simplify_record_none():
    record = {"some": "data", "attributes": {"event.name": "irrelevant"}}
    result = simplify_record(record, SimplificationLevel.NONE)
    assert result == record

def test_simplify_record_events_only_keep_response():
    record = {"attributes": {"event.name": "gemini_cli.api_response"}}
    result = simplify_record(record, SimplificationLevel.EVENTS_ONLY)
    assert result == record

def test_simplify_record_events_only_keep_request():
    record = {"attributes": {"event.name": "gemini_cli.api_request"}}
    result = simplify_record(record, SimplificationLevel.EVENTS_ONLY)
    assert result == record

def test_simplify_record_events_only_discard_other():
    record = {"attributes": {"event.name": "other_event"}}
    result = simplify_record(record, SimplificationLevel.EVENTS_ONLY)
    assert result is None

def test_simplify_record_events_only_no_attributes():
    record: dict[str, Any] = {"other": "data"}
    result = simplify_record(record, SimplificationLevel.EVENTS_ONLY)
    assert result is None
