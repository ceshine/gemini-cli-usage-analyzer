"""Module for simplifying log records."""

from enum import IntEnum
from typing import Any, Dict, Optional


class SimplificationLevel(IntEnum):
    """Enumeration for different levels of log simplification."""

    NONE = 0
    EVENTS_ONLY = 1


def simplify_record(record: Dict[str, Any], level: int) -> Optional[Dict[str, Any]]:
    """Processes a log record based on the simplification level.

    Args:
        record (Dict[str, Any]): The log record to process.
        level (int): The simplification level to apply.

    Returns:
        Optional[Dict[str, Any]]: The processed record if it should be kept (potentially modified),
            or None if it should be discarded.
    """
    if level == SimplificationLevel.NONE:
        return record

    # Level 1: Filter for specific events
    if level >= SimplificationLevel.EVENTS_ONLY:
        attributes = record.get("attributes", {})
        event_name = attributes.get("event.name")
        if event_name not in ("gemini_cli.api_response", "gemini_cli.api_request"):
            return None

    return record
