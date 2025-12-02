import json
import orjson
from gemini_cli_usage_analyzer.convert_logs import convert_log_file, get_last_timestamp


def test_get_last_timestamp_empty(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.touch()
    assert get_last_timestamp(f) is None


def test_get_last_timestamp_basic(tmp_path):
    f = tmp_path / "test.jsonl"
    entries = [
        {"attributes": {"event.timestamp": "2023-01-01T00:00:00"}},
        {"attributes": {"event.timestamp": "2023-01-01T00:00:01"}},
    ]
    with open(f, "wb") as out:
        for e in entries:
            out.write(orjson.dumps(e) + b"\n")

    assert get_last_timestamp(f) == "2023-01-01T00:00:01"


def test_get_last_timestamp_corrupt_last_line(tmp_path):
    f = tmp_path / "corrupt.jsonl"
    entries = [
        {"attributes": {"event.timestamp": "2023-01-01T00:00:00"}},
    ]
    with open(f, "wb") as out:
        for e in entries:
            out.write(orjson.dumps(e) + b"\n")
        out.write(b"{incompl")  # corrupted line

    # The current implementation might fail if it expects strictly one line per chunk logic or newlines.
    # But the logic splits by newline.

    # Wait, get_last_timestamp reads from end.
    # If it finds newline, it checks last part. If last part is valid JSON, uses it.
    # If not, it checks previous parts.

    assert get_last_timestamp(f) == "2023-01-01T00:00:00"


def test_convert_log_file_basic(tmp_path):
    input_file = tmp_path / "input.log"
    output_file = tmp_path / "output.jsonl"

    # Create input file with concatenated JSON (pretty printed or just separated)
    # The tool supports concatenated objects that might span lines.

    data = [
        {"attributes": {"event.timestamp": "t1", "event.name": "e1"}},
        {"attributes": {"event.timestamp": "t2", "event.name": "e2"}},
    ]

    # Emulate the format:
    # {
    #   ...
    # }
    # {
    #   ...
    # }

    with open(input_file, "w") as f:
        for item in data:
            f.write(json.dumps(item, indent=2))
            f.write("\n")

    count, skipped = convert_log_file(input_file, output_file)

    assert count == 2
    assert skipped == 0

    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["attributes"]["event.timestamp"] == "t1"


def test_convert_log_file_incremental(tmp_path):
    input_file = tmp_path / "input.log"
    output_file = tmp_path / "output.jsonl"

    data = [
        {"attributes": {"event.timestamp": "t1"}},
        {"attributes": {"event.timestamp": "t2"}},
        {"attributes": {"event.timestamp": "t3"}},
    ]

    with open(input_file, "w") as f:
        for item in data:
            f.write(json.dumps(item, indent=2))  # Multiline json
            f.write("\n")

    # Simulate existing output up to t2
    last_ts = "t2"

    count, skipped = convert_log_file(input_file, output_file, last_timestamp=last_ts)

    assert count == 1  # Only t3
    assert skipped == 2  # t1 and t2

    # Check content (should append t3)
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["attributes"]["event.timestamp"] == "t3"


def test_convert_log_file_simplify(tmp_path):
    input_file = tmp_path / "input.log"
    output_file = tmp_path / "output.jsonl"

    data = [
        {"attributes": {"event.timestamp": "t1", "event.name": "gemini_cli.api_response"}},
        {"attributes": {"event.timestamp": "t2", "event.name": "other"}},
    ]

    with open(input_file, "w") as f:
        for item in data:
            f.write(json.dumps(item, indent=2))
            f.write("\n")

    count, skipped = convert_log_file(input_file, output_file, simplify_level=1)

    assert count == 1
    assert skipped == 0

    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["attributes"]["event.timestamp"] == "t1"
