import json
import pytest
from datetime import date
from zoneinfo import ZoneInfo
from gemini_cli_usage_analyzer.calculate_token_usage import (
    calculate_cost,
    process_log_file,
    UsageStats,
    analyze_token_usage,
)


@pytest.fixture
def price_spec():
    return {
        "model-a": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 0.5e-6,
            "input_cost_per_token_above_200k_tokens": 1.5e-6,
            "output_cost_per_token_above_200k_tokens": 2.5e-6,
            "cache_read_input_token_cost_above_200k_tokens": 0.75e-6,
        }
    }


def test_calculate_cost_basic(price_spec):
    attributes = {
        "model": "model-a",
        "input_token_count": 1000,
        "output_token_count": 500,
        "cached_content_token_count": 0,
        "thoughts_token_count": 0,
    }
    # cost = 1000 * 1e-6 + 500 * 2e-6 = 0.001 + 0.001 = 0.002
    cost = calculate_cost(attributes, price_spec)
    assert cost == pytest.approx(0.002)


def test_calculate_cost_cached(price_spec):
    attributes = {
        "model": "model-a",
        "input_token_count": 1000,
        "output_token_count": 0,
        "cached_content_token_count": 400,
        "thoughts_token_count": 0,
    }
    # effective input = 1000 - 400 = 600
    # cost = 600 * 1e-6 + 400 * 0.5e-6 = 0.0006 + 0.0002 = 0.0008
    cost = calculate_cost(attributes, price_spec)
    assert cost == pytest.approx(0.0008)


def test_calculate_cost_tiered(price_spec):
    attributes = {
        "model": "model-a",
        "input_token_count": 250000,
        "output_token_count": 100,
        "cached_content_token_count": 0,
        "thoughts_token_count": 0,
    }
    # tiered prices apply
    # cost = 250000 * 1.5e-6 + 100 * 2.5e-6 = 0.375 + 0.00025 = 0.37525
    cost = calculate_cost(attributes, price_spec)
    assert cost == pytest.approx(0.37525)


def test_calculate_cost_thoughts(price_spec):
    attributes = {
        "model": "model-a",
        "input_token_count": 0,
        "output_token_count": 100,
        "cached_content_token_count": 0,
        "thoughts_token_count": 50,
    }
    # cost = 150 (output+thoughts) * 2e-6 = 0.0003
    cost = calculate_cost(attributes, price_spec)
    assert cost == pytest.approx(0.0003)


def test_calculate_cost_unknown_model(price_spec):
    attributes = {
        "model": "unknown-model",
        "input_token_count": 1000,
    }
    # defaults to 0
    cost = calculate_cost(attributes, price_spec)
    assert cost == 0.0


def test_process_log_file(tmp_path, price_spec):
    log_file = tmp_path / "test.jsonl"
    entries = [
        {
            "attributes": {
                "event.name": "gemini_cli.api_response",
                "model": "model-a",
                "input_token_count": 1000,
                "output_token_count": 500,
                "event.timestamp": "2023-10-26T10:00:00+00:00",
            }
        },
        {
            "attributes": {
                "event.name": "other_event",
            }
        },
        {
            "attributes": {
                "event.name": "gemini_cli.api_response",
                "model": "model-a",
                "input_token_count": 2000,
                "output_token_count": 100,
                "event.timestamp": "2023-10-26T23:00:00+00:00",
            }
        },
        {
            "attributes": {
                "event.name": "gemini_cli.api_response",
                "model": "model-a",
                "input_token_count": 100,
                "output_token_count": 50,
                "event.timestamp": "2023-10-27T01:00:00+00:00",
            }
        },
    ]
    with open(log_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    usage, count, error = process_log_file(log_file, price_spec, timezone=ZoneInfo("UTC"))

    assert count == 3
    assert not error

    key1 = ("model-a", date(2023, 10, 26))
    key2 = ("model-a", date(2023, 10, 27))

    assert key1 in usage
    assert key2 in usage

    stats1 = usage[key1]
    assert isinstance(stats1, UsageStats)
    assert stats1.count == 2
    assert stats1.input_tokens == 3000
    assert stats1.output_tokens == 600
    assert stats1.cost == pytest.approx(0.0042)

    stats2 = usage[key2]
    assert stats2.count == 1
    assert stats2.input_tokens == 100
    assert stats2.output_tokens == 50


def test_process_log_file_timezone_shift(tmp_path, price_spec):
    log_file = tmp_path / "test_tz.jsonl"
    # 2023-10-27T01:00:00Z is 2023-10-26 20:00:00 in UTC-5
    entry = {
        "attributes": {
            "event.name": "gemini_cli.api_response",
            "model": "model-a",
            "input_token_count": 100,
            "output_token_count": 50,
            "event.timestamp": "2023-10-27T01:00:00Z",
        }
    }
    with open(log_file, "w") as f:
        f.write(json.dumps(entry) + "\n")

    # Test with UTC-5
    tz = ZoneInfo("Etc/GMT+5")  # Corrected to ZoneInfo

    usage, count, error = process_log_file(log_file, price_spec, timezone=tz)

    expected_date = date(2023, 10, 26)
    key = ("model-a", expected_date)

    assert key in usage
    assert usage[key].count == 1


def test_process_log_file_no_timestamp(tmp_path, price_spec):
    log_file = tmp_path / "test_no_ts.jsonl"
    entry = {
        "attributes": {
            "event.name": "gemini_cli.api_response",
            "model": "model-a",
            "input_token_count": 100,
        }
    }
    with open(log_file, "w") as f:
        f.write(json.dumps(entry) + "\n")

    usage, count, error = process_log_file(log_file, price_spec)

    assert count == 1
    key = ("model-a", date.min)
    assert key in usage


def test_process_log_file_error(tmp_path, price_spec):
    log_file = tmp_path / "broken.jsonl"
    log_file.write_text('{"valid": "json"}\n{broken json\n')

    usage, count, error = process_log_file(log_file, price_spec)

    assert error
    # Even if error, it might return partial results or empty


def test_analyze_token_usage_success(tmp_path, monkeypatch):
    # Create a dummy log file
    log_file = tmp_path / "telemetry.jsonl"
    log_file.write_text(
        '{"attributes": {"event.name": "gemini_cli.api_response", "model": "gemini-pro", "input_token_count": 10, "output_token_count": 10, "event.timestamp": "2023-10-26T10:00:00+00:00"}}\n'
    )

    # Mock get_price_spec to avoid external dependency or file reading if needed
    # But the real one reads from ~/.gemini/prices.json or defaults.
    # Assuming defaults work or file exists.
    # Let's just run it.

    ret_code = analyze_token_usage(log_file)
    assert ret_code == 0


def test_analyze_token_usage_file_not_found(tmp_path):
    log_file = tmp_path / "nonexistent.jsonl"
    ret_code = analyze_token_usage(log_file)
    assert ret_code == 1


def test_analyze_token_usage_invalid_extension(tmp_path):
    log_file = tmp_path / "test.txt"
    log_file.touch()

    with pytest.raises(ValueError, match="Log file must be a .jsonl file"):
        analyze_token_usage(log_file)


def test_analyze_token_usage_dir_no_file(tmp_path):
    # Empty directory
    with pytest.raises(FileNotFoundError, match="Could not find telemetry.log or telemetry.jsonl"):
        analyze_token_usage(tmp_path)
