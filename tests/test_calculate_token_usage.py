import json
import pytest
from gemini_cli_usage_analyzer.calculate_token_usage import calculate_cost, process_log_file


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
            }
        },
    ]
    with open(log_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    usage, count, error = process_log_file(log_file, price_spec)

    assert count == 2
    assert not error
    assert "model-a" in usage
    assert usage["model-a"]["count"] == 2
    assert usage["model-a"]["input"] == 3000
    assert usage["model-a"]["output"] == 600

    # Check total cost
    # Entry 1: 1000*1e-6 + 500*2e-6 = 0.002
    # Entry 2: 2000*1e-6 + 100*2e-6 = 0.002 + 0.0002 = 0.0022
    # Total: 0.0042
    assert usage["model-a"]["cost"] == pytest.approx(0.0042)


def test_process_log_file_error(tmp_path, price_spec):
    log_file = tmp_path / "broken.jsonl"
    log_file.write_text('{"valid": "json"}\n{broken json\n')

    # process_log_file expects orjsonl stream, which might fail on broken json
    # Let's see how orjsonl handles it. It likely raises an error.

    usage, count, error = process_log_file(log_file, price_spec)

    assert error
