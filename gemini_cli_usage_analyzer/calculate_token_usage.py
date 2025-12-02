import os
import logging
from pathlib import Path
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any

import orjsonl
import typer
from rich.console import Console
from rich.table import Table

from .price_spec import get_price_spec
from .convert_logs import main as convert_log_file

LOGGER = logging.getLogger(__name__)


def calculate_cost(attributes: dict[str, Any], price_spec: dict[str, Any]) -> float:
    """Calculates the cost of an inference event based on token usage and price specification.

    Args:
        attributes (dict[str, Any]): The attributes dictionary from the log entry containing token counts and model info.
        price_spec (dict[str, Any]): The entire price specification dictionary mapping model names to their pricing details.

    Returns:
        The calculated cost (float) in USD.
    """
    model = attributes.get("model", "unknown")
    input_tokens = attributes.get("input_token_count") or 0
    output_tokens = attributes.get("output_token_count") or 0
    cached_tokens = attributes.get("cached_content_token_count") or 0
    thoughts_tokens = attributes.get("thoughts_token_count") or 0

    # Get price information for the model
    model_price_spec = price_spec.get(model, {})

    input_cost_per_token = model_price_spec.get("input_cost_per_token", 0)
    output_cost_per_token = model_price_spec.get("output_cost_per_token", 0)
    # Use cache_read_input_token_cost for cached tokens if available
    cached_cost_per_token = model_price_spec.get("cache_read_input_token_cost", 0)

    # Check for tiered pricing based on context length
    # Assuming input_tokens represents the total context size (including cached)
    if input_tokens > 200000:
        input_cost_per_token = model_price_spec.get("input_cost_per_token_above_200k_tokens", input_cost_per_token)
        output_cost_per_token = model_price_spec.get("output_cost_per_token_above_200k_tokens", output_cost_per_token)
        cached_cost_per_token = model_price_spec.get(
            "cache_read_input_token_cost_above_200k_tokens", cached_cost_per_token
        )

    # Calculate cost for this entry
    cost = (
        ((input_tokens - cached_tokens) * input_cost_per_token)
        + ((output_tokens + thoughts_tokens) * output_cost_per_token)
        + (cached_tokens * cached_cost_per_token)
    )
    return cost


def process_log_file(
    log_file_path: Path, price_spec: dict[str, Any]
) -> tuple[MutableMapping[str, dict[str, int | float]], int, bool]:
    """Processes a log file to aggregate token usage and cost by model.

    Args:
        log_file_path (Path): The path to the log file.
        price_spec (dict[str, Any]): The price specification dictionary.

    Returns:
        A tuple containing:
            - A dictionary (MutableMapping[str, dict[str, int | float]]) mapping model names to usage statistics (input, output, cached, thoughts, count, cost).
            - The total count (int) of processed inference events.
            - A boolean (bool) indicating if any errors were encountered during processing.
    """
    usage_by_model: MutableMapping[str, dict[str, int | float]] = defaultdict(
        lambda: {"input": 0, "output": 0, "cached": 0, "thoughts": 0, "count": 0, "cost": 0.0}
    )
    count = 0
    encountered_errors = False

    try:
        for entry in orjsonl.stream(log_file_path):
            assert isinstance(entry, dict), f"Got an unexpected entry that is not a dict {entry}"
            attributes = entry.get("attributes", {})
            event_name = attributes.get("event.name")

            if event_name == "gemini_cli.api_response":
                model = attributes.get("model", "unknown")
                input_tokens = int(attributes.get("input_token_count") or 0)
                output_tokens = int(attributes.get("output_token_count") or 0)
                cached_tokens = int(attributes.get("cached_content_token_count") or 0)
                thoughts_tokens = int(attributes.get("thoughts_token_count") or 0)

                cost = calculate_cost(attributes, price_spec)

                usage_by_model[model]["input"] += input_tokens
                usage_by_model[model]["output"] += output_tokens
                usage_by_model[model]["cached"] += cached_tokens
                usage_by_model[model]["thoughts"] += thoughts_tokens
                usage_by_model[model]["count"] += 1
                usage_by_model[model]["cost"] += cost
                count += 1
    except Exception as e:
        LOGGER.error("Error processing the JSONL data: %s", e)
        encountered_errors = True

    return usage_by_model, count, encountered_errors


def main(log_file_path: Path, enable_archiving: bool = False, log_simplify_level: int = 1):
    """Calculates and displays token usage and cost from a Gemini CLI log file.

    Args:
        log_file_path (Path): Path to the JSONL log file to analyze.
          - If a folder is provided, it attempts to find `telemetry.log` (or `.gemini/telemetry.log`) to convert.
          - It also checks for `telemetry.jsonl` (or `gemini/telemetry.jsonl`) to use directly if no log file is found.
          - If a .jsonl file is provided, it will be processed directly.
        enable_archiving (bool): Enable archiving (moving to `/tmp`) when a folder is provided as `log_file_path`. Only enable it if no Gemini CLI is currently running.
        log_simplify_level (int): Level of simplification for the JSONL log file. Available levels: 0 (no simplification), 1 (default), 2 (trim fields), and 3 (trim attributes)
    """
    console = Console()

    if log_file_path.is_dir():
        source_log_file, jsonl_file = None, None
        # Try to find the the default file in the specified folder
        if (log_file_path / "telemetry.log").exists():
            source_log_file = log_file_path / "telemetry.log"
        elif (log_file_path / ".gemini" / "telemetry.log").exists():
            source_log_file = log_file_path / ".gemini" / "telemetry.log"
        elif (log_file_path / "telemetry.jsonl").exists():
            jsonl_file = log_file_path / "telemetry.jsonl"
        elif (log_file_path / ".gemini" / "telemetry.jsonl").exists():
            jsonl_file = log_file_path / ".gemini" / "telemetry.jsonl"
        else:
            raise typer.BadParameter(
                f"Could not find telemetry.log or telemetry.jsonl in {log_file_path} nor in its '.gemini' or 'gemini' subdirectories."
            )
        if source_log_file is not None and jsonl_file is None:
            log_file_path = convert_log_file(
                source_log_file,
                source_log_file.parent / "telemetry.jsonl",
                simplify_level=log_simplify_level,
                archiving_enabled=enable_archiving,
                archive_folder_path=Path("/tmp"),
            )
            console.print(
                f"Converted {source_log_file} to {log_file_path} with archiving [bold]{'ENABLED' if enable_archiving else 'DISABLED'}[/bold]",
                style="green",
            )
        elif jsonl_file is not None:
            console.print(f"Using [bold]{jsonl_file}[/bold] as the JSONL log file", style="green")
            log_file_path = jsonl_file
        else:
            raise RuntimeError("Incorrect combination of the `source_log_file` and the `jsonl_file` values.")
    elif log_file_path.suffix != ".jsonl":
        raise typer.BadParameter(f"Log file must be a .jsonl file, got {log_file_path}")

    if not log_file_path.exists():
        console.print(f"Error: {log_file_path} not found.", style="bold red")
        return

    price_spec = get_price_spec()

    console.print(f"Reading {log_file_path}...")
    usage_by_model, count, encountered_errors = process_log_file(log_file_path, price_spec)

    if encountered_errors:
        console.print("Displaying results processed so far...", style="bold yellow")

    if count == 0:
        console.print("No inference events found in the log.")
        return

    console.print(f"\nFound {count} inference events.\n")
    table = Table(
        title="Token Usage by Model",
        show_footer=True,
        footer_style="bold",
        title_justify="left",
    )

    table.add_column("Model", footer="Grand Total", justify="left")
    table.add_column("Requests", footer_style="bold", justify="right")
    table.add_column("Input Tokens", footer_style="bold", justify="right")
    table.add_column("Output Tokens", footer_style="bold", justify="right")
    table.add_column("Cached Tokens", footer_style="bold", justify="right")
    table.add_column("Thoughts Tokens", footer_style="bold", justify="right")
    table.add_column("Cost ($)", footer_style="bold", justify="right")
    table.add_column("Total", footer_style="bold", justify="right")

    total_req = 0
    total_input = 0
    total_output = 0
    total_cached = 0
    total_thoughts = 0
    total_cost = 0.0

    for model, usage in sorted(usage_by_model.items()):
        # Cached tokens are part of input, so not added here directly for total
        total_tokens = usage["input"] + usage["output"] + usage["thoughts"]
        model_cost = usage.get("cost", 0.0)
        table.add_row(
            model,
            str(usage["count"]),
            f"{usage['input']:,}",
            f"{usage['output']:,}",
            f"{usage['cached']:,}",
            f"{usage['thoughts']:,}",
            f"{model_cost:,.6f}",
            f"{total_tokens:,}",
        )
        total_req += usage["count"]
        total_input += usage["input"]
        total_output += usage["output"]
        total_cached += usage["cached"]
        total_thoughts += usage["thoughts"]
        total_cost += model_cost

    table.columns[1].footer = str(total_req)
    table.columns[2].footer = f"{total_input:,}"
    table.columns[3].footer = f"{total_output:,}"
    table.columns[4].footer = f"{total_cached:,}"
    table.columns[5].footer = f"{total_thoughts:,}"
    table.columns[6].footer = f"{total_cost:,.6f}"
    table.columns[7].footer = f"{total_input + total_output + total_thoughts:,}"
    console.print(table)

    if encountered_errors:
        # Return a non-zero exit code
        raise typer.Exit(1)


if __name__ == "__main__":
    os.environ["PRICE_CACHE_PATH"] = "~/.gemini/prices.json"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    typer.run(main)
