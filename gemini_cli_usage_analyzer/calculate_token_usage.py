import os
from pathlib import Path
from collections import defaultdict
from collections.abc import MutableMapping

import orjsonl
import typer
from rich.console import Console
from rich.table import Table

from .price_spec import get_price_spec


def main(log_file_path: Path):
    console = Console()
    price_spec = get_price_spec(cache_path=Path(".gemini/prices_cache.json"))
    usage_by_model: MutableMapping[str, dict[str, int]] = defaultdict(
        lambda: {"input": 0, "output": 0, "cached": 0, "thoughts": 0, "count": 0}
    )

    if not os.path.exists(log_file_path):
        console.print(f"Error: {log_file_path} not found.", style="bold red")
        return

    console.print(f"Reading {log_file_path}...")

    count = 0
    encountered_errors = False
    try:
        for entry in orjsonl.stream(log_file_path):
            assert isinstance(entry, dict), f"Got an unexpected entry that is not a dict {entry}"
            attributes = entry.get("attributes", {})
            event_name = attributes.get("event.name")

            if event_name == "gemini_cli.api_response":
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
                    input_cost_per_token = model_price_spec.get(
                        "input_cost_per_token_above_200k_tokens", input_cost_per_token
                    )
                    output_cost_per_token = model_price_spec.get(
                        "output_cost_per_token_above_200k_tokens", output_cost_per_token
                    )
                    cached_cost_per_token = model_price_spec.get(
                        "cache_read_input_token_cost_above_200k_tokens", cached_cost_per_token
                    )

                # Calculate cost for this entry
                cost = (
                    ((input_tokens - cached_tokens) * input_cost_per_token)
                    + ((output_tokens + thoughts_tokens) * output_cost_per_token)
                    + (cached_tokens * cached_cost_per_token)
                )

                usage_by_model[model]["input"] += int(input_tokens)
                usage_by_model[model]["output"] += int(output_tokens)
                usage_by_model[model]["cached"] += int(cached_tokens)
                usage_by_model[model]["thoughts"] += int(thoughts_tokens)
                usage_by_model[model]["count"] += 1
                usage_by_model[model]["cost"] = usage_by_model[model].get("cost", 0.0) + cost
                count += 1
    except Exception as e:
        console.print(f"\nWarning: Error occurred while processing logs: {e}", style="bold yellow")
        console.print("Displaying results processed so far...", style="bold yellow")
        encountered_errors = True

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
    typer.run(main)
