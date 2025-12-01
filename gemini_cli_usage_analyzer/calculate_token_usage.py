import os
from pathlib import Path
from collections import defaultdict
from collections.abc import MutableMapping

import orjsonl
import typer
from rich.console import Console
from rich.table import Table


def main(log_file_path: Path):
    console = Console()
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

                usage_by_model[model]["input"] += int(input_tokens)
                usage_by_model[model]["output"] += int(output_tokens)
                usage_by_model[model]["cached"] += int(cached_tokens)
                usage_by_model[model]["thoughts"] += int(thoughts_tokens)
                usage_by_model[model]["count"] += 1
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
    table.add_column("Total", footer_style="bold", justify="right")

    total_req = 0
    total_input = 0
    total_output = 0
    total_cached = 0
    total_thoughts = 0

    for model, usage in sorted(usage_by_model.items()):
        total = (
            usage["input"] + usage["output"] + usage["thoughts"]
        )  # Cached tokens are part of input, so not added here directly for total
        table.add_row(
            model,
            str(usage["count"]),
            f"{usage['input']:,}",
            f"{usage['output']:,}",
            f"{usage['cached']:,}",
            f"{usage['thoughts']:,}",
            f"{total:,}",
        )
        total_req += usage["count"]
        total_input += usage["input"]
        total_output += usage["output"]
        total_cached += usage["cached"]
        total_thoughts += usage["thoughts"]

    table.columns[1].footer = str(total_req)
    table.columns[2].footer = f"{total_input:,}"
    table.columns[3].footer = f"{total_output:,}"
    table.columns[4].footer = f"{total_cached:,}"
    table.columns[5].footer = f"{total_thoughts:,}"
    table.columns[6].footer = f"{total_input + total_output + total_thoughts:,}"
    console.print(table)

    if encountered_errors:
        # Return a non-zero exit code
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
