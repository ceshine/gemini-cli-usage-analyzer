import os
from pathlib import Path
from collections import defaultdict
from collections.abc import MutableMapping

import orjson
import typer


def parse_json_stream(file_path: Path):
    """Parses a file where each line is a JSON object (JSONL format)."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped_line = line.strip()
            if not stripped_line:
                continue  # Skip empty lines

            try:
                obj = orjson.loads(stripped_line)
                yield obj
            except orjson.JSONDecodeError as e:
                # Log the error but continue processing other lines
                typer.echo(
                    f"Warning: Could not parse JSON on line {line_num}: {stripped_line[:100]}... Error: {e}",
                    err=True,
                )
                continue


def main(log_file_path: Path):
    usage_by_model: MutableMapping[str, dict[str, int]] = defaultdict(
        lambda: {"input": 0, "output": 0, "cached": 0, "thoughts": 0, "count": 0}
    )

    if not os.path.exists(log_file_path):
        typer.echo(f"Error: {log_file_path} not found.", err=True)
        return

    typer.echo(f"Reading {log_file_path}...")

    count = 0
    encountered_errors = False
    try:
        for entry in parse_json_stream(log_file_path):
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
        typer.echo(f"\nWarning: Error occurred while processing logs: {e}", err=True)
        typer.echo("Displaying results processed so far...", err=True)
        encountered_errors = True

    if count == 0:
        typer.echo("No inference events found in the log.")
        return

    typer.echo(f"\nFound {count} inference events.\n")
    typer.echo(
        f"{'Model':<30} | {'Requests':<10} | {'Input Tokens':<15} | {'Output Tokens':<15} | {'Cached Tokens':<15} | {'Thoughts Tokens':<15} | {'Total':<15}"
    )
    typer.echo("-" * 135)

    total_req = 0
    total_input = 0
    total_output = 0
    total_cached = 0
    total_thoughts = 0

    for model, usage in sorted(usage_by_model.items()):
        total = (
            usage["input"] + usage["output"] + usage["thoughts"]
        )  # Cached tokens are part of input, so not added here directly for total
        typer.echo(
            f"{model:<30} | {usage['count']:<10} | {usage['input']:<15,} | {usage['output']:<15,} | {usage['cached']:<15,} | {usage['thoughts']:<15,} | {total:<15,}"
        )
        total_req += usage["count"]
        total_input += usage["input"]
        total_output += usage["output"]
        total_cached += usage["cached"]
        total_thoughts += usage["thoughts"]

    typer.echo("-" * 135)
    typer.echo(
        f"{'Grand Total':<30} | {total_req:<10} | {total_input:<15,} | {total_output:<15,} | {total_cached:<15,} | {total_thoughts:<15,} | {total_input + total_output + total_thoughts:<15,}"
    )

    if encountered_errors:
        # Return a non-zero exit code
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
