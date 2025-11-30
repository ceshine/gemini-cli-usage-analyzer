import os
import re
from pathlib import Path
from collections import defaultdict

import orjson
import typer


def parse_json_stream(file_path: Path):
    """Parses a file containing concatenated JSON objects."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback for potential encoding issues, though unlikely for JSON
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()

    # Split using regex to find boundaries between objects: } followed by whitespace and {
    # We use lookbehind (?<=}) and lookahead (?=&#123;) to keep the braces in the chunks.
    # We also capture the whitespace (\s*) to preserve it if we need to merge chunks
    # (e.g. if the split occurred inside a string).
    parts = re.split(r'(?<=})(\s*)(?={)', content)

    buffer = ""
    for part in parts:
        buffer += part

        # Skip if buffer contains only whitespace
        if not buffer.strip():
            # If we cleared the buffer after a success, and part is just whitespace, keep it cleared.
            # If we are accumulating (buffer has content), adding whitespace doesn't make it parseable usually,
            # unless it was just trailing space.
            # But here, if buffer is just whitespace, it's definitely not a valid object start.
            # However, if we are accumulating a partial object, buffer won't be just whitespace.
            if not buffer.strip():
                buffer = "" 
            continue

        try:
            # Attempt to parse the current buffer
            obj = orjson.loads(buffer)
            yield obj
            # If successful, clear buffer for the next object
            buffer = ""
        except orjson.JSONDecodeError:
            # If parsing fails (e.g., split inside a string), continue accumulating
            continue


def main(log_file_path: Path):
    usage_by_model = defaultdict(lambda: {"input": 0, "output": 0, "cached": 0, "thoughts": 0, "count": 0})

    if not os.path.exists(log_file_path):
        print(f"Error: {log_file_path} not found.")
        return

    print(f"Reading {log_file_path}...")

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
        print(f"\nWarning: Error occurred while processing logs: {e}")
        print("Displaying results processed so far...")
        encountered_errors = True

    if count == 0:
        print("No inference events found in the log.")
        return

    print(f"\nFound {count} inference events.\n")
    print(
        f"{'Model':<30} | {'Requests':<10} | {'Input Tokens':<15} | {'Output Tokens':<15} | {'Cached Tokens':<15} | {'Thoughts Tokens':<15} | {'Total':<15}"
    )
    print("-" * 135)

    total_req = 0
    total_input = 0
    total_output = 0
    total_cached = 0
    total_thoughts = 0

    for model, usage in sorted(usage_by_model.items()):
        total = (
            usage["input"] + usage["output"] + usage["thoughts"]
        )  # Cached tokens are part of input, so not added here directly for total
        print(
            f"{model:<30} | {usage['count']:<10} | {usage['input']:<15,} | {usage['output']:<15,} | {usage['cached']:<15,} | {usage['thoughts']:<15,} | {total:<15,}"
        )
        total_req += usage["count"]
        total_input += usage["input"]
        total_output += usage["output"]
        total_cached += usage["cached"]
        total_thoughts += usage["thoughts"]

    print("-" * 135)
    print(
        f"{'Grand Total':<30} | {total_req:<10} | {total_input:<15,} | {total_output:<15,} | {total_cached:<15,} | {total_thoughts:<15,} | {total_input + total_output + total_thoughts:<15,}"
    )

    if encountered_errors:
        # Return a non-zero exit code
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)