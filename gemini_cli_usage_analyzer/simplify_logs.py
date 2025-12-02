"""Module for simplifying log records.

## Simplification Levels:

-   **Level 0**: No simplification. All log records and all their fields are retained.
-   **Level 1**: Filters records to include only `gemini_cli.api_response` and `gemini_cli.api_request` events. All fields within these selected records are retained.
-   **Level 2**: Applies Level 1 filtering and further simplifies the structure of the retained records. Only the "attributes" and "_body" fields of the selected records are kept.
-   **Level 3**: The most aggressive simplification. Only `gemini_cli.api_response` events are retained. For these events, only specific token-related attributes (`duration_ms`, `input_token_count`, `output_token_count`, `cached_content_token_count`, `thoughts_token_count`, `total_token_count`, `tool_token_count`, `model`, and `session.id`) are preserved within the "attributes" field. The "_body" field is also retained.
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

import typer
import orjson
import orjsonl


def simplify_record(record: dict[str, Any], level: int) -> dict[str, Any] | None:
    """Processes a log record based on the simplification level.

    Args:
        record (Dict[str, Any]): The log record to process.
        level (int): The simplification level to apply.

    Returns:
        Optional[Dict[str, Any]]: The processed record if it should be kept (potentially modified),
            or None if it should be discarded.
    """
    if level <= 0:
        return record

    if "attributes" not in record or "event.name" not in record["attributes"]:
        return None

    attributes = record["attributes"]
    event_name = attributes["event.name"]

    # Level 3: Keep only gemini_cli.api_response and keep only the token count and duration attributes
    if level >= 3:
        if event_name != "gemini_cli.api_response":
            return None
        record["attributes"] = {
            "event.timestamp": attributes["event.timestamp"],
            "duration_ms": attributes["duration_ms"],
            "input_token_count": attributes["input_token_count"],
            "output_token_count": attributes["output_token_count"],
            "cached_content_token_count": attributes["cached_content_token_count"],
            "thoughts_token_count": attributes["thoughts_token_count"],
            "total_token_count": attributes["total_token_count"],
            "tool_token_count": attributes["tool_token_count"],
            "model": attributes["model"],
            "session.id": attributes["session.id"],
            # The event name is always "gemini_cli.api_response" for now
            "event.name": attributes["event.name"],
        }

    # Level 1: Filter for specific events
    if level >= 1 and level < 3:
        if event_name not in ("gemini_cli.api_response", "gemini_cli.api_request"):
            return None

    # Level 2: Only keep essential fields
    if level >= 2:
        record = {"attributes": record["attributes"], "_body": record["_body"]}

    return record


def simplify_existing_jsonl(
    input_file_path: Path,
    level: int,
    archive_folder: Path = Path("/tmp"),
    disable_archiving: bool = False,
):
    """Simplifies an existing JSONL file by filtering and transforming its records.

    Args:
        input_file_path (Path): The path to the input JSONL file.
        level (int): The simplification level to apply to the records.
        archive_folder (Path): The folder where the original file will be archived
            before simplification, if archiving is not disabled. Defaults to /tmp.
        disable_archiving (bool): If True, the original file will be removed instead
            of archived. Defaults to False.

    Raises:
        typer.BadParameter: If the input file is not a .jsonl file, does not exist,
            or if a temporary/archive file already exists from a previous failed run.
    """
    if input_file_path.suffix != ".jsonl":
        raise typer.BadParameter("Input file must be a .jsonl file")
    if not input_file_path.exists():
        raise typer.BadParameter("Input file does not exist")

    temp_file = input_file_path.with_suffix(".jsonl.tmp")
    if temp_file.exists():
        raise typer.BadParameter(
            f"Temp file {temp_file} already exists (possibly from a previous failed run). Please clean up manually and try again."
        )

    archive_file_path: Path | None = None
    if disable_archiving is False:
        # Move the original file to the archive folder
        archive_target_name = f"{input_file_path.stem}.{int(datetime.now().timestamp())}{input_file_path.suffix}"
        archive_file_path = archive_folder / archive_target_name
        if archive_file_path.exists():
            raise typer.BadParameter(
                f"Archive file {archive_file_path} already exists. Please clean up manually and try again."
            )

    with open(temp_file, "wb") as fout:
        for obj in orjsonl.stream(input_file_path):
            if not isinstance(obj, dict):
                typer.echo(f"Warning: found malformed record in {input_file_path}. Skipping...")
                continue
            if (simplified_obj := simplify_record(obj, level=level)) is not None:
                _ = fout.write(orjson.dumps(simplified_obj) + b"\n")

    if disable_archiving is False:
        assert archive_file_path is not None, "Archive file path is not set"
        _ = shutil.move(input_file_path, archive_file_path)
        typer.echo(f"Archived {input_file_path} to {archive_file_path}")
    else:
        # Remove the original file
        shutil.rmtree(input_file_path)
        typer.echo(f"Removed {input_file_path}")

    # Move the temp file to the original file path
    _ = temp_file.rename(input_file_path)

    typer.echo(f"{input_file_path} has been simplified at level {level}")


if __name__ == "__main__":
    typer.run(simplify_existing_jsonl)
