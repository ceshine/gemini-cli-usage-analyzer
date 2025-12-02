"""A script to convert a loose OpenTelemetry log file into an optimized JSONL file."""

import os
import shutil
from datetime import datetime
from pathlib import Path

import orjson
import typer

from .simplify_logs import SimplificationLevel, simplify_record


def get_last_timestamp(file_path: Path) -> str | None:
    """Read the last timestamp from an existing JSONL file.

    This function efficiently reads the JSONL file from the end to find the timestamp
    of the last complete JSON object. It is optimized for large files by reading in
    chunks and reconstructing lines as it goes backward.
    """
    if not file_path.exists():
        return None

    try:
        with open(file_path, "rb") as f:
            _ = f.seek(0, os.SEEK_END)  # Move to the end of the file
            pos = f.tell()  # Get the current position (end of file)
            remainder = b""  # Buffer to store incomplete lines from previous chunks

            chunk_size = 64 * 1024  # Size of each chunk to read (64 KB)

            while pos > 0:
                read_len = min(chunk_size, pos)  # Determine how much to read in this chunk
                pos -= read_len  # Move the position backward
                _ = f.seek(pos)  # Seek to the new position
                chunk = f.read(read_len)  # Read the chunk

                # Check if a newline character is present in the chunk
                if b"\n" in chunk:
                    parts = chunk.split(b"\n")

                    # The last part of the chunk combined with the remainder
                    # from the previous chunk forms a potentially complete line.
                    last_line = parts[-1] + remainder
                    if last_line.strip():  # If the line is not empty
                        # Attempt to parse the JSON and return the timestamp
                        return orjson.loads(last_line)["attributes"]["event.timestamp"]

                    # If the last line was empty or invalid, check other full lines
                    # within this chunk, going backward.
                    for line in reversed(parts[1:-1]):
                        if line.strip():
                            return orjson.loads(line)["attributes"]["event.timestamp"]

                    # The first part of the chunk might be the end of a line
                    # that started in an even earlier chunk. Store it as remainder.
                    remainder = parts[0]
                else:
                    # If no newline in the chunk, it means the current chunk is part
                    # of a larger line. Prepend it to the remainder.
                    remainder = chunk + remainder

            # If the loop finishes, it means we reached the start of the file.
            # The 'remainder' will contain the first line of the file, if any.
            if remainder.strip():
                return orjson.loads(remainder)["attributes"]["event.timestamp"]

    except Exception as e:
        raise ValueError(f"Failed to read last timestamp from {file_path}: {e}")

    return None


def main(
    input_file_path: Path = typer.Argument(
        Path(".gemini/telemetry.log"),
        help="Path to the input log file containing concatenated JSON objects.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_file_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the output JSONL file. Defaults to the same location of the input file path, but with the suffix changed to .jsonl",
        writable=True,
    ),
    simplify: bool = typer.Option(
        False,
        "--simplify",
        help="(Deprecated) Only keep objects whose 'attributes' field's 'event.name' value is 'gemini_cli.api_response' or 'gemini_cli.api_request'. Use --simplify-level instead.",
    ),
    simplify_level: int = typer.Option(
        0,
        "--simplify-level",
        help="Simplification level. 0: None, 1: Events Only. Defaults to 0.",
    ),
    archiving_disabled: bool = typer.Option(
        False, "--disable-archiving", help="Disable archiving of the input log file."
    ),
    archive_folder_path: Path = typer.Option(
        "/tmp",
        "--archive-folder",
        help="Folder to move the input log file to after processing.",
    ),
):
    """Convert a log file with concatenated JSON objects to JSONL format.

    This function processes the input log file, converting it into a JSON Lines (JSONL) format.
    It supports incremental updates by checking the last timestamp in the existing output file
    and appending new records.

    Key features:

    - **Incremental Updates**: Skips records with timestamps earlier than or equal to the
      last recorded timestamp in the output file.
    - **Simplification**: Optionally filters records based on the specified simplification level.
    - **Robust Parsing**: Accumulatively reads lines and parses JSON objects, handling potential
      formatting issues in raw logs.
    - **Archiving**: Optionally moves the processed input file to an archive directory with
      a timestamped filename.
    """
    count = 0
    skipped_count = 0

    if output_file_path is None:
        output_file_path = input_file_path.with_suffix(".jsonl")

    # Determine the last timestamp to support incremental updates
    last_timestamp = get_last_timestamp(output_file_path)
    mode = "ab" if last_timestamp else "wb"

    if last_timestamp:
        typer.echo(f"Found existing output. Appending new entries after {last_timestamp}...")
    else:
        typer.echo("Starting fresh conversion...")

    # Determine effective simplification level
    effective_level = simplify_level
    if simplify and effective_level == 0:
        effective_level = SimplificationLevel.EVENTS_ONLY.value

    try:
        # Open input in text mode (utf-8) and output in binary mode (for orjson)
        with open(input_file_path, "r", encoding="utf-8") as f_in, open(output_file_path, mode) as f_out:
            buffer = ""

            for line in f_in:
                buffer += line

                # heuristic: only attempt parse if line looks like end of object
                if line.strip() == "}":
                    try:
                        # Attempt to parse the accumulated buffer
                        obj = orjson.loads(buffer)
                        # Clear buffer
                        buffer = ""
                        attributes = obj.get("attributes", {})

                        # Pre-write validation: Ensure event.timestamp exists
                        current_ts = attributes.get("event.timestamp")
                        if not current_ts:
                            continue

                        # Incremental check
                        if last_timestamp and current_ts <= last_timestamp:
                            skipped_count += 1
                            continue

                        if (obj := simplify_record(obj, effective_level)) is None:
                            continue

                        # If successful, write to output and reset buffer
                        _ = f_out.write(orjson.dumps(obj) + b"\n")
                        count += 1
                    except orjson.JSONDecodeError:
                        # Buffer might contain a nested object ending in '}', or incomplete data
                        # Continue accumulating
                        pass

            if buffer.strip():
                typer.echo("Warning: End of file reached with incomplete JSON data in buffer.", err=True)

        typer.echo(f"Successfully converted {count} records to {output_file_path} (Skipped {skipped_count})")

        if not archiving_disabled:
            archive_folder_path.mkdir(exist_ok=True, parents=True)
            new_file_name = f"{input_file_path.stem}.{int(datetime.now().timestamp())}{input_file_path.suffix}"
            new_file_path = archive_folder_path / new_file_name
            shutil.move(input_file_path, new_file_path)
            typer.echo(f"Archived {input_file_path} to {new_file_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
