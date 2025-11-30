"""A script to convert a loose OpenTelemetry log file into an optimized JSONL file."""

from pathlib import Path

import typer
import orjson


def main(
    input_file: Path = typer.Argument(
        Path(".gemini/telemetry.log"),
        help="Path to the input log file containing concatenated JSON objects.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Option(
        Path(".gemini/telemetry.jsonl"),
        "--output",
        "-o",
        help="Path to the output JSONL file.",
        writable=True,
    ),
    simplify: bool = typer.Option(
        False,
        "--simplify",
        help="Only keep objects whose 'attributes' field's 'event.name' value is 'gemini_cli.api_response' or 'gemini_cli.api_request'.",
    ),
):
    """Convert a log file with concatenated JSON objects to JSONL format.

    Reads the input file line by line. Accumulates lines in a single string buffer
    and attempts to decode using orjson only when a line consisting solely of "}"
    is encountered.

    Args:
        input_file: The path to the source log file.
        output_file: The path where the JSONL output will be written.
    """
    count = 0
    try:
        # Open input in text mode (utf-8) and output in binary mode (for orjson)
        with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "wb") as f_out:
            buffer = ""

            for line in f_in:
                buffer += line

                # heuristic: only attempt parse if line looks like end of object
                if line.strip() == "}":
                    try:
                        # Attempt to parse the accumulated buffer
                        obj = orjson.loads(buffer)

                        should_write = True
                        if simplify:
                            attributes = obj.get("attributes", {})
                            event_name = attributes.get("event.name")
                            if event_name not in ("gemini_cli.api_response", "gemini_cli.api_request"):
                                should_write = False

                        # If successful, write to output and reset buffer
                        if should_write:
                            _ = f_out.write(orjson.dumps(obj) + b"\n")
                            count += 1
                        buffer = ""
                    except orjson.JSONDecodeError:
                        # Buffer might contain a nested object ending in '}', or incomplete data
                        # Continue accumulating
                        pass

            if buffer.strip():
                typer.echo("Warning: End of file reached with incomplete JSON data in buffer.", err=True)

        typer.echo(f"Successfully converted {count} records to {output_file}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
