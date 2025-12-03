from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gemini-cli-usage-analyzer")
except PackageNotFoundError:
    # This handles the case where the package is imported
    # without being installed (e.g., local script execution)
    __version__ = "unknown"
