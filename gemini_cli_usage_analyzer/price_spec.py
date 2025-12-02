import time
import urllib.request
from pathlib import Path
from typing import Any

import orjson

URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"


def _fetch_from_url() -> dict[str, Any]:
    """Fetches the latest price specification from the remote URL.

    Returns:
        A dictionary containing the model pricing information.

    Raises:
        RuntimeError: If the HTTP request fails or returns a non-200 status code.
    """
    try:
        with urllib.request.urlopen(URL) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to fetch price spec: HTTP {response.status}")
            data = response.read()
            return orjson.loads(data)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch price spec from {URL}") from e


def get_price_spec(cache_path: str | Path | None = None, update_interval_seconds: int = 86400) -> dict[str, Any]:
    """Fetches the LLM price spec mapping from the remote URL, with local caching.

    Args:
        cache_path (str | Path | None): The path to the local cache file. If None, the data is fetched
            directly from the URL without caching.
        update_interval_seconds (int): The minimum interval in seconds to update the cache.
            Defaults to 86400 (24 hours).

    Returns:
        A dictionary containing the price specification for various models.

    Raises:
        RuntimeError: If fetching from the URL fails and no valid cache is available.
    """
    if cache_path is None:
        return _fetch_from_url()

    cache_file = Path(cache_path)

    # Check if cache exists and is fresh enough
    if cache_file.exists():
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime < update_interval_seconds:
            try:
                with open(cache_file, "rb") as f:
                    return orjson.loads(f.read())
            except Exception:
                # If read fails, fall through to re-fetch
                pass

    # Fetch from URL
    try:
        json_data = _fetch_from_url()
    except Exception as e:
        # If fetch fails but we have a stale cache, try to return that
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return orjson.loads(f.read())
            except Exception:
                pass  # Reraise original error if cache read also fails
        raise RuntimeError(f"Failed to fetch price spec from {URL}") from e

    # Save to cache
    try:
        # Ensure directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            f.write(orjson.dumps(json_data))
    except Exception as e:
        # Warning: Failed to write cache, but we can still return the data
        print(f"Warning: Failed to write to cache at {cache_file}: {e}")

    return json_data
