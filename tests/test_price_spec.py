import time
import json
import os
from unittest.mock import MagicMock, patch
import pytest
from gemini_cli_usage_analyzer.price_spec import get_price_spec, URL

@pytest.fixture
def mock_urlopen():
    with patch("urllib.request.urlopen") as mock:
        yield mock

@pytest.fixture
def sample_price_data():
    return {"model-a": {"input_cost_per_token": 0.1}}

def test_get_price_spec_no_cache(mock_urlopen, sample_price_data):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(sample_price_data).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    result = get_price_spec(cache_path=None)
    
    assert result == sample_price_data
    mock_urlopen.assert_called_once_with(URL)

def test_get_price_spec_fetch_error(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 404
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    with pytest.raises(RuntimeError, match="Failed to fetch price spec"):
        get_price_spec(cache_path=None)

def test_get_price_spec_cache_fresh(tmp_path, mock_urlopen, sample_price_data):
    cache_file = tmp_path / "prices.json"
    cache_file.write_bytes(json.dumps(sample_price_data).encode("utf-8"))
    
    # Ensure the file is fresh (mtime is now)
    # get_price_spec checks if time.time() - mtime < update_interval
    
    result = get_price_spec(cache_path=cache_file)
    
    assert result == sample_price_data
    mock_urlopen.assert_not_called()

def test_get_price_spec_cache_stale_fetch_success(tmp_path, mock_urlopen, sample_price_data):
    cache_file = tmp_path / "prices.json"
    old_data = {"old": "data"}
    cache_file.write_bytes(json.dumps(old_data).encode("utf-8"))
    
    # Set mtime to be old
    old_time = time.time() - 100000
    os.utime(cache_file, (old_time, old_time))
    
    # Mock successful fetch
    new_data = {"new": "data"}
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(new_data).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    result = get_price_spec(cache_path=cache_file)
    
    assert result == new_data
    mock_urlopen.assert_called_once()
    
    # Verify cache was updated
    assert json.loads(cache_file.read_bytes()) == new_data

def test_get_price_spec_cache_stale_fetch_fail_fallback(tmp_path, mock_urlopen, sample_price_data):
    cache_file = tmp_path / "prices.json"
    cache_file.write_bytes(json.dumps(sample_price_data).encode("utf-8"))
    
    # Set mtime to be old
    old_time = time.time() - 100000
    os.utime(cache_file, (old_time, old_time))
    
    # Mock failed fetch
    mock_urlopen.side_effect = RuntimeError("Network error")

    result = get_price_spec(cache_path=cache_file)
    
    # Should return stale cache
    assert result == sample_price_data
    mock_urlopen.assert_called_once()

def test_get_price_spec_cache_create_dirs(tmp_path, mock_urlopen, sample_price_data):
    # Cache path in a nested directory that doesn't exist
    cache_file = tmp_path / "subdir" / "prices.json"
    
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(sample_price_data).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    result = get_price_spec(cache_path=cache_file)
    
    assert result == sample_price_data
    assert cache_file.exists()
    assert json.loads(cache_file.read_bytes()) == sample_price_data