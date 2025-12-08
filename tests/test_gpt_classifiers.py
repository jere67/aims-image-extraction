import pytest
from unittest.mock import MagicMock, patch
import json
import base64
import os

from gpt_classifiers import base_classifier

MOCK_IMAGE_PATH = "/tmp/fake_image.jpg"

@pytest.fixture
def mock_clients(mocker):
    mock = mocker.patch('gpt_classifiers.base_classifier.client')
    return mock

def test_encode_image_to_base64(tmp_path):
    """Test standard base64 encoding."""
    p = tmp_path / "test.jpg"
    p.write_bytes(b"fakedata")
    
    encoded = base_classifier._encode_image_to_base64(str(p))
    expected = base64.b64encode(b"fakedata").decode('utf-8')
    assert encoded == expected

    # Test missing file behavior
    assert base_classifier._encode_image_to_base64("/nonexistent/file.jpg") is None

def test_classify_images_with_gpt_success(mock_clients, tmp_path):
    """Test successful API call and JSON parsing."""
    p = tmp_path / "test_api.jpg"
    p.write_bytes(b"data")
    
    # Mock Response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "test_api.jpg": {
            "is_data_plot": True,
            "decision": "pass"
        }
    })
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    
    mock_clients.chat.completions.create.return_value = mock_response

    result = base_classifier.classify_images_with_gpt([str(p)], "gpt-test", "system prompt")

    assert "test_api.jpg" in result
    assert result["test_api.jpg"]["is_data_plot"] is True
    assert result["usage"]["total_tokens"] == 30

def test_classify_images_with_gpt_json_error(mock_clients, tmp_path):
    """Test handling of malformed JSON."""
    p = tmp_path / "bad_json.jpg"
    p.write_bytes(b"data")
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Not a JSON string"
    mock_clients.chat.completions.create.return_value = mock_response
    
    result = base_classifier.classify_images_with_gpt([str(p)], "gpt-test", "prompt")
    
    assert "error" in result
    assert "Failed to parse JSON" in result["error"]
