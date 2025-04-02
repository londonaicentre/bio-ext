from unittest.mock import Mock, patch

import pytest
import requests

from ..oncollama_epic.main import (
    extract_epic_section,
    extract_json_from_text,
    process_document,
)


def test_extract_epic_section_diagnosis():
    text = "Header\nDiagnosis: Cancer details"
    result = extract_epic_section(text)
    assert result == "Diagnosis: Cancer details"


def test_extract_epic_section_diagnoses():
    text = "Header\nDiagnoses: Multiple cancers"
    result = extract_epic_section(text)
    assert result == "Diagnoses: Multiple cancers"


def test_extract_epic_section_empty():
    text = "this is utter guff"
    result = extract_epic_section(text)
    assert result is None


# === extract_json_from_text ===


def test_extract_json_from_text_valid_json():
    text = '{"key": "value"}'
    result = extract_json_from_text(text)
    assert result == {"key": "value"}


def test_extract_json_from_text_nested():
    text = '{"key": {"nested": "value"}}'
    result = extract_json_from_text(text)
    assert result == {"key": {"nested": "value"}}


def test_extract_json_from_text_with_surrounding_text():
    text = 'Some text before {"key": "value"} some text after'
    result = extract_json_from_text(text)
    assert result == {"key": "value"}


def test_extract_json_from_text_invalid_json():
    text = "Not a JSON at all"
    with pytest.raises(ValueError):
        extract_json_from_text(text)


def test_extract_json_from_text_malformed_json():
    text = "{key: value}"  # Missing quotes
    with pytest.raises(ValueError):
        extract_json_from_text(text)


def test_extract_json_from_text_array():
    text = '[{"key": "value"}]'
    result = extract_json_from_text(text)
    assert result == [{"key": "value"}]


def test_extract_json_from_text_empty_string():
    with pytest.raises(ValueError):
        extract_json_from_text("")


# === process_document ===


@pytest.fixture
def mock_response():
    mock = Mock()
    mock.json.return_value = {
        "choices": [{"message": {"content": '{"test": "response"}'}}]
    }
    return mock


def test_process_document_success(mock_response):
    with (
        patch("oncollama_epic.main.requests.post") as mock_post,
    ):
        mock_post.return_value = mock_response

        result = process_document("Diagnosis: Cancer details")

        assert result == {"test": "response"}
        mock_post.assert_called_once()


def test_process_document_no_relevant_section():
    with pytest.raises(ValueError):
        process_document("Test document")


def test_process_document_connection_error():
    with (
        patch("oncollama_epic.main.requests.post") as mock_post,
    ):
        mock_post.side_effect = requests.ConnectionError()

        with pytest.raises(requests.ConnectionError):
            process_document("Diagnosis: Cancer details")


def test_process_document_invalid_json_response(mock_response):
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Invalid JSON"}}]
    }

    with (
        patch("oncollama_epic.main.extract_epic_section") as mock_extract,
        patch("oncollama_epic.main.requests.post") as mock_post,
    ):
        mock_extract.return_value = "Focused content"
        mock_post.return_value = mock_response

        with pytest.raises(ValueError):
            process_document("Test document")
