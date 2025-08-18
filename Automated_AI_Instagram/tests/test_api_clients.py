import pytest
from unittest.mock import MagicMock, patch
import logging
import os
import base64
from openai import APIStatusError  # Import directly
import requests
import json

import Automated_AI_Instagram.src.config as config
from Automated_AI_Instagram.src.api_clients import (
    get_page_id,
    get_ig_user_id,
    get,
    get_meme_prompt_and_caption,
    get_meme_prompt_via_ai,
    gen_gpt_image_to_file,
    _try_responses,
    upload_to_catbox,
    upload_with_fallbacks,
    post_graph_with_retry,
)  # Import new functions

# Removed sys.modules mocking for openai


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setattr(config, "ACCESS_TOKEN", "test_access_token")
    monkeypatch.setattr(config, "BASE_URL", "https://graph.facebook.com/v18.0")


@pytest.fixture
def mock_openai_client():
    with patch("Automated_AI_Instagram.src.api_clients.OpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_responses_create(mock_openai_client):
    with patch.object(mock_openai_client.responses, "create") as mock_create:
        yield mock_create


@pytest.fixture
def mock_images_generate(mock_openai_client):
    with patch.object(mock_openai_client.images, "generate") as mock_generate:
        yield mock_generate


def test_get_page_id(monkeypatch, caplog):
    test_page_id = "test_page_id_123"
    monkeypatch.setattr(config, "PAGE_ID", test_page_id)

    with caplog.at_level(logging.INFO):
        result = get_page_id()

    assert result == test_page_id
    assert "Using PAGE_ID: test_page_id_123" in caplog.text


def test_get_ig_user_id_success(monkeypatch, caplog):
    mock_page_id = "test_page_id"
    mock_ig_user_id = "test_ig_user_id"
    mock_username = "test_username"

    mock_response = {
        "instagram_business_account": {"id": mock_ig_user_id, "username": mock_username}
    }

    with patch(
        "Automated_AI_Instagram.src.api_clients.get", return_value=mock_response
    ) as mock_get:
        with caplog.at_level(logging.INFO):
            result = get_ig_user_id(mock_page_id)

    mock_get.assert_called_once_with(
        f"/{mock_page_id}", fields="instagram_business_account{id,username}"
    )
    assert result == mock_ig_user_id
    assert f"Linked IG: @{mock_username} (id={mock_ig_user_id})" in caplog.text


def test_get_ig_user_id_no_instagram_business_account(monkeypatch, caplog):
    mock_page_id = "test_page_id"
    mock_response = {}

    with patch(
        "Automated_AI_Instagram.src.api_clients.get", return_value=mock_response
    ) as mock_get:
        with pytest.raises(RuntimeError) as excinfo:
            get_ig_user_id(mock_page_id)

    mock_get.assert_called_once_with(
        f"/{mock_page_id}", fields="instagram_business_account{id,username}"
    )
    assert (
        "ERROR: Page found but no 'instagram_business_account' is linked or visible."
        in str(excinfo.value)
    )


def test_get_ig_user_id_instagram_business_account_no_id(monkeypatch, caplog):
    mock_page_id = "test_page_id"
    mock_response = {"instagram_business_account": {"username": "test_username_no_id"}}

    with patch(
        "Automated_AI_Instagram.src.api_clients.get", return_value=mock_response
    ) as mock_get:
        with pytest.raises(RuntimeError) as excinfo:
            get_ig_user_id(mock_page_id)

    mock_get.assert_called_once_with(
        f"/{mock_page_id}", fields="instagram_business_account{id,username}"
    )
    assert (
        "ERROR: Page found but no 'instagram_business_account' is linked or visible."
        in str(excinfo.value)
    )


def test_get_meme_prompt_and_caption_success(caplog):
    mock_raw_output = "Prompt: This is a test prompt.\nCaption: This is a test caption."
    with patch(
        "Automated_AI_Instagram.src.api_clients.get_meme_prompt_via_ai",
        return_value=mock_raw_output,
    ) as mock_get_prompt:
        prompt, caption = get_meme_prompt_and_caption("test request")

    mock_get_prompt.assert_called_once_with("test request", None, test_mode=False)
    assert prompt == "This is a test prompt."
    assert caption == "This is a test caption."
    assert not caplog.records  # No warnings expected


def test_get_meme_prompt_and_caption_no_caption(caplog):
    mock_raw_output = "Prompt: This is a test prompt without a caption."
    with patch(
        "Automated_AI_Instagram.src.api_clients.get_meme_prompt_via_ai",
        return_value=mock_raw_output,
    ) as mock_get_prompt:
        with caplog.at_level(logging.WARNING):
            prompt, caption = get_meme_prompt_and_caption("test request")

    mock_get_prompt.assert_called_once_with("test request", None, test_mode=False)
    assert prompt == "This is a test prompt without a caption."
    assert caption == "Funny meme #politics #AI"  # Default caption
    assert (
        "Could not parse a 'Caption:' from model output. Using a default."
        in caplog.text
    )


def test_get_meme_prompt_and_caption_no_prompt(caplog):
    mock_raw_output = "Just some random text without a prompt."
    with patch(
        "Automated_AI_Instagram.src.api_clients.get_meme_prompt_via_ai",
        return_value=mock_raw_output,
    ) as mock_get_prompt:
        with pytest.raises(RuntimeError) as excinfo:
            get_meme_prompt_and_caption("test request")

    mock_get_prompt.assert_called_once_with("test request", None, test_mode=False)
    assert "Could not parse a 'Prompt:' from the model's output" in str(excinfo.value)


def test_gen_gpt_image_to_file_api_status_error_moderation_blocked(
    mock_images_generate, caplog
):
    mock_images_generate.side_effect = [
        APIStatusError(
            "moderation_blocked", response=MagicMock(status_code=400), body={}
        ),
        APIStatusError(
            "moderation_blocked", response=MagicMock(status_code=400), body={}
        ),
        APIStatusError(
            "moderation_blocked", response=MagicMock(status_code=400), body={}
        ),
        APIStatusError(
            "moderation_blocked", response=MagicMock(status_code=400), body={}
        ),
        MagicMock(
            data=[MagicMock(b64_json=base64.b64encode(b"final_image").decode("utf-8"))]
        ),  # Success on 5th attempt
    ]

    prompt = "a test image prompt"
    with caplog.at_level(logging.WARNING):
        file_path = gen_gpt_image_to_file(prompt)

    assert mock_images_generate.call_count == 5  # Should be 5 calls now
    assert "Image generation blocked by safety system. Retrying..." in caplog.text
    assert os.path.exists(file_path)
    with open(file_path, "rb") as f:
        assert f.read() == b"final_image"
    os.remove(file_path)


def test_gen_gpt_image_to_file_api_status_error_other(mock_images_generate):
    mock_images_generate.side_effect = APIStatusError(
        "some other error", response=MagicMock(status_code=500), body={}
    )

    prompt = "a test image prompt"
    with pytest.raises(
        RuntimeError, match="OpenAI Image API call failed: some other error"
    ):
        gen_gpt_image_to_file(prompt)

    mock_images_generate.assert_called_once()


def test_gen_gpt_image_to_file_general_exception(mock_images_generate):
    mock_images_generate.side_effect = Exception("network error")

    prompt = "a test image prompt"
    with pytest.raises(
        RuntimeError, match="OpenAI Image API call failed: network error"
    ):
        gen_gpt_image_to_file(prompt)

    mock_images_generate.assert_called_once()


def test_gen_gpt_image_to_file_no_b64_json(mock_images_generate):
    mock_images_generate.return_value = MagicMock(
        data=[MagicMock(b64_json=None)]  # Simulate missing b64_json
    )

    prompt = "a test image prompt"
    with pytest.raises(RuntimeError, match="Image API did not return base64 content."):
        gen_gpt_image_to_file(prompt)

    mock_images_generate.assert_called_once()


# New tests for _try_responses
def test_try_responses_import_error(monkeypatch):
    # Simulate openai not being installed
    original_import = __builtins__["__import__"]

    def mock_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(__builtins__, "__import__", mock_import)

    with pytest.raises(RuntimeError, match="Missing dependency: openai"):
        _try_responses(None, "model", "system", "user")


def test_try_responses_missing_api_key(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="Missing OPENAI env var"):
        _try_responses(None, "model", "system", "user")


def test_try_responses_success(mock_openai_client):
    mock_openai_client.responses.create.return_value = MagicMock(
        output_text="Test response text"
    )
    result = _try_responses(
        None, "model", "system", "user", openai_client=mock_openai_client
    )
    assert result == "Test response text"
    mock_openai_client.responses.create.assert_called_once()


def test_try_responses_api_exception(mock_openai_client, caplog):
    mock_openai_client.responses.create.side_effect = Exception("API error")
    with caplog.at_level(logging.ERROR):
        result = _try_responses(
            None, "model", "system", "user", openai_client=mock_openai_client
        )
    assert result is None
    assert "[responses/model] API error" in caplog.text


# New tests for get_meme_prompt_via_ai
def test_get_meme_prompt_via_ai_success(monkeypatch):
    mock_response = "Prompt: This is a test prompt.\nCaption: This is a test caption."
    with patch(
        "Automated_AI_Instagram.src.api_clients._try_responses",
        return_value=mock_response,
    ) as mock_try_responses:
        result = get_meme_prompt_via_ai("test request")
    mock_try_responses.assert_called_once_with(
        None,
        config.PRIMARY_TEXT_MODEL,
        "You are a helpful assistant that generates meme prompts for use in image generation for instagram.Ensure that the prompt specifies that white meme text should be used on the top and bottom of the image away from the border to avoid being cutoff.",
        "test request",
    )
    assert result == mock_response


def test_get_meme_prompt_via_ai_no_response(monkeypatch):
    with patch(
        "Automated_AI_Instagram.src.api_clients._try_responses", return_value=None
    ) as mock_try_responses:
        with pytest.raises(
            RuntimeError, match="Failed to generate prompt with primary model"
        ):
            get_meme_prompt_via_ai("test request")
    mock_try_responses.assert_called_once()


# New tests for upload_to_catbox
def test_upload_to_catbox_success(tmp_path):
    mock_file_path = tmp_path / "test_image.jpg"
    mock_file_path.write_bytes(b"dummy_image_data")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "https://catbox.moe/a/test_url.jpg"
    mock_response.raise_for_status.return_value = None

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.post",
        return_value=mock_response,
    ) as mock_post:
        url = upload_to_catbox(str(mock_file_path))
        assert url == "https://catbox.moe/a/test_url.jpg"
        mock_post.assert_called_once()


def test_upload_to_catbox_network_error(tmp_path):
    mock_file_path = tmp_path / "test_image.jpg"
    mock_file_path.write_bytes(b"dummy_image_data")

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.post",
        side_effect=requests.exceptions.RequestException("Network error"),
    ) as mock_post:
        with pytest.raises(
            RuntimeError, match="Catbox upload failed after 3 tries: Network error"
        ):
            upload_to_catbox(str(mock_file_path))
        assert mock_post.call_count == 3  # Retries


def test_upload_to_catbox_unexpected_response(tmp_path):
    mock_file_path = tmp_path / "test_image.jpg"
    mock_file_path.write_bytes(b"dummy_image_data")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "not_a_url"
    mock_response.raise_for_status.return_value = None

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.post",
        return_value=mock_response,
    ) as mock_post:
        with pytest.raises(
            RuntimeError,
            match="Catbox upload failed after 3 tries: Unexpected Catbox response",
        ):
            upload_to_catbox(str(mock_file_path))
        assert mock_post.call_count == 3  # Retries


def test_upload_with_fallbacks_catbox_fails(tmp_path, caplog):
    mock_file_path = tmp_path / "test_image.jpg"
    mock_file_path.write_bytes(b"dummy_image_data")

    with patch(
        "Automated_AI_Instagram.src.api_clients.upload_to_catbox",
        side_effect=RuntimeError("Catbox failed"),
    ) as mock_catbox:
        with patch(
            "Automated_AI_Instagram.src.api_clients.ensure_url_fetchable",
            return_value=None,
        ) as mock_ensure_fetchable:
            with pytest.raises(RuntimeError, match="All host uploads failed."):
                upload_with_fallbacks(str(mock_file_path))

            assert mock_catbox.call_count == 1
            mock_ensure_fetchable.assert_not_called()
            assert "Catbox upload failed: Catbox failed" in caplog.text


# New tests for get
def test_get_success(monkeypatch):
    mock_response_json = {"data": "some_data"}
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_json

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.get", return_value=mock_response
    ) as mock_get_session:
        result = get("/test_url", param1="value1")
        mock_get_session.assert_called_once_with(
            "https://graph.facebook.com/v18.0/test_url",
            params={"access_token": "test_access_token", "param1": "value1"},
            timeout=30,
        )
        assert result == mock_response_json


def test_get_non_json_response(monkeypatch, caplog):
    mock_response = MagicMock()
    mock_response.json.side_effect = json.JSONDecodeError("Not JSON", "{}", 0)
    mock_response.status_code = 200
    mock_response.text = "Not JSON content"

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.get", return_value=mock_response
    ):
        with caplog.at_level(logging.ERROR):
            result = get("/test_url")
            assert "Non-JSON response: 200" in result["error"]
            assert result["text"] == "Not JSON content"
            assert "GET /test_url -> ERROR" in caplog.text


def test_get_error_response_from_api(monkeypatch, caplog):
    mock_error_json = {"error": {"message": "API Error", "code": 100}}
    mock_response = MagicMock()
    mock_response.json.return_value = mock_error_json

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.get", return_value=mock_response
    ):
        with caplog.at_level(logging.ERROR):
            result = get("/test_url")
            assert result == mock_error_json
            assert "GET /test_url -> ERROR" in caplog.text


def test_post_graph_with_retry_5xx_error_and_retry(monkeypatch, caplog):
    mock_response_500 = MagicMock(spec=requests.Response)
    mock_response_500.status_code = 500
    mock_response_500.text = "Server Error"
    mock_response_500.reason = "Internal Server Error"
    mock_response_500.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response_500
    )

    mock_response_success = MagicMock(spec=requests.Response)
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = {"id": "success_id"}

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.post",
        side_effect=[mock_response_500, mock_response_success],
    ) as mock_post_session:
        with patch("time.sleep", return_value=None) as mock_sleep:
            with caplog.at_level(logging.WARNING):
                result = post_graph_with_retry("/test_url", {"data": "some_data"})
                assert result == {"id": "success_id"}
                assert mock_post_session.call_count == 2
                assert "Server 500" in caplog.text
                assert "retrying" in caplog.text
                mock_sleep.assert_called_once()


def test_post_graph_with_retry_4xx_error_no_retry(monkeypatch, caplog):
    mock_response_400 = MagicMock(spec=requests.Response)
    mock_response_400.status_code = 400
    mock_response_400.json.return_value = {
        "error": {"message": "Bad Request", "code": 400}
    }
    mock_response_400.reason = "Bad Request"
    mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response_400
    )

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.post",
        return_value=mock_response_400,
    ) as mock_post_session:
        with patch("time.sleep", return_value=None) as mock_sleep:
            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError, match="Client Error: 400"):
                    post_graph_with_retry("/test_url", {"data": "some_data"})
                assert mock_post_session.call_count == 1
                assert "POST /test_url -> ERROR (4xx):" in caplog.text
                mock_sleep.assert_not_called()


def test_post_graph_with_retry_non_json_response(monkeypatch, caplog):
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Not JSON", "{}", 0)
    mock_response.text = "Not JSON content"

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.post",
        return_value=mock_response,
    ) as mock_post_session:
        with patch("time.sleep", return_value=None) as mock_sleep:
            with pytest.raises(
                RuntimeError, match="Non-JSON response: 200, text: Not JSON content"
            ):
                post_graph_with_retry("/test_url", {"data": "some_data"})
            assert mock_post_session.call_count == 1
            assert mock_sleep.call_count == 0


def test_post_graph_with_retry_error_in_json_response(monkeypatch, caplog):
    mock_error_json = {"error": {"message": "API Error", "code": 500}}
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_error_json

    with patch(
        "Automated_AI_Instagram.src.api_clients.SESSION.post",
        return_value=mock_response,
    ) as mock_post_session:
        with patch("time.sleep", return_value=None) as mock_sleep:
            with pytest.raises(RuntimeError, match="API Error"):
                post_graph_with_retry("/test_url", {"data": "some_data"})
            assert mock_post_session.call_count == 1
            assert mock_sleep.call_count == 0
