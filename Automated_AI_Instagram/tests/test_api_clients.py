import pytest
from unittest.mock import MagicMock, patch

from src.api_clients import get_meme_prompt_and_caption
import src.config as config

@pytest.fixture
def mock_openai_client():
    with patch('src.api_clients.OpenAI') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_responses_create(mock_openai_client):
    with patch.object(mock_openai_client.responses, 'create') as mock_create:
        yield mock_create

def test_get_meme_prompt_and_caption_success(mock_responses_create):
    mock_responses_create.return_value.output_text = (
        "Prompt: A photorealistic image of a cat wearing a tiny hat. "
        "TOP_TEXT=MEOW; BOTTOM_TEXT=PURRFECT; "
        "TEXT_STYLE=Impact white with black outline; TEXT_LAYOUT=top_and_bottom_centered_within_10_percent_margins\n"
        "Caption: This cat is ready for its close-up! #catmemes #cute #funny"
    )
    
    prompt, caption = get_meme_prompt_and_caption("test request")
    
    assert prompt == (
        "A photorealistic image of a cat wearing a tiny hat. "
        "TOP_TEXT=MEOW; BOTTOM_TEXT=PURRFECT; "
        "TEXT_STYLE=Impact white with black outline; TEXT_LAYOUT=top_and_bottom_centered_within_10_percent_margins"
    )
    assert caption == "This cat is ready for its close-up! #catmemes #cute #funny"

def test_get_meme_prompt_and_caption_no_caption(mock_responses_create):
    mock_responses_create.return_value.output_text = (
        "Prompt: A photorealistic image of a dog playing fetch. "
        "TOP_TEXT=GOOD BOY; BOTTOM_TEXT=FETCH; "
        "TEXT_STYLE=Impact white with black outline; TEXT_LAYOUT=top_and_bottom_centered_within_10_percent_margins\n"
    )
    
    prompt, caption = get_meme_prompt_and_caption("test request")
    
    assert prompt == (
        "A photorealistic image of a dog playing fetch. "
        "TOP_TEXT=GOOD BOY; BOTTOM_TEXT=FETCH; "
        "TEXT_STYLE=Impact white with black outline; TEXT_LAYOUT=top_and_bottom_centered_within_10_percent_margins"
    )
    assert caption == "Funny meme #politics #AI" # Default caption

def test_get_meme_prompt_and_caption_no_prompt(mock_responses_create):
    mock_responses_create.return_value.output_text = (
        "Caption: Just a test caption."
    )
    
    with pytest.raises(RuntimeError, match="Could not parse a 'Prompt:' from the model's output"):
        get_meme_prompt_and_caption("test request")

