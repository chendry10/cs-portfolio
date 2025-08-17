import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock the openai module before importing src.api_clients
sys.modules['openai'] = MagicMock()
sys.modules['openai.resources'] = MagicMock() # For client.responses
sys.modules['openai.resources.images'] = MagicMock() # For client.images

from Automated_AI_Instagram.src.api_clients import get_meme_prompt_and_caption
import Automated_AI_Instagram.src.config as config

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test_openai_key")

@pytest.fixture
def mock_openai_client():
    # Now that openai is mocked at sys.modules level, we can patch the specific client
    with patch('openai.OpenAI') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_responses_create(mock_openai_client):
    with patch.object(mock_openai_client.responses, 'create') as mock_create:
        yield mock_create

