import os
import tempfile
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from Automated_AI_Instagram.src.utils import resize_exact_for_instagram
from Automated_AI_Instagram.src.utils import (
    resize_exact_for_instagram,
    ensure_url_fetchable,
)

@pytest.fixture
def sample_image_path():
    fd, path = tempfile.mkstemp(suffix=".png")
    img = Image.new('RGB', (200, 150), color = 'blue')
    img.save(path)
    os.close(fd)
    yield path
    os.remove(path)

def test_resize_exact_for_instagram(sample_image_path):
    output_path = resize_exact_for_instagram(sample_image_path, 1080, 1080)
    assert os.path.exists(output_path)
    img = Image.open(output_path)
    assert img.size == (1080, 1080)
    img.close() # Explicitly close the image
    os.remove(output_path)

def test_ensure_url_fetchable_success():
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"data"]

    with patch(
        "Automated_AI_Instagram.src.utils.SESSION.get", return_value=mock_resp
    ) as mock_get:
        ensure_url_fetchable("http://example.com")
        mock_get.assert_called_once()


def test_ensure_url_fetchable_failure():
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    mock_resp.status_code = 404
    mock_resp.iter_content.return_value = [b"data"]

    with patch(
        "Automated_AI_Instagram.src.utils.SESSION.get", return_value=mock_resp
    ) as mock_get:
        with patch("time.sleep", return_value=None) as mock_sleep:
            with pytest.raises(RuntimeError, match="URL reachability check failed"):
                ensure_url_fetchable("http://example.com", attempts=3)
            assert mock_get.call_count == 3
            assert mock_sleep.call_count == 3
