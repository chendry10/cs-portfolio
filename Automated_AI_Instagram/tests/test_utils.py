import os
import tempfile
from PIL import Image
import pytest

from Automated_AI_Instagram.src.utils import resize_exact_for_instagram


@pytest.fixture
def sample_image_path():
    fd, path = tempfile.mkstemp(suffix=".png")
    img = Image.new("RGB", (200, 150), color="blue")
    img.save(path)
    os.close(fd)
    yield path
    os.remove(path)


def test_resize_exact_for_instagram(sample_image_path):
    output_path = resize_exact_for_instagram(sample_image_path, 1080, 1080)
    assert os.path.exists(output_path)
    img = Image.open(output_path)
    assert img.size == (1080, 1080)
    img.close()  # Explicitly close the image
    os.remove(output_path)
