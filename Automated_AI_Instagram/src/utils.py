"""
Utility functions for image processing and file handling.
"""

import os
import tempfile
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any

try:
    from PIL import Image, ImageOps, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError as e:
    raise SystemExit("Missing dependency: Pillow. Install with `pip install pillow`") from e


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; AIMemesBot/1.6)"})
    return s

SESSION = make_session()

def resize_exact_for_instagram(local_path: str, width: int = 1080, height: int = 1080) -> str:
    """
    Resize to an exact size (e.g., 1080x1080) with NO borders and
    NO aspect-ratio preservation (distorts non-square images).
    """
    img = Image.open(local_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    out = img.resize((int(width), int(height)), resample=Image.LANCZOS)

    fd, new_path = tempfile.mkstemp(prefix="ig_exact_", suffix=".jpeg"); os.close(fd)
    out.save(new_path, "JPEG", quality=92, optimize=True, progressive=True, subsampling=2)
    print(f"Resized image to {width}x{height}: {new_path}")
    return new_path

def ensure_url_fetchable(url: str, timeout: int = 20, max_bytes: int = 4096, attempts: int = 4) -> None:
    last_err = None
    for i in range(attempts):
        try:
            with SESSION.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"status {resp.status_code}")
                for _ in resp.iter_content(chunk_size=max_bytes):
                    break
            return
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"URL reachability check failed after {attempts} attempts: {last_err}")
