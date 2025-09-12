"""
Utility functions for image processing and file handling.
"""

import os
import tempfile
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

try:
    from PIL import Image, ImageOps, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError as e:
    raise SystemExit(
        "Missing dependency: Pillow. Install with `pip install pillow`"
    ) from e


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; AIMemesBot/1.6)"})
    return s


SESSION = make_session()


def resize_exact_for_instagram(
    local_path: str, width: int = 1080, height: int = 1080
) -> str:
    """
    Resize to an exact size (e.g., 1080x1080) while preserving aspect ratio.
    Uses background blur for padding instead of solid color.
    """
    img = Image.open(local_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    
    # Calculate aspect ratios
    target_ratio = width / height
    img_ratio = img.width / img.height
    
    if img_ratio > target_ratio:
        # Image is wider - fit to width
        new_width = width
        new_height = int(width / img_ratio)
    else:
        # Image is taller - fit to height
        new_height = height
        new_width = int(height * img_ratio)
    
    # Create background by scaling the image to fill and blurring it
    background = img.copy()
    background = background.resize((width, height), resample=Image.LANCZOS)
    from PIL import ImageFilter
    background = background.filter(ImageFilter.GaussianBlur(radius=30))
    
    # Resize main image maintaining aspect ratio
    img = img.resize((new_width, new_height), resample=Image.LANCZOS)
    
    # Paste the main image onto the blurred background
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2
    background.paste(img, (paste_x, paste_y))
    out = background

    fd, new_path = tempfile.mkstemp(prefix="ig_exact_", suffix=".jpeg")
    os.close(fd)
    out.save(
        new_path, "JPEG", quality=92, optimize=True, progressive=True, subsampling=2
    )
    logging.info(f"Resized image to {width}x{height}: {new_path}")
    return new_path


def ensure_url_fetchable(
    url: str, timeout: int = 20, max_bytes: int = 4096, attempts: int = 4
) -> None:
    last_err = None
    for i in range(attempts):
        try:
            with SESSION.get(
                url, stream=True, timeout=timeout, allow_redirects=True
            ) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"status {resp.status_code}")
                for _ in resp.iter_content(chunk_size=max_bytes):
                    break
            return
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(
        f"URL reachability check failed after {attempts} attempts: {last_err}"
    )
