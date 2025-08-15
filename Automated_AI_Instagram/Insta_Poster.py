# Instagram Poster — Live Editor (robust meme prompt + caption generation)
# - No headlines or web search references
# - GPT-5 → gpt-4o-mini → local fallback (guaranteed non-empty)
# - Generates both image prompt & caption
# - Quiet logging: no noisy GPT errors printed

import os
import time
import json
import base64
import sys
import argparse
import requests
import tempfile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Pillow for format/size fixes
try:
    from PIL import Image, ImageOps, ImageFile, ImageColor, ImageFilter
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError as e:
    raise SystemExit("Missing dependency: Pillow. Install with `pip install pillow`") from e

# ── CONFIG ─────────────────────────────────────────────────────────────────────
ACCESS_TOKEN = os.getenv("INSTA")
BASE_URL = "https://graph.facebook.com/v23.0"

# Page selection
FALLBACK_PAGE_ID = "742486745613154"
PAGE_USERNAME = ""

# Posting behavior
GENERATE_WITH_GPT_IMAGE = True
IMAGE_MODEL = "gpt-image-1"

# Text prompt models
USE_GPT5_FOR_PROMPT = True
GPT5_MODEL = "gpt-4.1-mini"
TEXT_PROMPT_FALLBACK_MODEL = os.getenv("TEXT_FALLBACK_MODEL", "gpt-4o-mini")

# GPT-5 request includes caption requirement
GPT5_REQUEST = (
    "Create a single extremely short meme prompt for gpt-image-1 AND a short, funny Instagram caption with relevant hashtags "
    "for that meme (max 1 sentence). "
    "Use web search to find a recent (within the last 2 days), funny political news event to base the meme on. "
    "It should have the classic white meme text. "
    "The meme concept should be extremely funny and be about politics with prominent political figures. "
    "The meme text should be extremely far away from the boarder of the image so it does not get cut off."
    "Do not create cartoons."
    "Format EXACTLY as:\nPrompt: <image prompt>\nCaption: <caption>"
)

# Static prompt fallback (only used if you bypass AI completely)
IMAGE_PROMPT = (
    "Create a single short meme prompt for gpt-image-1 AND a short, funny Instagram caption "
    "for that meme (max 1 sentence). "
    "The meme concept should be extremely funny and be about politics with prominent political figures "
    "The meme text should be extremelyfar away from the boarder of the image so it does not get cut off"
    "Format EXACTLY as:\nPrompt: <image prompt>\nCaption: <caption>"
)

QUIET = False

TEST_POST = True
# If you turn off generation, this URL will be used instead:
TEST_IMAGE_URL = "https://knowledge.wharton.upenn.edu/wp-content/uploads/2016/01/compassion-600x440.jpg"
TEST_CAPTION = None  # will be set dynamically

# ── HTTP SESSION ──────────────────────────────────────────────────────────────
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

def _qprint(*args, **kwargs):
    if not QUIET:
        print(*args, **kwargs)


# ── OPENAI HELPERS ────────────────────────────────────────────────────────────
def _try_responses(client, model: str, system: str, user: str, max_tokens: int = 1000) -> str | None:
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=max_tokens,
            tools=[{"type": "web_search_preview"}],
            max_tool_calls=2,
        )
        print(f"[responses/{model}] Response: {resp}")
        return (getattr(resp, "output_text", "") or "").strip() or None
    except Exception as e:
        _qprint(f"[responses/{model}] {e}")
        return None

def get_meme_prompt_via_ai(request_text: str, system_style: str | None = None, test_mode: bool = False) -> str:
    try:
        from openai import OpenAI
    except Exception:
        print("Error occurred while initializing OpenAI client.")

    api_key = os.getenv("OPENAI")
    if not api_key:
        print("Missing OPENAI env var. Set it before running.")

    client = OpenAI(api_key=api_key)

    sys_msg = system_style or (
        "You are a helpful assistant that generates meme prompts for use in image generation for instagram."
        "Ensure that the prompt specifies that white meme text should be used on the top and bottom of the image away from the border to avoid being cutoff."
    )

    out = _try_responses(client, GPT5_MODEL, sys_msg, request_text)
    if not out:
        _qprint(f"GPT-5 responses failed")
    return out


def get_meme_prompt_and_caption(request_text: str, system_style: str | None = None, test_mode: bool = False) -> tuple[str, str]:
    raw = get_meme_prompt_via_ai(request_text, system_style, test_mode=test_mode)
    prompt, caption = raw, None
    if "Caption:" in raw:
        parts = raw.split("Caption:", 1)
        prompt = parts[0].replace("Prompt:", "").strip()
        caption = parts[1].strip()

    return prompt, caption

# ── IMAGE GEN ─────────────────────────────────────────────────────────────────
def gen_gpt_image_to_file(prompt: str, model: str = "gpt-image-1") -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI")
    if not api_key:
        raise SystemExit("Missing OPENAI env var. Set it before running.")
    client = OpenAI(api_key=api_key)

    print(f"Requesting image from model '{model}'...")
    try:
        resp = client.images.generate(model=model, prompt=prompt, n=1, output_format="jpeg", size="1024x1024", moderation="low", quality="medium")
    except Exception as e:
        raise RuntimeError(f"OpenAI Image API call failed: {e}") from e

    data = resp.data[0]
    if not hasattr(data, "b64_json") or not data.b64_json:
        raise RuntimeError("Image API did not return base64 content.")

    img_bytes = base64.b64decode(data.b64_json)
    fd, tmp_path = tempfile.mkstemp(prefix="meme_", suffix=".jpeg")
    with os.fdopen(fd, "wb") as f:
        f.write(img_bytes)
    return tmp_path

# ── IMAGE UTIL ────────────────────────────────────────────────────────────────
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
    print(f"Resized EXACT to {width}x{height}: {new_path}")
    return new_path

def prepare_for_instagram_feed(local_path: str,
                               min_width: int = 1080,
                               bg_mode: str = "blur",   # "blur" or hex like "#ffffff"
                               force_ratio: str | None = None,
                               safe_margin_pct: float = 0.06) -> str:
    """
    Aspect-ratio safe method (no cropping, no distortion) with optional blurred background.
    Kept here in case you want non-distorting behavior.
    """
    R_MIN, R_MAX = 0.75, 1.91
    ratio_map = {"1:1": 1.0, "4:5": 4/5, "3:4": 3/4, "1.91:1": 1.91, "16:9": 16/9}

    img = Image.open(local_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    r = w / h

    if force_ratio:
        target_r = max(min(ratio_map.get(force_ratio.replace(" ", ""), 1.0), R_MAX), R_MIN)
    else:
        target_r = min(max(r, R_MIN), R_MAX)

    target_w = int(max(min_width, round(min_width)))
    target_h = int(round(target_w / target_r))

    if bg_mode == "blur":
        bg = ImageOps.fit(img, (target_w, target_h), method=Image.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=40))
    else:
        bg = Image.new("RGB", (target_w, target_h), ImageColor.getrgb(bg_mode))

    inner_w = max(1, int(target_w * (1 - 2 * safe_margin_pct)))
    inner_h = max(1, int(target_h * (1 - 2 * safe_margin_pct)))
    fg = ImageOps.contain(img, (inner_w, inner_h), method=Image.LANCZOS)

    paste_x = (target_w - fg.width) // 2
    paste_y = (target_h - fg.height) // 2
    bg.paste(fg, (paste_x, paste_y))

    fd, new_path = tempfile.mkstemp(prefix="ig_ready_", suffix=".jpeg"); os.close(fd)
    bg.save(new_path, "JPEG", quality=92, optimize=True, progressive=True, subsampling=2)
    print(f"Prepared IG feed (safe) {target_w}x{target_h}: {new_path}")
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

# ── HOST UPLOADERS ────────────────────────────────────────────────────────────
def upload_to_catbox(local_path: str, retries: int = 3, backoff: float = 2.0) -> str:
    last_err = None
    for i in range(1, retries + 1):
        try:
            with open(local_path, "rb") as f:
                files = {"fileToUpload": ("meme.jpg", f, "image/jpeg")}
                data = {"reqtype": "fileupload"}
                r = SESSION.post("https://catbox.moe/user/api.php", data=data, files=files, timeout=60)
            r.raise_for_status()
            url = r.text.strip()
            if url.startswith("http://") or url.startswith("https://"):
                return url
            last_err = f"Unexpected Catbox response: {url[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff * i)
    raise RuntimeError(f"Catbox upload failed after {retries} tries: {last_err}")

def upload_with_fallbacks(local_path: str) -> str:
    hosts = []
    try:
        hosts.append(upload_to_catbox(local_path))
    except Exception as e:
        _qprint(f"Catbox upload failed: {e}")    

    for url in hosts:
        try:
            ensure_url_fetchable(url)
            return url
        except Exception as e:
            _qprint(f"Host {url} not fetchable: {e}")

    raise SystemExit("All host uploads failed.")

# ── GRAPH API HELPERS ─────────────────────────────────────────────────────────
def post_graph_with_retry(url: str, data: dict, max_tries: int = 6, base_sleep: float = 1.2):
    last_err = None
    for i in range(max_tries):
        try:
            r = SESSION.post(f"{BASE_URL}{url}", data=data, timeout=60)
            if 500 <= r.status_code < 600:
                raise RuntimeError(f"Server {r.status_code}: {r.text[:200]}")
            try:
                j = r.json()
            except Exception:
                j = {"error": f"Non-JSON response: {r.status_code}", "text": r.text}
            if "error" in j:
                code = j["error"].get("code") if isinstance(j.get("error"), dict) else None
                if code and 400 <= int(code) < 500:
                    print(f"POST {url} -> ERROR (4xx):\n{json.dumps(j, indent=2)}")
                    return j
                raise RuntimeError(json.dumps(j)[:300])
            print(f"POST {url} -> OK")
            return j
        except Exception as e:
            last_err = e
            sleep = base_sleep * (2 ** i)
            print(f"POST {url} attempt {i+1}/{max_tries} failed: {e} — retrying in {sleep:.1f}s")
            time.sleep(sleep)
    raise SystemExit(f"POST {url} failed after {max_tries} attempts: {last_err}")

def get(url, **params):
    params["access_token"] = ACCESS_TOKEN
    r = SESSION.get(f"{BASE_URL}{url}", params=params, timeout=30)
    try:
        j = r.json()
    except Exception:
        j = {"error": f"Non-JSON response: {r.status_code}", "text": r.text}
    if "error" in j:
        print(f"GET {url} -> ERROR:\n{json.dumps(j, indent=2)}")
    else:
        print(f"GET {url} -> OK")
    return j

def post(url, **data):
    data["access_token"] = ACCESS_TOKEN
    return post_graph_with_retry(url, data)

def check_token_and_perms():
    who = get("/me", fields="id,name")
    print("Token owner:", who)
    perms = get("/me/permissions")
    if "error" in perms or not perms.get("data"):
        print("Looks like a Page token — skipping user-scope check.")
        return
    required = {"pages_show_list", "instagram_basic", "instagram_content_publish"}
    granted = {p["permission"] for p in perms.get("data", []) if p.get("status") == "granted"}
    missing = required - granted
    if missing:
        raise SystemExit(f"Missing permissions on token: {missing}")
    print("Token has required permissions:", granted)

def resolve_page_id_from_username(username: str) -> str | None:
    resp = get(f"/{username}", fields="id,name")
    if "id" in resp:
        print(f"Resolved Page username '{username}' -> ID {resp['id']} ({resp.get('name')})")
        return resp["id"]
    return None

def get_page_id() -> str:
    pages = get("/me/accounts")
    ids = [p["id"] for p in pages.get("data", [])]
    if ids:
        print("Using Page from /me/accounts:", ids[0])
        return ids[0]
    if PAGE_USERNAME:
        pid = resolve_page_id_from_username(PAGE_USERNAME)
        if pid:
            return pid
    if FALLBACK_PAGE_ID:
        print("Using FALLBACK_PAGE_ID:", FALLBACK_PAGE_ID)
        return FALLBACK_PAGE_ID
    raise SystemExit(
        "No Pages returned and no fallback provided.\n"
        "- Ensure this is a USER token with pages_show_list\n"
        "- Ensure you're an ADMIN of the Page\n"
        "- Set PAGE_USERNAME or FALLBACK_PAGE_ID above"
    )

def get_ig_user_id(page_id: str) -> str:
    resp = get(f"/{page_id}", fields="instagram_business_account{id,username}")
    ig_obj = resp.get("instagram_business_account")
    if not ig_obj or not ig_obj.get("id"):
        raise SystemExit(
            "Page found but no instagram_business_account linked.\n"
            "Link IG Business to this Page: FB Page Settings → Linked accounts → Instagram."
        )
    print(f"Linked IG: @{ig_obj.get('username')} (id={ig_obj.get('id')})")
    return ig_obj["id"]

def list_media(ig_user_id: str):
    media = get(f"/{ig_user_id}/media", fields="id,caption,media_type,permalink,timestamp")
    print(json.dumps(media, indent=2))

def post_image_with_rehosts(ig_user_id: str, image_url: str, caption: str, src_file_for_rehost: str | None) -> dict:
    def try_publish(url_to_use: str) -> dict:
        container = post(f"/{ig_user_id}/media", image_url=url_to_use, caption=caption)
        if "id" not in container:
            return container
        creation_id = container["id"]
        time.sleep(2)
        published = post(f"/{ig_user_id}/media_publish", creation_id=creation_id)
        return published

    published = try_publish(image_url)
    err_msg = published.get("error") if isinstance(published, dict) else None
    if (isinstance(err_msg, dict) and int(err_msg.get("code", 0)) >= 500) or ("error" in published and not published.get("id")):
        print("Graph returned error on publish; attempting to rehost and retry...")
        if not src_file_for_rehost:
            raise SystemExit("Cannot rehost: no local file path provided.")
        new_url = upload_with_fallbacks(src_file_for_rehost)
        print("Rehosted URL:", new_url)
        ensure_url_fetchable(new_url)
        published = try_publish(new_url)

    print("Publish response:", json.dumps(published, indent=2))
    return published

# ── HELPERS: fetch remote → convert → rehost (used for last-resort fallback) ──
def download_to_temp(url: str) -> str:
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    fd, p = tempfile.mkstemp(prefix="dl_", suffix=".img")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return p

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated AI Instagram Poster.")
    parser.add_argument(
        "-p", "--generate-prompt-only",
        action="store_true",
        help="Generate a meme prompt and caption, print it, and exit without posting."
    )
    args = parser.parse_args()

    if args.generate_prompt_only:
        print("── Generating prompt and caption only (no posting, no fallbacks) ──")
        meme_prompt, caption = get_meme_prompt_and_caption(GPT5_REQUEST, test_mode=True)
        print("\n" + "="*50)
        print("  GENERATED PROMPT & CAPTION")
        print("="*50)
        print("\n[PROMPT]")
        print(meme_prompt)
        print("\n[CAPTION]")
        print(caption)
        print("\n" + "="*50)
        sys.exit(0)

    if not ACCESS_TOKEN:
        raise SystemExit("Missing INSTA env var.")

    print("── Checking token and permissions ──")
    check_token_and_perms()

    print("\n── Resolving Page ID ──")
    page_id = get_page_id()
    print("Using Page ID:", page_id)

    print("\n── Resolving IG user id ──")
    ig_user_id = get_ig_user_id(page_id)

    print("\n── Listing recent media ──")
    list_media(ig_user_id)

    original_image_path = None
    ig_ready_path = None
    try:
        if TEST_POST:
            if GENERATE_WITH_GPT_IMAGE:
                print("\n── Generating meme prompt & caption ──")
                meme_prompt, TEST_CAPTION = get_meme_prompt_and_caption(GPT5_REQUEST)
                print("Using image prompt →", meme_prompt)
                print("Using caption →", TEST_CAPTION)

                print("\n── Generating image with gpt-image-1 ──")
                original_image_path = gen_gpt_image_to_file(meme_prompt, model=IMAGE_MODEL)
                print("Saved local (jpeg):", original_image_path)

                print("Resizing EXACTLY to 1080x1080 (no borders, may distort)...")
                ig_ready_path = resize_exact_for_instagram(original_image_path, width=1080, height=1080)
                # Non-distorting option:
                #print("Preparing for IG feed (aspect-ratio safe, no cropping)...")
                #ig_ready_path = prepare_for_instagram_feed(original_image_path, min_width=1080, bg_mode="blur")

                print("Uploading to host…")
                TEST_IMAGE_URL = upload_with_fallbacks(ig_ready_path)
                print("Public URL:", TEST_IMAGE_URL)

            print("\n── Posting image ──")
            post_image_with_rehosts(ig_user_id, TEST_IMAGE_URL, TEST_CAPTION, src_file_for_rehost=ig_ready_path)

    except Exception as e:
        raise SystemExit(f"Failed: {e}")
    finally:
        for p in (original_image_path, ig_ready_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
