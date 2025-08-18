# API clients for OpenAI and Facebook Graph API.
import base64
import json
import re
import tempfile
import time
import logging
from typing import Optional, Tuple, Dict, Any

from openai import OpenAI
import requests  # Added import

from .utils import SESSION, ensure_url_fetchable
from . import config as config


# ── OPENAI HELPERS ────────────────────────────────────────────────────────────
def _try_responses(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1000,
    openai_client: Optional[OpenAI] = None,
) -> Optional[str]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: openai. Install with `pip install openai`"
        ) from e

    if not config.OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI env var. Set it before running.")

    if openai_client:  # Use provided client if available
        client = openai_client
    else:  # Otherwise, create a new one
        client = OpenAI(api_key=config.OPENAI_API_KEY)
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
            temperature=2,
        )
        logging.info(f"[responses/{model}] OK")
        return (getattr(resp, "output_text", "") or "").strip() or None
    except Exception as e:
        logging.error(f"[responses/{model}] {e}")
        return None


def get_meme_prompt_via_ai(
    request_text: str, system_style: Optional[str] = None, test_mode: bool = False
) -> str:
    sys_msg = system_style or (
        "You are a helpful assistant that generates meme prompts for use in image generation for instagram."
        "Ensure that the prompt specifies that white meme text should be used on the top and bottom of the image away from the border to avoid being cutoff."
    )

    # This will raise an error and stop the script if the API call fails.
    out = _try_responses(None, config.PRIMARY_TEXT_MODEL, sys_msg, request_text)
    if not out:
        raise RuntimeError(
            f"Failed to generate prompt with primary model ({config.PRIMARY_TEXT_MODEL}). No fallbacks are configured."
        )
    return out


def get_meme_prompt_and_caption(
    request_text: str, system_style: Optional[str] = None, test_mode: bool = False
) -> Tuple[str, str]:
    raw = get_meme_prompt_via_ai(request_text, system_style, test_mode=test_mode)

    # Use regex for more robust parsing than a simple split()
    prompt_match = re.search(
        r"Prompt:\s*(.*?)(?:\nCaption:|$)", raw, re.IGNORECASE | re.DOTALL
    )
    caption_match = re.search(r"Caption:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)

    prompt = prompt_match.group(1).strip() if prompt_match else None
    caption = caption_match.group(1).strip() if caption_match else None

    # If parsing fails for the prompt, it's a critical error.
    if not prompt:
        raise RuntimeError(
            f"Could not parse a 'Prompt:' from the model's output:\n---\n{raw}\n---"
        )

    # If caption parsing fails, just use a default.
    if caption is None:
        logging.warning(
            "Could not parse a 'Caption:' from model output. Using a default."
        )
        caption = "Funny meme #politics #AI"

    return prompt, caption


# ── IMAGE GEN ─────────────────────────────────────────────────────────────────
def gen_gpt_image_to_file(prompt: str, model: str = "gpt-image-1") -> str:
    import os
    from openai import APIStatusError

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    max_retries = 5
    for attempt in range(max_retries):
        logging.info(
            f"Requesting image from model '{model}' (Attempt {attempt + 1}/{max_retries})..."
        )
        try:
            resp = client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                output_format="jpeg",
                size="1024x1024",
                moderation="low",
                quality="medium",
            )
            data = resp.data[0]
            if not hasattr(data, "b64_json") or not data.b64_json:
                raise RuntimeError("Image API did not return base64 content.")

            img_bytes = base64.b64decode(data.b64_json)
            fd, tmp_path = tempfile.mkstemp(prefix="meme_", suffix=".jpeg")
            with os.fdopen(fd, "wb") as f:
                f.write(img_bytes)
            return tmp_path
        except APIStatusError as e:
            if e.status_code == 400 and "moderation_blocked" in str(e):
                logging.warning(
                    f"Image generation blocked by safety system. Retrying... (Error: {e})"
                )
                time.sleep(2**attempt)  # Exponential backoff
            else:
                raise RuntimeError(f"OpenAI Image API call failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"OpenAI Image API call failed: {e}") from e

    raise RuntimeError(
        f"Failed to generate image after {max_retries} attempts due to moderation blocks."
    )


# ── HOST UPLOADERS ────────────────────────────────────────────────────────────
def upload_to_catbox(local_path: str, retries: int = 3, backoff: float = 2.0) -> str:
    last_err = None
    for i in range(1, retries + 1):
        try:
            with open(local_path, "rb") as f:
                files = {"fileToUpload": ("meme.jpg", f, "image/jpeg")}
                data = {"reqtype": "fileupload"}
                r = SESSION.post(
                    "https://catbox.moe/user/api.php",
                    data=data,
                    files=files,
                    timeout=60,
                )
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
        logging.error(f"Catbox upload failed: {e}")

    for url in hosts:
        try:
            ensure_url_fetchable(url)
            return url
        except Exception as e:
            logging.error(f"Host {url} not fetchable: {e}")

    raise RuntimeError("All host uploads failed.")


# ── GRAPH API HELPERS ─────────────────────────────────────────────────────────
def post_graph_with_retry(
    url: str, data: Dict[str, Any], max_tries: int = 6, base_sleep: float = 1.2
) -> Dict[str, Any]:
    last_err = None
    for i in range(max_tries):
        try:
            r = SESSION.post(f"{config.BASE_URL}{url}", data=data, timeout=60)
            r.raise_for_status()  # This will raise HTTPError for 4xx/5xx responses

            try:
                j = r.json()
            except json.JSONDecodeError:
                # If response is not JSON, it's a non-retryable error
                raise RuntimeError(
                    f"Non-JSON response: {r.status_code}, text: {r.text}"
                )

            if "error" in j:
                code = (
                    j["error"].get("code") if isinstance(j.get("error"), dict) else None
                )
                if code and 400 <= int(code) < 500:
                    logging.error(
                        f"POST {url} -> ERROR (4xx):\n{json.dumps(j, indent=2)}"
                    )
                    return j  # Return immediately for 4xx errors
                else:
                    # Other errors in JSON, including 5xx from API itself, or other client errors
                    # These are considered non-retryable if not caught by raise_for_status
                    raise RuntimeError(json.dumps(j)[:300])

            logging.info(f"POST {url} -> OK")
            return j
        except requests.exceptions.ConnectionError as e:
            last_err = e
            sleep = base_sleep * (2**i)
            logging.warning(
                f"POST {url} attempt {i+1}/{max_tries} failed: Connection Error — retrying in {sleep:.1f}s"
            )
            time.sleep(sleep)
        except requests.exceptions.Timeout as e:
            last_err = e
            sleep = base_sleep * (2**i)
            logging.warning(
                f"POST {url} attempt {i+1}/{max_tries} failed: Timeout Error — retrying in {sleep:.1f}s"
            )
            time.sleep(sleep)
        except requests.exceptions.HTTPError as e:
            last_err = e
            if 500 <= e.response.status_code < 600:
                # Retry for 5xx server errors
                sleep = base_sleep * (2**i)
                logging.warning(
                    f"POST {url} attempt {i+1}/{max_tries} failed: Server {e.response.status_code} — retrying in {sleep:.1f}s"
                )
                time.sleep(sleep)
            else:
                # Re-raise for 4xx client errors (non-retryable)
                logging.error(f"POST {url} -> ERROR (4xx):\n{e.response.text}")
                raise RuntimeError(
                    f"Client Error: {e.response.status_code}, {e.response.text[:200]}"
                ) from e
        except RuntimeError:
            # Catch RuntimeErrors raised for non-JSON or other JSON errors
            # These are non-retryable
            raise
        except Exception as e:
            # Catch any other unexpected exceptions and re-raise as non-retryable
            raise RuntimeError(f"Unexpected error: {e}") from e

    raise RuntimeError(f"POST {url} failed after {max_tries} attempts: {last_err}")


def get(url: str, **params: Any) -> Dict[str, Any]:
    params["access_token"] = config.ACCESS_TOKEN
    r = SESSION.get(f"{config.BASE_URL}{url}", params=params, timeout=30)
    try:
        j = r.json()
    except Exception:
        j = {"error": f"Non-JSON response: {r.status_code}", "text": r.text}
    if "error" in j:
        logging.error(f"GET {url} -> ERROR:\n{json.dumps(j, indent=2)}")
    else:
        logging.info(f"GET {url} -> OK")
    return j


def post(url: str, **data: Any) -> Dict[str, Any]:
    data["access_token"] = config.ACCESS_TOKEN
    return post_graph_with_retry(url, data)


def get_page_id() -> str:
    logging.info(f"Using PAGE_ID: {config.PAGE_ID}")
    return config.PAGE_ID


def get_ig_user_id(page_id: str) -> str:
    resp = get(f"/{page_id}", fields="instagram_business_account{id,username}")
    ig_obj = resp.get("instagram_business_account")
    if not ig_obj or not ig_obj.get("id"):
        error_message = (
            "ERROR: Page found but no 'instagram_business_account' is linked or visible.\n\n"
            "This can happen for a few reasons:\n"
            "1. The linked Instagram account is not a 'Business' or 'Creator' account.\n"
            "   -> Please convert it in the Instagram app settings.\n"
            "2. The Access Token is missing permissions. It needs at least:\n"
            "   -> instagram_management, pages_read_engagement, instagram_content_publish\n"
            "3. The API access connection needs to be re-confirmed.\n"
            "   -> Go to 'Facebook Business Integrations' for your user, remove your app, then re-authenticate.\n\n"
            f"API Response for page {page_id}:\n{json.dumps(resp, indent=2)}"
        )
        raise RuntimeError(error_message)
    logging.info(f"Linked IG: @{ig_obj.get('username')} (id={ig_obj.get('id')})")
    return ig_obj["id"]


def post_image_with_rehosts(
    ig_user_id: str, image_url: str, caption: str, src_file_for_rehost: Optional[str]
) -> Dict[str, Any]:
    def try_publish(url_to_use: str) -> Dict[str, Any]:
        container = post(f"/{ig_user_id}/media", image_url=url_to_use, caption=caption)
        if "id" not in container:
            raise RuntimeError(f"Failed to create media container: {container}")
        creation_id = container["id"]
        time.sleep(2)
        published = post(f"/{ig_user_id}/media_publish", creation_id=creation_id)
        if "id" not in published:
            raise RuntimeError(f"Failed to publish media: {published}")
        return published

    try:
        published = try_publish(image_url)
    except RuntimeError as e:
        logging.warning(
            "Graph returned error on publish; attempting to rehost and retry..."
        )
        if not src_file_for_rehost:
            raise RuntimeError("Cannot rehost: no local file path provided.") from e
        new_url = upload_with_fallbacks(src_file_for_rehost)
        logging.info(f"Rehosted URL: {new_url}")
        ensure_url_fetchable(new_url)
        try:  # Added try-except for rehosted publish
            published = try_publish(new_url)
        except RuntimeError as rehost_e:
            # If rehosted publish fails, re-raise the rehost error
            raise rehost_e

    if "id" in published:
        logging.info("Successfully posted to Instagram.")
    else:
        logging.error("Failed to post to Instagram.")
        logging.error(f"Publish response: {json.dumps(published, indent=2)}")
    return published
