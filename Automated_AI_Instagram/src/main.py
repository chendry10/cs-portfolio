# Main script for the Automated AI Instagram Poster.
import argparse
import os
import sys
import logging
import json
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Add the parent directory of src (Automated_AI_Instagram) to sys.path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(
    os.path.join(script_dir, "..")
)  # This is Automated_AI_Instagram
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import src.config as config  # noqa: E402
from src.api_clients import (  # noqa: E402
    get_ig_user_id,
    get_meme_prompt_and_caption,
    gen_gpt_image_to_file,
    post_image_with_rehosts,
    upload_with_fallbacks,
    get_page_id,
)
from src.utils import resize_exact_for_instagram  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

POST_HISTORY_PATH = os.path.join(parent_dir, "post_history.json")


def _load_post_history(path: str = POST_HISTORY_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        logging.warning("Post history file is not a list. Resetting history.")
        return []
    except Exception as e:
        logging.warning(f"Could not read post history ({path}): {e}")
        return []


def _save_post_history(history: List[Dict[str, Any]], path: str = POST_HISTORY_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
    except Exception as e:
        logging.warning(f"Could not write post history ({path}): {e}")


def _extract_primary_person(prompt: str) -> str:
    text = prompt.split("TOP_TEXT=")[0]
    stopwords = {
        "Prompt",
        "Caption",
        "Impact",
        "Instagram",
        "Meme",
        "Top",
        "Bottom",
        "White",
        "Black",
        "Text",
        "Style",
        "Layout",
        "The",
        "A",
        "An",
    }

    candidates: List[str] = []
    for match in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
        name = match.strip()
        if name in stopwords:
            continue
        if len(name) < 3:
            continue
        candidates.append(name)

    if not candidates:
        return "Unknown"
    return candidates[0]


def _build_rotation_constraints(recent_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    recent_7 = recent_posts[:7]
    last_primary = (recent_posts[0].get("primary_person") or "").strip() if recent_posts else ""
    trump_count = sum(
        1
        for post in recent_7
        if "trump" in (post.get("primary_person") or "").strip().lower()
    )

    recent_people = [
        (post.get("primary_person") or "Unknown").strip() or "Unknown"
        for post in recent_7
    ]

    constraints = (
        "\n\nRotation constraints (HARD REQUIREMENTS):\n"
        "1) Do NOT use the same primary person as the most recent published post.\n"
        "2) At most 2 of the last 7 published posts may feature Trump as the primary person.\n"
        f"Recent primary people (most recent first): {recent_people if recent_people else ['None yet']}.\n"
        f"Current Trump count in last 7: {trump_count}.\n"
        "If constraints would be violated, choose a different primary person.\n"
    )

    return {
        "last_primary": last_primary,
        "trump_count": trump_count,
        "request_suffix": constraints,
    }


def _violates_rotation(primary_person: str, last_primary: str, trump_count: int) -> bool:
    primary_lower = (primary_person or "").strip().lower()
    if not primary_lower:
        return False
    if last_primary and primary_lower == last_primary.strip().lower():
        return True
    if "trump" in primary_lower and trump_count >= 2:
        return True
    return False


def main() -> None:
    """Main function to run the Instagram poster."""
    parser = argparse.ArgumentParser(description="Automated AI Instagram Poster.")
    parser.add_argument(
        "-p",
        "--generate-prompt-only",
        action="store_true",
        help="Generate a meme prompt and caption, print it, and exit without posting.",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.generate_prompt_only:
        logging.info("Generating prompt and caption only...")
        history = _load_post_history()
        rotation = _build_rotation_constraints(history)
        meme_prompt: str = ""
        caption: str = ""
        for attempt in range(1, 6):
            request_text = config.PRIMARY_MODEL_REQUEST + rotation["request_suffix"]
            meme_prompt, caption = get_meme_prompt_and_caption(request_text, test_mode=True)
            primary_person = _extract_primary_person(meme_prompt)
            if _violates_rotation(
                primary_person,
                rotation["last_primary"],
                rotation["trump_count"],
            ):
                logging.warning(
                    f"Prompt-only generation attempt {attempt}/5 violated rotation rules (primary person: {primary_person}). Retrying..."
                )
                continue
            break
        else:
            raise RuntimeError("Could not generate a prompt that satisfies rotation rules after 5 attempts.")

        logging.info(f"\nPrompt: {meme_prompt}\n\nCaption: {caption}\n")
        sys.exit(0)

    if not config.ACCESS_TOKEN:
        logging.error("Missing INSTA env var.")
        sys.exit(1)

    logging.info("\nResolving Page ID...")
    page_id: str = get_page_id()
    logging.info(f"Using Page ID: {page_id}")

    logging.info("\nResolving IG user id...")
    ig_user_id: str = get_ig_user_id(page_id)
    history = _load_post_history()
    rotation = _build_rotation_constraints(history)

    original_image_path: Optional[str] = None
    ig_ready_path: Optional[str] = None
    try:
        if config.TEST_POST:
            if config.GENERATE_WITH_GPT_IMAGE:
                logging.info("\nGenerating meme prompt & caption...")
                meme_prompt = ""
                primary_person = "Unknown"
                for attempt in range(1, 6):
                    request_text = config.PRIMARY_MODEL_REQUEST + rotation["request_suffix"]
                    meme_prompt, config.TEST_CAPTION = get_meme_prompt_and_caption(
                        request_text
                    )
                    primary_person = _extract_primary_person(meme_prompt)
                    if _violates_rotation(
                        primary_person,
                        rotation["last_primary"],
                        rotation["trump_count"],
                    ):
                        logging.warning(
                            f"Generation attempt {attempt}/5 violated rotation rules (primary person: {primary_person}). Retrying..."
                        )
                        continue
                    break
                else:
                    raise RuntimeError(
                        "Could not generate a meme that satisfies rotation rules after 5 attempts."
                    )

                logging.info(f"Using image prompt: {meme_prompt}")
                logging.info(f"Using caption: {config.TEST_CAPTION}")
                logging.info(f"Primary person selected: {primary_person}")

                logging.info("\nGenerating image with gpt-image-1...")
                original_image_path = gen_gpt_image_to_file(
                    meme_prompt, model=config.IMAGE_MODEL
                )
                logging.info(f"Saved local (jpeg): {original_image_path}")

                logging.info("Skipping resize for Instagram; using original image")
                # Do not resize — use the original image to avoid cropping or padding.
                # This keeps the existing code path (ig_ready_path used later for upload)
                ig_ready_path = original_image_path

                logging.info("Uploading to host...")
                config.TEST_IMAGE_URL = upload_with_fallbacks(ig_ready_path)
                logging.info(f"Public URL: {config.TEST_IMAGE_URL}")

            logging.info("\nPosting image to Instagram...")
            publish_result = post_image_with_rehosts(
                ig_user_id,
                config.TEST_IMAGE_URL,
                config.TEST_CAPTION,
                src_file_for_rehost=ig_ready_path,
            )

            published_id = publish_result.get("id")
            if published_id:
                new_entry = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "primary_person": primary_person if config.GENERATE_WITH_GPT_IMAGE else "Unknown",
                    "caption": config.TEST_CAPTION,
                    "published_id": published_id,
                }
                updated_history = [new_entry] + history
                _save_post_history(updated_history[:200])
                logging.info(
                    f"Saved post history to {POST_HISTORY_PATH}. Last 7 Trump count was {rotation['trump_count']}."
                )

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        for p in (original_image_path, ig_ready_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logging.warning(f"Could not remove temporary file {p}: {e}")


if __name__ == "__main__":
    main()
