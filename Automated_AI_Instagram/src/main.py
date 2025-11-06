# Main script for the Automated AI Instagram Poster.
import argparse
import os
import sys
import logging
from typing import Optional

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
        meme_prompt: str
        caption: str
        meme_prompt, caption = get_meme_prompt_and_caption(
            config.PRIMARY_MODEL_REQUEST, test_mode=True
        )
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

    original_image_path: Optional[str] = None
    ig_ready_path: Optional[str] = None
    try:
        if config.TEST_POST:
            if config.GENERATE_WITH_GPT_IMAGE:
                logging.info("\nGenerating meme prompt & caption...")
                meme_prompt, config.TEST_CAPTION = get_meme_prompt_and_caption(
                    config.PRIMARY_MODEL_REQUEST
                )
                logging.info(f"Using image prompt: {meme_prompt}")
                logging.info(f"Using caption: {config.TEST_CAPTION}")

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
            post_image_with_rehosts(
                ig_user_id,
                config.TEST_IMAGE_URL,
                config.TEST_CAPTION,
                src_file_for_rehost=ig_ready_path,
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
