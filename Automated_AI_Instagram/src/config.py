#Configuration for the Automated AI Instagram Poster.

import os

# --- API KEYS ---
ACCESS_TOKEN = os.getenv("INSTA")
OPENAI_API_KEY = os.getenv("OPENAI")

# --- FACEBOOK GRAPH API ---
BASE_URL = "https://graph.facebook.com/v23.0"
PAGE_ID = "742486745613154"
PAGE_USERNAME = ""

# --- OPENAI API ---
GENERATE_WITH_GPT_IMAGE = True
IMAGE_MODEL = "gpt-image-1"
USE_PRIMARY_MODEL_FOR_PROMPT = True
PRIMARY_TEXT_MODEL = "gpt-4.1-mini"
TEXT_PROMPT_FALLBACK_MODEL = os.getenv("TEXT_FALLBACK_MODEL", "gpt-4o-mini")

# --- POSTING BEHAVIOR ---
QUIET = False
TEST_POST = True

# --- TEST DATA ---
# If you turn off generation, this URL will be used instead:
TEST_IMAGE_URL = "https://knowledge.wharton.upenn.edu/wp-content/uploads/2016/01/compassion-600x440.jpg"
TEST_CAPTION = None  # This gets set by the AI if generation is on

# --- PROMPTS ---
PRIMARY_MODEL_REQUEST = (
    "Create ONE photorealistic meme prompt for gpt-image-1 and ONE short Instagram caption (≤1 sentence, 2–4 hashtags). "
    "Use web search to find a fresh political headline from the last 24 hours and base the meme on it (use the actual public figure(s); do not quote the headline). "
    "You MUST include classic meme text **rendered inside the image**: Impact/Impact-like, WHITE with BLACK outline, ALL CAPS, centered. "
    "There must be BOTH a TOP line and a BOTTOM line. Keep both inside an 8–10% safe margin; if text overflows, SHRINK TEXT TO FIT; never omit. "
    "Avoid party scenes/flag backdrops unless the headline is a celebration; choose a mundane location and specify a camera angle (overhead/extreme close-up/dutch). "
    "Output EXACTLY:\n"
    "Prompt: <1–2 sentence scene with named person(s), location, action, and angle. Then append EXACT directives: "
    "TOP_TEXT=<YOUR TOP LINE IN ALL CAPS>; BOTTOM_TEXT=<YOUR BOTTOM LINE IN ALL CAPS>; "
    "TEXT_STYLE=Impact white with black outline; TEXT_LAYOUT=top_and_bottom_centered_within_10_percent_margins>\n"
    "Caption: <funny ≤1 sentence with 2–4 relevant hashtags>"
    "Format EXACTLY as:\nPrompt: <image prompt>\nCaption: <caption>"
)

