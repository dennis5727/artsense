import random
from pathlib import Path
from PIL import Image

from app import classifier, llm_agent, fuzzy_match

IMAGES_DIR = Path(__file__).parent.parent / "images"
DECLINE_MESSAGE = (
    "We were unable to confidently identify the artist of this artwork. "
    "The painting may be by an artist not represented in our collection, "
    "a contemporary or modern work, or an AI-generated image. "
    "Our system specialises in historical Western art from the 15th to early 20th century."
)
NOT_FOUND_MESSAGE = (
    "We could not find that artist in our collection. "
    "ArtSense AI covers 50 historical Western artists from the 15th to early 20th century. "
    "Try asking about artists like Vincent van Gogh, Claude Monet, or Pablo Picasso."
)


def _sample_images(artist_name: str, n: int = 5) -> list[str]:
    """Return up to n image file paths for the given artist."""
    folder_name = artist_name.replace(" ", "_")
    artist_dir = IMAGES_DIR / folder_name
    if not artist_dir.exists():
        return []
    images = list(artist_dir.glob("*.jpg")) + list(artist_dir.glob("*.jpeg")) + list(artist_dir.glob("*.png"))
    random.shuffle(images)
    return [str(p) for p in images[:n]]


def run_image_pipeline(image: Image.Image) -> dict:
    """
    Pipeline A: classify a painting and return an explanation.

    Returns:
      - artist (str)
      - confidence (float)
      - top3 (list)
      - explanation (str)
      - sample_images (list of str paths)
      - declined (bool)
    """
    result = classifier.classify_painting(image)

    if result["below_threshold"]:
        return {
            "artist": "Unknown",
            "confidence": result["confidence"],
            "top3": result["top3"],
            "explanation": DECLINE_MESSAGE,
            "sample_images": [],
            "declined": True,
        }

    artist_name = result["artist"]
    artist_row = fuzzy_match.match_artist(artist_name)
    if artist_row is None:
        return {
            "artist": artist_name,
            "confidence": result["confidence"],
            "top3": result["top3"],
            "explanation": DECLINE_MESSAGE,
            "sample_images": [],
            "declined": True,
        }

    explanation = llm_agent.explain_artwork(artist_row, user_context=f"identified painting by {artist_name}")
    return {
        "artist": artist_name,
        "confidence": result["confidence"],
        "top3": result["top3"],
        "explanation": explanation,
        "sample_images": _sample_images(artist_name),
        "declined": False,
    }


def run_text_pipeline(question: str) -> dict:
    """
    Pipeline B: answer a natural language question about an artist.

    Returns:
      - artist (str)
      - explanation (str)
      - sample_images (list of str paths)
      - declined (bool)
    """
    artist_names = fuzzy_match.get_artist_names()
    extracted = llm_agent.extract_artist_name(question, artist_names)

    if extracted == "UNKNOWN":
        return {
            "artist": "Unknown",
            "explanation": NOT_FOUND_MESSAGE,
            "sample_images": [],
            "declined": True,
        }

    artist_row = fuzzy_match.match_artist(extracted)
    if artist_row is None:
        return {
            "artist": extracted,
            "explanation": NOT_FOUND_MESSAGE,
            "sample_images": [],
            "declined": True,
        }

    artist_name = artist_row["name"]
    explanation = llm_agent.explain_artwork(artist_row, user_context=question)
    return {
        "artist": artist_name,
        "explanation": explanation,
        "sample_images": _sample_images(artist_name),
        "declined": False,
    }
