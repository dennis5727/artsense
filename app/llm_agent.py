import logging
import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def extract_artist_name(question: str, artist_names: list[str]) -> str:
    """
    Use Groq (Llama 3.3 70B) to extract/resolve the artist name being asked about.
    Returns a best-guess artist name string (may still need fuzzy matching).
    """
    names_list = "\n".join(f"- {n}" for n in artist_names)
    prompt = (
        f"You are helping resolve which artist a user is asking about.\n\n"
        f"The user asked: \"{question}\"\n\n"
        f"Here are the 50 artists in our collection:\n{names_list}\n\n"
        f"Based on the question, which artist from the list above is the user most likely referring to? "
        f"Reply with only the artist's full name exactly as it appears in the list above. "
        f"If the question does not refer to any artist in the list, reply with 'UNKNOWN'."
    )
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception("Groq extract_artist_name failed (question=%r)", question)
        raise


def explain_artwork(artist_row: dict, user_context: str) -> str:
    """
    Use Groq (Llama 3.3 70B) to generate a beginner-friendly educational explanation.
    `artist_row` is a dict from artists.csv; `user_context` is the original user input.
    """
    metadata = "\n".join(
        f"{k}: {v}"
        for k, v in artist_row.items()
        if k != "wikipedia" and pd.notna(v)
    )
    prompt = (
        f"You are an art history tutor helping a beginner student learn about artists.\n\n"
        f"The user asked or uploaded something related to: \"{user_context}\"\n\n"
        f"Here is factual information about the artist from our database:\n{metadata}\n\n"
        f"Write a warm, engaging, beginner-friendly response (3-4 paragraphs) that:\n"
        f"1. Introduces who the artist is and their place in art history\n"
        f"2. Describes their distinctive style and what makes their work recognisable\n"
        f"3. Mentions 1-2 of their most famous works and why they matter\n"
        f"4. Answers the user's specific question if they asked one\n\n"
        f"Avoid jargon. Write as if explaining to a curious student with no art history background."
    )
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        logger.exception(
            "Groq explain_artwork failed (artist=%r, user_context=%r)",
            artist_row.get("name"), user_context,
        )
        raise
