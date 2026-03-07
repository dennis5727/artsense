import os
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def extract_artist_name(question: str, artist_names: list[str]) -> str:
    """
    Use Claude Haiku to extract/resolve the artist name being asked about.
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
    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def explain_artwork(artist_row: dict, user_context: str) -> str:
    """
    Use Claude Sonnet to generate a beginner-friendly educational explanation.
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
    response = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
