"""Errors the CLI reports rather than recovers from. Kept apart from `schema` so catching one does
not import pydantic on every CLI start."""
from __future__ import annotations


class QuestionError(ValueError):
    """A question bank the server would refuse with a 422."""
