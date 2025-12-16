
from __future__ import annotations
from .typing_ import GenreSet

def is_relevant(genre_q: set, genre_t: set) -> int:
    return 1 if (genre_q and genre_t and len(genre_q.intersection(genre_t)) > 0) else 0
