from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence


MAX_REPLY_WORDS = 11
RECENT_REPLY_LIMIT = 10
SIMILARITY_THRESHOLD = 0.82
_WORD_PATTERN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_GENERIC_OPENINGS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:ну )?ты (?:серьезно|реально)(?: что ли)?(?: |$)",
        r"^(?:ну )?серьезно(?: что ли)?(?: |$)",
        r"^(?:ну )?реально(?: что ли)?(?: |$)",
        r"^(?:и )?это все(?: |$)",
        r"^ну давай(?: |$)",
        r"^опять ты(?: |$)",
    )
)


@dataclass(frozen=True)
class ReplyValidation:
    is_valid: bool
    reason: str | None = None


def count_reply_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def normalize_reply_for_comparison(text: str) -> str:
    return " ".join(_WORD_PATTERN.findall(text.casefold().replace("ё", "е")))


def validate_reply(
    text: str,
    recent_replies: Sequence[str],
) -> ReplyValidation:
    normalized = normalize_reply_for_comparison(text)
    word_count = count_reply_words(text)
    if not normalized or word_count == 0:
        return ReplyValidation(False, "empty")
    if word_count > MAX_REPLY_WORDS:
        return ReplyValidation(False, "too_long")
    if any(pattern.match(normalized) for pattern in _GENERIC_OPENINGS):
        return ReplyValidation(False, "generic_reaction")
    for recent in recent_replies[-RECENT_REPLY_LIMIT:]:
        recent_normalized = normalize_reply_for_comparison(recent)
        if not recent_normalized:
            continue
        if normalized == recent_normalized:
            return ReplyValidation(False, "repetition")
        if SequenceMatcher(None, normalized, recent_normalized).ratio() >= SIMILARITY_THRESHOLD:
            return ReplyValidation(False, "repetition")
    return ReplyValidation(True)
