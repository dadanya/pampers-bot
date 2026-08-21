from __future__ import annotations

import re
import unicodedata


ARCHIVE_TO_CANONICAL = {
    ".": "Арина",
    "tanaka": "Андрей",
    ". Sür Маленький": "Сюр",
    "V0VAH?": "Вовах",
    "Чернобыль": "Чернобыль",
    "ꀘꍏ꓄ꃅꍟꋪꀤꈤꍟ": "Катя",
    "Denis": "Денис",
    "Анастасия": "Настя",
    "fiya zimmerman🎚️": "Фия",
    "Мария": "Маша",
}

USERNAME_TO_CANONICAL = {
    "chirishkin": "Арина",
    "sarahlouisekerrigan": "Андрей",
    "surstromming": "Сюр",
    "evilgeniusforever": "Вовах",
    "cchhrnobyl": "Чернобыль",
    "kvtya_jin": "Катя",
    "denisxcmr": "Денис",
    "anastasiia04n": "Настя",
    "fiyazaykatixitx": "Фия",
}

DISPLAY_TO_CANONICAL = {"мария": "Маша"}


def normalize_alias(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"\s+", " ", value)


_ARCHIVE_NORMALIZED = {
    normalize_alias(alias): canonical_name
    for alias, canonical_name in ARCHIVE_TO_CANONICAL.items()
}


def canonical_from_archive(author: str | None) -> str | None:
    return _ARCHIVE_NORMALIZED.get(normalize_alias(author or ""))


def canonical_from_telegram(
    username: str | None,
    display_name: str | None,
) -> str | None:
    by_username = USERNAME_TO_CANONICAL.get(normalize_alias(username or ""))
    if by_username:
        return by_username
    return DISPLAY_TO_CANONICAL.get(normalize_alias(display_name or ""))


def archive_aliases_for(canonical_name: str) -> tuple[str, ...]:
    return tuple(
        alias
        for alias, mapped_name in ARCHIVE_TO_CANONICAL.items()
        if mapped_name == canonical_name
    )
