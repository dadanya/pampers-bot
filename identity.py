from __future__ import annotations

import json
import os
import re
import unicodedata


def _load_alias_tables() -> tuple[dict, dict, dict]:
    path = os.path.join(os.path.dirname(__file__), "sender_aliases.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, {}, {}
    archive = dict(data.get("archive_aliases", {}))
    username = {k: v[0] for k, v in data.get("username_aliases", {}).items()}
    display = {k: v[0] for k, v in data.get("name_aliases", {}).items()}
    return archive, username, display


ARCHIVE_TO_CANONICAL, USERNAME_TO_CANONICAL, DISPLAY_TO_CANONICAL = _load_alias_tables()


def normalize_alias(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"\s+", " ", value)


def canonical_from_archive(author: str | None) -> str | None:
    target = normalize_alias(author or "")
    for alias, canonical_name in ARCHIVE_TO_CANONICAL.items():
        if normalize_alias(alias) == target:
            return canonical_name
    return None


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
