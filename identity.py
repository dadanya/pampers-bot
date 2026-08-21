from __future__ import annotations

import json
import os
import re
import unicodedata


def load_sender_aliases_raw() -> dict:
    """Load the sender-aliases config.

    Checks the SENDER_ALIASES_JSON env var first (its value is the JSON
    content itself, not a path) — this is how a host like Railway, which
    doesn't get sender_aliases.json from git, can supply it. Falls back to
    the local file for normal/local runs. Returns {} if neither is available
    so callers degrade gracefully instead of crashing.
    """
    env_value = os.environ.get("SENDER_ALIASES_JSON")
    if env_value:
        try:
            return json.loads(env_value)
        except Exception:
            return {}
    path = os.path.join(os.path.dirname(__file__), "sender_aliases.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_alias_tables() -> tuple[dict, dict, dict]:
    data = load_sender_aliases_raw()
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
