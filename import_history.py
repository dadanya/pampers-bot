from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Sequence

from archive_parser import ArchiveMessage, parse_export
from identity import (
    ARCHIVE_TO_CANONICAL,
    archive_aliases_for,
    canonical_from_archive,
)
from memory import MemoryStore, MessageRecord
from memory_summaries import (
    EpisodeSummarizer,
    SummaryInput,
    build_relationship_summary_prompt,
    sanitize_source_for_model,
    update_relationship_summary_blocks,
)


_BOT_ARCHIVE_AUTHOR = "Памперс2004"
_ARCHIVE_CHAT_KEY = "archive:telegram-export"
_SUMMARY_BATCH_SIZE = 20
_DEFAULT_EXTERNAL_CALL_BUDGET = 100


@dataclass(frozen=True)
class EpisodeDraft:
    canonical_name: str
    messages: tuple[ArchiveMessage, ...]
    direct_exchange_count: int
    direct_reply_pairs: tuple[tuple[int, int], ...]

    @property
    def message_ids(self) -> tuple[int, ...]:
        return tuple(message.id for message in self.messages)

    @property
    def started_at(self) -> datetime:
        return min(message.sent_at for message in self.messages)

    @property
    def ended_at(self) -> datetime:
        return max(message.sent_at for message in self.messages)


@dataclass(frozen=True)
class ImportStats:
    parsed_messages: int
    created_messages: int
    created_episodes: int
    ready_episodes: int
    pending_episodes: int
    failed_episodes: int
    summary_requests: int


def _direct_user(
    message: ArchiveMessage,
    parent: ArchiveMessage,
) -> str | None:
    if message.author == _BOT_ARCHIVE_AUTHOR:
        return canonical_from_archive(parent.author)
    if parent.author == _BOT_ARCHIVE_AUTHOR:
        return canonical_from_archive(message.author)
    return None


def _window_indices(
    messages: Sequence[ArchiveMessage],
    anchor_index: int,
    window_size: int,
    max_gap: timedelta,
) -> set[int]:
    indices = {anchor_index}
    anchor_time = messages[anchor_index].sent_at
    for direction in (-1, 1):
        for distance in range(1, window_size + 1):
            candidate_index = anchor_index + direction * distance
            if candidate_index < 0 or candidate_index >= len(messages):
                break
            if abs(messages[candidate_index].sent_at - anchor_time) > max_gap:
                break
            indices.add(candidate_index)
    return indices


def build_episode_drafts(
    messages: Sequence[ArchiveMessage],
    window_size: int = 3,
    max_gap_minutes: int = 15,
) -> list[EpisodeDraft]:
    message_index_by_id = {
        message.id: index for index, message in enumerate(messages)
    }
    windows_by_user: dict[
        str, list[tuple[set[int], tuple[int, int]]]
    ] = {}
    max_gap = timedelta(minutes=max_gap_minutes)

    for index, message in enumerate(messages):
        if message.reply_to is None:
            continue
        parent_index = message_index_by_id.get(message.reply_to)
        if parent_index is None:
            continue
        canonical_name = _direct_user(message, messages[parent_index])
        if canonical_name is None:
            continue
        windows_by_user.setdefault(canonical_name, []).append(
            (
                _window_indices(messages, index, window_size, max_gap)
                | _window_indices(messages, parent_index, window_size, max_gap),
                (message.id, messages[parent_index].id),
            )
        )

    drafts: list[EpisodeDraft] = []
    for canonical_name, windows in windows_by_user.items():
        merged: list[tuple[set[int], set[tuple[int, int]]]] = []
        for window, direct_reply_pair in windows:
            indices = set(window)
            direct_reply_pairs = {direct_reply_pair}
            while True:
                overlapping = [
                    existing
                    for existing in merged
                    if existing[0].intersection(indices)
                ]
                if not overlapping:
                    break
                for existing_indices, existing_pairs in overlapping:
                    merged.remove((existing_indices, existing_pairs))
                    indices.update(existing_indices)
                    direct_reply_pairs.update(existing_pairs)
            merged.append((indices, direct_reply_pairs))
            merged.sort(key=lambda item: min(item[0]))

        for indices, direct_reply_pairs in merged:
            drafts.append(
                EpisodeDraft(
                    canonical_name=canonical_name,
                    messages=tuple(messages[index] for index in sorted(indices)),
                    direct_exchange_count=len(direct_reply_pairs),
                    direct_reply_pairs=tuple(sorted(direct_reply_pairs)),
                )
            )

    return sorted(drafts, key=lambda draft: draft.started_at)


def _episode_fingerprint(draft: EpisodeDraft) -> str:
    payload = {
        "version": 1,
        "canonical_name": draft.canonical_name,
        "message_ids": list(draft.message_ids),
        "direct_exchange_count": draft.direct_exchange_count,
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _direct_reply_keys(draft: EpisodeDraft) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            f"v1:{draft.canonical_name}:{reply_id}:{parent_id}".encode("utf-8")
        ).hexdigest()
        for reply_id, parent_id in draft.direct_reply_pairs
    )


def _search_text(draft: EpisodeDraft) -> str:
    parts = []
    for message in draft.messages:
        if message.text:
            parts.append(message.text)
        if message.media_description:
            parts.append(message.media_description)
    return "\n".join(parts)


def _upsert_confirmed_users(store: MemoryStore) -> dict[str, int]:
    user_ids: dict[str, int] = {}
    for canonical_name in dict.fromkeys(ARCHIVE_TO_CANONICAL.values()):
        user_id = store.upsert_user(canonical_name)
        user_ids[canonical_name] = user_id
        for alias in archive_aliases_for(canonical_name):
            store.bind_archive_alias(user_id, alias)
    return user_ids


def _validated_summaries(
    requested: Sequence[SummaryInput], results: object
) -> dict[int, str]:
    if not isinstance(results, dict):
        raise ValueError("batch summary result must be a mapping")
    requested_ids = {item.episode_id for item in requested}
    if set(results) != requested_ids:
        raise ValueError("batch summary IDs must match requested IDs exactly")

    validated: dict[int, str] = {}
    for episode_id, summary in results.items():
        if not isinstance(episode_id, int) or isinstance(episode_id, bool):
            raise ValueError("batch summary IDs must be integers")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("batch summaries must be non-empty strings")
        validated[episode_id] = sanitize_source_for_model(summary.strip())
    return validated


async def import_archive(
    export_dir: Path,
    store: MemoryStore,
    summarizer: EpisodeSummarizer | None,
    summarize: bool,
    *,
    resume: bool = False,
) -> ImportStats:
    """Import one Telegram export and optionally summarize pending episodes.

    Every write phase is idempotent. A regular run processes pending episodes;
    ``resume=True`` additionally retries failed episodes. Ready episodes are
    never sent to the summarizer again.
    """

    if summarize and summarizer is None:
        raise ValueError("a summarizer is required when summarize=True")

    export_dir = Path(export_dir)
    store.initialize()
    user_ids = _upsert_confirmed_users(store)

    messages = parse_export(export_dir)
    message_row_ids: dict[int, int] = {}
    created_messages = 0
    for message in messages:
        canonical_name = canonical_from_archive(message.author)
        user_id = user_ids.get(canonical_name) if canonical_name else None
        row_id, created = store.store_message_with_status(
            MessageRecord(
                source="archive",
                chat_key=_ARCHIVE_CHAT_KEY,
                external_message_id=message.id,
                user_id=user_id,
                author_label=message.author or "",
                sent_at=message.sent_at,
                text=message.text,
                media_description=message.media_description,
                reply_to_external_id=message.reply_to,
            )
        )
        message_row_ids[message.id] = row_id
        created_messages += int(created)

    created_episodes = 0
    for draft in build_episode_drafts(messages):
        linked_rows = tuple(
            message_row_ids[message_id] for message_id in draft.message_ids
        )
        _, created = store.store_episode_with_status(
            user_id=user_ids[draft.canonical_name],
            source="archive",
            started_at=draft.started_at,
            ended_at=draft.ended_at,
            search_text=_search_text(draft),
            direct_exchange_count=draft.direct_exchange_count,
            message_row_ids=linked_rows,
            fingerprint=_episode_fingerprint(draft),
            overlap_keys=_direct_reply_keys(draft),
        )
        created_episodes += int(created)

    summary_requests = 0
    if summarize:
        statuses = ("pending", "failed") if resume else ("pending",)
        candidates = store.get_episodes_for_summary(statuses)
        for offset in range(0, len(candidates), _SUMMARY_BATCH_SIZE):
            batch = candidates[offset : offset + _SUMMARY_BATCH_SIZE]
            inputs = [
                SummaryInput(
                    episode_id=item.episode_id,
                    canonical_name=item.canonical_name,
                    messages=item.messages,
                )
                for item in batch
            ]
            summary_requests += 1
            try:
                generated = await summarizer.summarize_batch(inputs)
                validated = _validated_summaries(inputs, generated)
                store.mark_episodes_ready(validated)
            except Exception as exc:
                error_class = type(exc).__name__[:100]
                store.mark_episodes_failed(
                    tuple(item.episode_id for item in inputs), error_class
                )

        if callable(getattr(summarizer, "update_relationship_summary", None)):
            for canonical_name, user_id in user_ids.items():
                try:
                    await update_relationship_summary_blocks(
                        store,
                        summarizer,
                        user_id,
                        canonical_name,
                    )
                except Exception:
                    # The durable count is intentionally unchanged, so a later
                    # import/resume retries the same complete block.
                    continue

    counts = store.get_episode_status_counts()
    return ImportStats(
        parsed_messages=len(messages),
        created_messages=created_messages,
        created_episodes=created_episodes,
        ready_episodes=counts["ready"],
        pending_episodes=counts["pending"],
        failed_episodes=counts["failed"],
        summary_requests=summary_requests,
    )


def dry_run_archive(export_dir: Path) -> tuple[int, int, bool]:
    messages = parse_export(Path(export_dir))
    return (
        len(messages),
        len(build_episode_drafts(messages)),
        any(message.author == _BOT_ARCHIVE_AUTHOR for message in messages),
    )


def read_import_status(db_path: Path) -> dict[str, int]:
    path = Path(db_path).resolve()
    uri = f"{path.as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM episodes GROUP BY status"
        ).fetchall()
        potentially_ready_by_user = connection.execute(
            """
            SELECT user_id, COUNT(*)
            FROM episodes
            WHERE relationship_processed = 0
              AND status IN ('pending', 'failed', 'ready')
            GROUP BY user_id
            """
        ).fetchall()
    counts = {"pending": 0, "ready": 0, "failed": 0}
    for status, count in rows:
        if status in counts:
            counts[status] = int(count)
    counts["episode_summary_requests"] = math.ceil(
        (counts["pending"] + counts["failed"]) / _SUMMARY_BATCH_SIZE
    )
    counts["relationship_summary_requests"] = sum(
        int(count) // 10 for _, count in potentially_ready_by_user
    )
    counts["estimated_requests"] = (
        counts["episode_summary_requests"]
        + counts["relationship_summary_requests"]
    )
    return counts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import Telegram history into the bot memory database."
    )
    parser.add_argument("export_dir", type=Path, help="Telegram HTML export directory")
    parser.add_argument("--db", type=Path, default=Path("memory.db"))
    summary_mode = parser.add_mutually_exclusive_group()
    summary_mode.add_argument("--no-summarize", action="store_true")
    summary_mode.add_argument("--resume", action="store_true")
    inspection_mode = parser.add_mutually_exclusive_group()
    inspection_mode.add_argument("--dry-run", action="store_true")
    inspection_mode.add_argument("--status", action="store_true")
    parser.add_argument(
        "--allow-over-budget",
        action="store_true",
        help="allow more than 100 estimated external model calls",
    )
    return parser


def _print_stats(stats: ImportStats) -> None:
    for field_name in ImportStats.__dataclass_fields__:
        print(f"{field_name}={getattr(stats, field_name)}")


class _ImportEpisodeSummarizer(EpisodeSummarizer):
    """Use structured output for batches and normal text for relationship prose."""

    def __init__(
        self,
        generate_json: Callable[[str], Awaitable[str]],
        generate_plain_text: Callable[[str], Awaitable[str]],
    ):
        super().__init__(generate_json)
        self._generate_plain_text = generate_plain_text

    async def update_relationship_summary(
        self,
        canonical_name: str,
        previous_summary: str,
        episode_summaries: Sequence[str],
    ) -> str:
        prompt = build_relationship_summary_prompt(
            canonical_name, previous_summary, episode_summaries
        )
        result = (await self._generate_plain_text(prompt)).strip()
        if not result:
            raise ValueError("Gemini returned an empty relationship summary")
        return sanitize_source_for_model(result)


def _build_summarizer_from_environment() -> EpisodeSummarizer:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types as genai_types

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    client = genai.Client(api_key=api_key)

    async def generate_json(prompt: str) -> str:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        return (response.text or "").strip()

    async def generate_plain_text(prompt: str) -> str:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
        return (response.text or "").strip()

    return _ImportEpisodeSummarizer(generate_json, generate_plain_text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.status:
        if not args.db.is_file():
            parser.error("memory database does not exist")
        try:
            status = read_import_status(args.db)
        except sqlite3.Error:
            parser.error("memory database cannot be read")
        for name in (
            "pending",
            "ready",
            "failed",
            "episode_summary_requests",
            "relationship_summary_requests",
            "estimated_requests",
        ):
            print(f"{name}={status[name]}")
        return 0

    if not args.export_dir.is_dir():
        parser.error("export directory does not exist")

    if args.dry_run:
        parsed_messages, episode_drafts, bot_author_found = dry_run_archive(
            args.export_dir
        )
        print(f"parsed_messages={parsed_messages}")
        print(f"episode_drafts={episode_drafts}")
        print(f"confirmed_mappings={len(ARCHIVE_TO_CANONICAL)}")
        print(f"bot_author_found={int(bot_author_found)}")
        return 0

    store = MemoryStore(args.db)
    if args.no_summarize:
        stats = asyncio.run(
            import_archive(
                args.export_dir,
                store,
                None,
                summarize=False,
                resume=args.resume,
            )
        )
    else:
        # Import is local and idempotent. Do it before constructing a model
        # client so the complete request estimate can stop an expensive run.
        raw_stats = asyncio.run(
            import_archive(
                args.export_dir,
                store,
                None,
                summarize=False,
                resume=args.resume,
            )
        )
        status = read_import_status(args.db)
        estimated_requests = status["estimated_requests"]
        if (
            estimated_requests > _DEFAULT_EXTERNAL_CALL_BUDGET
            and not args.allow_over_budget
        ):
            parser.error(
                f"estimated external calls ({estimated_requests}) exceed "
                f"the {_DEFAULT_EXTERNAL_CALL_BUDGET}-call budget; inspect "
                "--status and pass --allow-over-budget to continue"
            )
        try:
            summarizer = _build_summarizer_from_environment()
        except RuntimeError as exc:
            parser.error(str(exc))
        summarized_stats = asyncio.run(
            import_archive(
                args.export_dir,
                store,
                summarizer,
                summarize=True,
                resume=args.resume,
            )
        )
        stats = ImportStats(
            parsed_messages=raw_stats.parsed_messages,
            created_messages=raw_stats.created_messages,
            created_episodes=raw_stats.created_episodes,
            ready_episodes=summarized_stats.ready_episodes,
            pending_episodes=summarized_stats.pending_episodes,
            failed_episodes=summarized_stats.failed_episodes,
            summary_requests=summarized_stats.summary_requests,
        )
    _print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
