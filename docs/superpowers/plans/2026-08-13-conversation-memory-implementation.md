# Conversation Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить Pampers Bot изолированную по пользователям долговременную память старых и новых разговоров, используемую только в одном разрешённом Telegram-чате.

**Architecture:** HTML-архив один раз импортируется в локальную SQLite-базу как сообщения и эпизоды общения. Во время ответа бот определяет канонического пользователя, получает только его сводку и до шести релевантных нейтральных описаний эпизодов, добавляет их в скрытую инструкцию Gemini и сохраняет новую беседу обратно в базу.

**Tech Stack:** Python 3.11, aiogram 3, google-genai, стандартные `sqlite3`, `html.parser`, `unittest`, `asyncio`.

## Global Constraints

- Старый экспорт: `E:\ЧАт\ChatExport_2026-08-13`.
- Автор `Памперс2004` — это Дима/Памперс; допустимые самоназвания бота: только «Памперс» и «Дима».
- Долговременная память читается и пополняется только в `ALLOWED_CHAT_ID`.
- Бот продолжает отвечать только на упоминание или Telegram Reply; остальные сообщения разрешённого чата только сохраняются.
- Память разных пользователей никогда не смешивается; основной ключ после привязки — числовой Telegram ID.
- В рабочий промпт попадают только нейтральные описания эпизодов, не сырые архивные реплики.
- В ответах нельзя раскрывать поиск по архиву, даты, ID и дословные цитаты старого чата.
- SQLite и временные базы остаются локальными и исключаются из `.gitignore`.
- В текущей папке нет Git-репозитория. Не выполнять `git init`; после каждого задания фиксировать список изменённых файлов и результаты тестов. Если пользователь позже подключит Git, каждое задание соответствует одному самостоятельному коммиту.

---

## File Map

- Create `identity.py`: нормализация и каноническое сопоставление участников.
- Create `archive_parser.py`: чтение Telegram HTML в типизированные сообщения.
- Create `memory.py`: схема SQLite, запись, поиск и получение контекста.
- Create `memory_summaries.py`: нейтральные сводки эпизодов и пользовательские сводки.
- Create `import_history.py`: повторяемый импорт и возобновление обработки.
- Modify `bot.py`: подключение памяти к существующему aiogram-обработчику.
- Modify `persona.json`: оставить только Памперса/Диму, удалить прежнее альтернативное самоназвание.
- Keep `requirements.txt` unchanged: SQLite и тесты использовать из стандартной библиотеки.
- Modify `.gitignore`: исключить рабочие и временные SQLite-файлы.
- Create `README.md`: команды инициализации, импорта, проверки и запуска.
- Create `tests/`: модульные и интеграционные тесты без реальных Telegram/Gemini-вызовов.

---

### Task 1: Canonical identities and persona names

**Files:**
- Create: `identity.py`
- Modify: `persona.json:1-40`
- Test: `tests/test_identity.py`
- Test: `tests/test_persona.py`

**Interfaces:**
- Produces: `normalize_alias(value: str) -> str`
- Produces: `canonical_from_archive(author: str | None) -> str | None`
- Produces: `canonical_from_telegram(username: str | None, display_name: str | None) -> str | None`
- Produces: `archive_aliases_for(canonical_name: str) -> tuple[str, ...]`

- [ ] **Step 1: Write failing identity tests**

```python
# tests/test_identity.py
import unittest

from identity import canonical_from_archive, canonical_from_telegram, normalize_alias


class IdentityTests(unittest.TestCase):
    def test_all_confirmed_archive_names(self):
        expected = {
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
        self.assertEqual({name: canonical_from_archive(name) for name in expected}, expected)

    def test_live_username_mapping(self):
        self.assertEqual(canonical_from_telegram("evilgeniusforever", "Неважно"), "Вовах")
        self.assertEqual(canonical_from_telegram(None, "Мария"), "Маша")

    def test_unknown_identity_is_not_guessed(self):
        self.assertIsNone(canonical_from_archive("Похожее имя"))
        self.assertIsNone(canonical_from_telegram("new_user", "Арина2"))

    def test_normalization_is_case_and_space_insensitive(self):
        self.assertEqual(normalize_alias("  V0VAH?  "), normalize_alias("v0vah?"))
```

- [ ] **Step 2: Write failing persona-name test**

```python
# tests/test_persona.py
import json
import unittest
from pathlib import Path


class PersonaTests(unittest.TestCase):
    def test_only_approved_persona_names_remain(self):
        persona = json.loads(Path("persona.json").read_text(encoding="utf-8"))
        serialized = json.dumps(persona, ensure_ascii=False).casefold()
        self.assertEqual(persona["persona_name"], "Памперс")
        self.assertEqual(persona["persona_aliases"], ["Дима"])
        self.assertNotIn("аллан", serialized)
        self.assertNotIn("allan", serialized)
```

- [ ] **Step 3: Run the tests and verify the missing module/schema failures**

Run: `venv\Scripts\python.exe -m unittest tests.test_identity tests.test_persona -v`

Expected: `ModuleNotFoundError: No module named 'identity'` and/or failure for missing `persona_aliases`.

- [ ] **Step 4: Implement the exact identity maps and normalization**

```python
# identity.py
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


_ARCHIVE_NORMALIZED = {normalize_alias(key): value for key, value in ARCHIVE_TO_CANONICAL.items()}


def canonical_from_archive(author: str | None) -> str | None:
    return _ARCHIVE_NORMALIZED.get(normalize_alias(author or ""))


def canonical_from_telegram(username: str | None, display_name: str | None) -> str | None:
    by_username = USERNAME_TO_CANONICAL.get(normalize_alias(username or ""))
    if by_username:
        return by_username
    return DISPLAY_TO_CANONICAL.get(normalize_alias(display_name or ""))


def archive_aliases_for(canonical_name: str) -> tuple[str, ...]:
    return tuple(key for key, value in ARCHIVE_TO_CANONICAL.items() if value == canonical_name)
```

- [ ] **Step 5: Correct `persona.json`**

Set `"persona_name": "Памперс"`, add `"persona_aliases": ["Дима"]`, and remove the quirk that permits the old alternative self-name. Do not change unrelated speech traits.

- [ ] **Step 6: Run identity and persona tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_identity tests.test_persona -v`

Expected: all tests pass.

- [ ] **Step 7: Record checkpoint**

Record changed files `identity.py`, `persona.json`, `tests/test_identity.py`, `tests/test_persona.py` and the passing test command in the implementation log.

---

### Task 2: Persistent SQLite store and strict user binding

**Files:**
- Create: `memory.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Consumes: `normalize_alias()` from `identity.py`
- Produces: `MessageRecord`, `EpisodeRecord`, `MemoryContext`
- Produces: `MemoryStore.initialize() -> None`
- Produces: `MemoryStore.upsert_user(...) -> int`
- Produces: `MemoryStore.bind_telegram_identity(...) -> int`
- Produces: `MemoryStore.store_message(record: MessageRecord) -> int`
- Produces: `MemoryStore.get_recent_messages(chat_key: str, limit: int) -> list[MessageRecord]`

- [ ] **Step 1: Write failing persistence, deduplication, and binding tests**

```python
# tests/test_memory_store.py
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from memory import MemoryStore, MessageRecord


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_message_is_persistent_and_idempotent(self):
        user_id = self.store.upsert_user("Вовах")
        record = MessageRecord(
            source="archive", chat_key="old-chat", external_message_id=42,
            user_id=user_id, author_label="V0VAH?",
            sent_at=datetime(2026, 8, 1, tzinfo=timezone.utc), text="Привет",
            media_description="", reply_to_external_id=None,
        )
        first = self.store.store_message(record)
        second = self.store.store_message(record)
        self.assertEqual(first, second)
        reopened = MemoryStore(self.db_path)
        reopened.initialize()
        self.assertEqual(len(reopened.get_recent_messages("old-chat", 10)), 1)

    def test_two_telegram_ids_cannot_bind_to_one_known_user(self):
        user_id = self.store.upsert_user("Вовах")
        self.store.bind_telegram_identity(user_id, 1001, "evilgeniusforever", "Вова")
        with self.assertRaises(ValueError):
            self.store.bind_telegram_identity(user_id, 2002, "other", "Другой")
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `venv\Scripts\python.exe -m unittest tests.test_memory_store -v`

Expected: import failure because `memory.py` does not exist.

- [ ] **Step 3: Implement dataclasses and schema**

Define these public types exactly:

```python
@dataclass(frozen=True)
class MessageRecord:
    source: str
    chat_key: str
    external_message_id: int
    user_id: int | None
    author_label: str
    sent_at: datetime
    text: str
    media_description: str
    reply_to_external_id: int | None


@dataclass(frozen=True)
class EpisodeRecord:
    id: int
    user_id: int
    summary: str
    started_at: datetime
    ended_at: datetime
    direct_exchange_count: int


@dataclass(frozen=True)
class MemoryContext:
    canonical_name: str
    relationship_summary: str
    episodes: tuple[EpisodeRecord, ...]
    recent_messages: tuple[MessageRecord, ...]
```

Implement schema version `1` with the `users`, `aliases`, `messages`, `episodes`, `episode_messages`, `user_summaries`, and `schema_meta` tables from the approved design. Open short-lived connections with `PRAGMA foreign_keys=ON`, `PRAGMA journal_mode=WAL`, `timeout=5`, and explicit transactions.

- [ ] **Step 4: Implement idempotent writes and strict binding**

`store_message()` must return the existing row ID after a uniqueness conflict. `bind_telegram_identity()` must reject binding a canonical user already bound to a different numeric ID and reject reusing one numeric ID for another canonical user.

- [ ] **Step 5: Run the persistence tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_memory_store -v`

Expected: all tests pass and the temporary database can be reopened.

- [ ] **Step 6: Record checkpoint**

Record `memory.py`, `tests/test_memory_store.py`, the schema version, and the passing test command.

---

### Task 3: Telegram HTML parser and episode extraction

**Files:**
- Create: `archive_parser.py`
- Create: `import_history.py`
- Test: `tests/test_archive_parser.py`
- Test: `tests/test_episode_extraction.py`
- Test fixture: `tests/fixtures/telegram_export/messages.html`

**Interfaces:**
- Consumes: `canonical_from_archive()` from `identity.py`
- Produces: `ArchiveMessage`
- Produces: `parse_export(export_dir: Path) -> list[ArchiveMessage]`
- Produces: `EpisodeDraft`
- Produces: `build_episode_drafts(messages: Sequence[ArchiveMessage], window_size: int = 3, max_gap_minutes: int = 15) -> list[EpisodeDraft]`

- [ ] **Step 1: Create a minimal Telegram export fixture**

The fixture must contain five normal messages with IDs `1..5`: a user message from `V0VAH?`, a direct reply by `Памперс2004`, one neighboring context message before and after, and an unrelated message more than 15 minutes later. Use Telegram's `message default`, `from_name`, `date`, `text`, and `reply_to` HTML classes so the test exercises the real parser shape.

- [ ] **Step 2: Write failing parser and episode tests**

```python
# tests/test_episode_extraction.py
import unittest
from datetime import datetime, timezone

from archive_parser import ArchiveMessage
from import_history import build_episode_drafts


def msg(mid, author, minute, text, reply_to=None):
    return ArchiveMessage(
        id=mid, author=author,
        sent_at=datetime(2026, 8, 1, 12, minute, tzinfo=timezone.utc),
        text=text, media_description="", reply_to=reply_to, page="messages.html",
    )


class EpisodeExtractionTests(unittest.TestCase):
    def test_direct_reply_gets_three_message_window_but_not_late_message(self):
        messages = [
            msg(1, "V0VAH?", 0, "контекст"),
            msg(2, "V0VAH?", 1, "вопрос"),
            msg(3, "Памперс2004", 2, "ответ", reply_to=2),
            msg(4, "V0VAH?", 3, "продолжение"),
            msg(5, "V0VAH?", 30, "другая тема"),
        ]
        episodes = build_episode_drafts(messages)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].canonical_name, "Вовах")
        self.assertEqual(episodes[0].message_ids, (1, 2, 3, 4))

    def test_overlapping_reply_windows_merge(self):
        messages = [
            msg(1, "V0VAH?", 0, "один"),
            msg(2, "Памперс2004", 1, "два", reply_to=1),
            msg(3, "V0VAH?", 2, "три", reply_to=2),
        ]
        episodes = build_episode_drafts(messages)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].direct_exchange_count, 2)
```

- [ ] **Step 3: Run tests and verify missing parser/extractor failures**

Run: `venv\Scripts\python.exe -m unittest tests.test_archive_parser tests.test_episode_extraction -v`

Expected: imports fail because the two modules do not yet exist.

- [ ] **Step 4: Implement `archive_parser.py`**

Adapt only the generic parsing behavior from `E:\ЧАт\extract_chat.py`: parse message ID, inherited author for joined messages, date, text, media paths, and Reply target. Do not copy the insult classifiers or profile-specific regular expressions.

Use this public dataclass:

```python
@dataclass(frozen=True)
class ArchiveMessage:
    id: int
    author: str | None
    sent_at: datetime
    text: str
    media_description: str
    reply_to: int | None
    page: str
```

- [ ] **Step 5: Implement deterministic episode extraction**

Use exact-author equality for `Памперс2004`. A direct anchor exists only when one side of a Reply is `Памперс2004` and the other side resolves through `canonical_from_archive()`. Add up to three indices before and after the anchor only while each neighboring message is within 15 minutes of the anchor. Merge overlapping windows for the same canonical user and sort final drafts chronologically.

- [ ] **Step 6: Run parser and extractor tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_archive_parser tests.test_episode_extraction -v`

Expected: all tests pass.

- [ ] **Step 7: Record checkpoint**

Record both modules, the fixture, both test modules, and the passing command.

---

### Task 4: Neutral summaries and model-facing privacy boundary

**Files:**
- Create: `memory_summaries.py`
- Test: `tests/test_memory_summaries.py`

**Interfaces:**
- Consumes: sanitized text derived from `ArchiveMessage`
- Produces: `sanitize_source_for_model(text: str) -> str`
- Produces: `build_episode_summary_prompt(canonical_name: str, messages: Sequence[str]) -> str`
- Produces: `build_relationship_summary_prompt(canonical_name: str, previous_summary: str, episode_summaries: Sequence[str]) -> str`
- Produces: `EpisodeSummarizer.summarize_episode(...) -> str`
- Produces: `EpisodeSummarizer.summarize_batch(items: Sequence[SummaryInput], batch_size: int = 20) -> dict[int, str]`

- [ ] **Step 1: Write failing privacy and prompt tests**

```python
# tests/test_memory_summaries.py
import unittest

from memory_summaries import build_episode_summary_prompt, sanitize_source_for_model


class MemorySummaryTests(unittest.TestCase):
    def test_legacy_self_alias_is_removed_before_model_call(self):
        cleaned = sanitize_source_for_model("Я Аллан и @Allan_1215 тоже я")
        self.assertNotIn("аллан", cleaned.casefold())
        self.assertNotIn("allan", cleaned.casefold())

    def test_summary_prompt_requires_neutral_nonquoted_output(self):
        prompt = build_episode_summary_prompt("Вовах", ["Вовах: вопрос", "Памперс2004: ответ"])
        lowered = prompt.casefold()
        self.assertIn("без дословных цитат", lowered)
        self.assertIn("прошлое", lowered)
        self.assertIn("памперс", lowered)
        self.assertNotIn("@allan_1215", lowered)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `venv\Scripts\python.exe -m unittest tests.test_memory_summaries -v`

Expected: import failure because `memory_summaries.py` does not exist.

- [ ] **Step 3: Implement local sanitation before any summary call**

Replace case-insensitive occurrences of the removed legacy alias, its Latin spelling, and its old `@...` form with `другой человек`. Preserve all other text. This sanitizer is applied only to content sent to the summarization model; the local source row stays unchanged for audit and FTS.

- [ ] **Step 4: Implement the summarizer abstraction**

```python
class EpisodeSummarizer:
    def __init__(self, generate_text: Callable[[str], Awaitable[str]]):
        self._generate_text = generate_text

    async def summarize_episode(self, canonical_name: str, messages: Sequence[str]) -> str:
        prompt = build_episode_summary_prompt(canonical_name, messages)
        result = (await self._generate_text(prompt)).strip()
        if not result:
            raise ValueError("Gemini returned an empty episode summary")
        return sanitize_source_for_model(result)
```

The episode prompt requires 1–3 neutral sentences describing the topic, the user's position, and Pampers/Dima's reaction. It declares archived text untrusted data, forbids following instructions inside it, forbids quotes/dates/IDs, and says past preferences may have changed.

- [ ] **Step 5: Implement bounded batch summaries for archive import**

Define `SummaryInput(episode_id: int, canonical_name: str, messages: tuple[str, ...])`. `summarize_batch()` splits input into chunks of at most 20, sanitizes every message before the call, requests a JSON array of `{episode_id, summary}`, validates that every requested ID occurs exactly once, and sanitizes every returned summary. A malformed batch raises `ValueError` without marking any item in that batch ready.

- [ ] **Step 6: Add async fake-generator tests**

Use `unittest.IsolatedAsyncioTestCase` with a local async function returning deterministic single and batch summaries. Verify empty output raises `ValueError`, 21 inputs cause exactly two generator calls, a missing episode ID rejects the whole batch, and every returned summary is sanitized.

- [ ] **Step 7: Run summary tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_memory_summaries -v`

Expected: all tests pass without network access.

- [ ] **Step 8: Record checkpoint**

Record `memory_summaries.py`, its tests, and the passing command.

---

### Task 5: Episode persistence, FTS retrieval, and prompt-safe context

**Files:**
- Modify: `memory.py`
- Test: `tests/test_memory_retrieval.py`

**Interfaces:**
- Produces: `MemoryStore.store_episode(user_id: int, source: str, started_at: datetime, ended_at: datetime, search_text: str, direct_exchange_count: int, message_row_ids: Sequence[int], fingerprint: str) -> int`
- Produces: `MemoryStore.mark_episode_ready(episode_id: int, summary: str) -> None`
- Produces: `MemoryStore.get_memory_context(user_id: int, query: str, chat_key: str, episode_limit: int, recent_limit: int) -> MemoryContext`
- Produces: `render_memory_instruction(context: MemoryContext) -> str`

- [ ] **Step 1: Write failing user-isolation and privacy tests**

```python
# tests/test_memory_retrieval.py
class MemoryRetrievalTests(unittest.TestCase):
    def test_search_never_returns_another_users_episode(self):
        vovah = self.store.upsert_user("Вовах")
        maria = self.store.upsert_user("Маша")
        self.add_ready_episode(vovah, "обсуждали переезд и язык", "переезд язык")
        self.add_ready_episode(maria, "обсуждали переезд и работу", "переезд работа")
        context = self.store.get_memory_context(vovah, "переезд", "live:1", 6, 6)
        self.assertEqual([item.user_id for item in context.episodes], [vovah])

    def test_pending_episode_is_not_model_context(self):
        user_id = self.store.upsert_user("Вовах")
        self.add_pending_episode(user_id, "СЫРАЯ СЕКРЕТНАЯ РЕПЛИКА")
        context = self.store.get_memory_context(user_id, "секретная", "live:1", 6, 6)
        instruction = render_memory_instruction(context)
        self.assertNotIn("СЫРАЯ СЕКРЕТНАЯ РЕПЛИКА", instruction)
```

- [ ] **Step 2: Run tests and verify missing method failures**

Run: `venv\Scripts\python.exe -m unittest tests.test_memory_retrieval -v`

Expected: `AttributeError` for retrieval methods.

- [ ] **Step 3: Implement episode writes and FTS5 feature detection**

Create `episode_fts` with `summary` and `search_text` when FTS5 is available. Keep a boolean capability in `schema_meta`. When FTS creation fails, continue with the core schema and use normalized token overlap in Python over the latest 100 ready episodes for that user.

- [ ] **Step 4: Implement isolated ranking**

Every query includes `WHERE episodes.user_id = ? AND episodes.status = 'ready'`. Rank FTS candidates by BM25, then direct exchange count, then `ended_at DESC`. Remove repeated summaries by normalized text. Return no more than `episode_limit`. With no lexical candidate, return the latest three ready episodes for that same user.

- [ ] **Step 5: Implement safe instruction rendering**

`render_memory_instruction()` includes only `canonical_name`, `relationship_summary`, and `EpisodeRecord.summary`; it never includes `search_text`, raw `MessageRecord.text`, dates, message IDs, database paths, or another user's data. The instruction labels memory as past observations and explicitly bans quotation or disclosure of the memory mechanism.

- [ ] **Step 6: Add fallback, recency, and six-item limit tests**

Insert eight ready episodes for one user, query for a shared token, and assert exactly six are returned. Disable FTS in the test database and assert fallback still returns only the requested user's episodes.

- [ ] **Step 7: Run storage and retrieval tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_memory_store tests.test_memory_retrieval -v`

Expected: all tests pass.

- [ ] **Step 8: Record checkpoint**

Record the schema change, `memory.py`, the retrieval tests, and the passing command.

---

### Task 6: Idempotent archive importer with resume support

**Files:**
- Modify: `import_history.py`
- Modify: `memory.py`
- Test: `tests/test_import_history.py`

**Interfaces:**
- Consumes: `parse_export()`, `build_episode_drafts()`, `MemoryStore`, `EpisodeSummarizer`
- Produces: `ImportStats`
- Produces: `import_archive(export_dir: Path, store: MemoryStore, summarizer: EpisodeSummarizer | None, summarize: bool) -> ImportStats`
- CLI: `python import_history.py EXPORT_DIR --db PATH [--no-summarize] [--resume] [--dry-run] [--status]`

`ImportStats` is defined exactly as:

```python
@dataclass(frozen=True)
class ImportStats:
    parsed_messages: int
    created_messages: int
    created_episodes: int
    ready_episodes: int
    pending_episodes: int
    failed_episodes: int
    summary_requests: int
```

- [ ] **Step 1: Write a failing repeat-import test**

```python
# tests/test_import_history.py
class ImportHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_import_does_not_duplicate_rows_or_summaries(self):
        first = await import_archive(self.fixture_dir, self.store, self.summarizer, summarize=True)
        second = await import_archive(self.fixture_dir, self.store, self.summarizer, summarize=True)
        self.assertEqual(first.created_messages, 5)
        self.assertEqual(second.created_messages, 0)
        self.assertEqual(second.created_episodes, 0)
        self.assertEqual(self.generator_calls, 1)
```

- [ ] **Step 2: Run the importer test and verify it fails**

Run: `venv\Scripts\python.exe -m unittest tests.test_import_history -v`

Expected: missing `import_archive`/`ImportStats` failure.

- [ ] **Step 3: Implement import phases**

Phase order is fixed: initialize schema; upsert ten canonical users and aliases; persist parsed messages; build and persist episode drafts; summarize `pending` episodes in batches of at most 20; update a whole validated batch to `ready`; print aggregate counts. Use the episode fingerprint to skip overlapping imports.

- [ ] **Step 4: Implement safe resumability**

`--resume` selects existing `pending` and `failed` episodes, retries their summaries in 20-item batches, and never resends `ready` episodes. A failed batch writes `status='failed'` plus a short exception class name without source text. `--no-summarize` imports rows as `pending`. `--dry-run` parses and reports aggregate counts without opening a writable database or calling Gemini. `--status` opens the database read-only and prints only `pending`, `ready`, `failed`, and the estimated request count `ceil((pending + failed) / 20)`.

- [ ] **Step 5: Add CLI parsing tests**

Test defaults, explicit Windows paths with Cyrillic characters, `--dry-run`, and rejection of a nonexistent export directory with exit code `2` and a concise message that does not expose `.env` values.

- [ ] **Step 6: Run importer tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_import_history -v`

Expected: all tests pass without live Gemini calls.

- [ ] **Step 7: Run a full-export dry run**

Run: `venv\Scripts\python.exe import_history.py "E:\ЧАт\ChatExport_2026-08-13" --db "memory.test.db" --dry-run`

Expected: 27,485 parsed messages, author `Памперс2004` found, ten confirmed mappings loaded, and a nonzero episode count. The command prints counts only.

- [ ] **Step 8: Record checkpoint**

Record importer files, the dry-run counts, and the passing commands.

---

### Task 7: Integrate memory into the aiogram message path

**Files:**
- Modify: `bot.py:24-32,94-228,248-377,409-419`
- Test: `tests/test_bot_memory.py`

**Interfaces:**
- Consumes: `canonical_from_telegram()`, `MemoryStore`, `MemoryContext`, `render_memory_instruction()`, `EpisodeSummarizer`
- Produces: `parse_allowed_chat_id(value: str | None) -> int | None`
- Produces: `memory_enabled_for(chat_id: int) -> bool`
- Produces: `persist_incoming_message(message: Message) -> tuple[int | None, int | None]`
- Changes: `deliver(...) -> Message | None`

- [ ] **Step 1: Write failing chat-boundary and response-rule tests**

Use `unittest.IsolatedAsyncioTestCase`, `unittest.mock.AsyncMock`, and small fake objects with the fields used by the handler. Assert:

```python
async def test_unaddressed_message_in_allowed_chat_is_saved_but_not_answered(self):
    await handle_message(self.fake_message(chat_id=100, text="обычная реплика", mentioned=False))
    self.assertEqual(self.store.saved_incoming_count, 1)
    self.gemini.assert_not_awaited()

async def test_other_chat_uses_no_long_term_memory(self):
    await handle_message(self.fake_message(chat_id=200, text="@bot привет", mentioned=True))
    self.store.get_memory_context.assert_not_called()

async def test_memory_instruction_contains_only_current_user(self):
    await handle_message(self.fake_message(chat_id=100, user_id=55, username="evilgeniusforever", text="@bot переезд", mentioned=True))
    instruction = self.gemini.await_args.args[1]
    self.assertIn("Вовах", instruction)
    self.assertNotIn("Маша", instruction)
```

- [ ] **Step 2: Run the bot-memory tests and verify failures**

Run: `venv\Scripts\python.exe -m unittest tests.test_bot_memory -v`

Expected: missing configuration/helper failures.

- [ ] **Step 3: Add fail-closed configuration**

Parse `ALLOWED_CHAT_ID` once. Missing, empty, or nonnumeric values yield `None`, log one warning, and disable long-term memory everywhere. Initialize `MemoryStore` only when a valid ID exists. A database initialization exception logs only its class and disables memory for the process.

- [ ] **Step 4: Save every eligible incoming message before `should_respond()`**

For the allowed group, resolve the canonical user, safely bind Telegram ID, and persist text/caption/media metadata using `source='live'`, `chat_key=f'live:{chat_id}'`, and the Telegram message ID. Unknown users get canonical key `telegram:{telegram_user_id}` plus their current display name, never an archive alias.

- [ ] **Step 5: Retrieve memory only for addressed messages**

After `should_respond()` succeeds, fetch recent local context and the current user's memory using `await asyncio.to_thread(...)`. Append `render_memory_instruction(context)` to a per-request system instruction; do not mutate the global `SYSTEM_PROMPT`.

- [ ] **Step 6: Preserve output behavior and capture the sent Telegram message**

Change `deliver()` to return the `Message` from `reply`, `reply_voice`, `send_message`, or `send_voice`. Sticker-only responses return the sticker message but create no textual memory episode. Preserve existing probabilities, cleanup, skip sentinel, private-chat rejection, and spontaneous message behavior.

- [ ] **Step 7: Persist successful bot replies and queue summarization**

Only after Telegram confirms delivery, save the textual response with the returned message ID and Reply target. Create a live episode from the input, output, and nearest persisted context. Track background summary tasks in a `set[asyncio.Task]`; remove completed tasks with `add_done_callback`. Failures mark the episode pending/failed without affecting the sent reply.

- [ ] **Step 8: Enforce approved persona names in the generated system prompt**

Update `build_system_prompt()` so it states: `Ты Памперс, также тебя называют Дима.` It must not load any other self-name from the persona. Add an assertion in `tests/test_bot_memory.py` that the case-folded prompt contains both approved names and excludes the removed legacy name in Cyrillic and Latin spelling.

- [ ] **Step 9: Run bot integration tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_bot_memory -v`

Expected: all tests pass with no network access.

- [ ] **Step 10: Run the full unit suite**

Run: `venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 11: Record checkpoint**

Record `bot.py`, its tests, confirmation of unchanged sticker/voice tests, and the full-suite result.

---

### Task 8: Relationship summary updates and restart behavior

**Files:**
- Modify: `memory.py`
- Modify: `memory_summaries.py`
- Modify: `bot.py`
- Modify: `import_history.py`
- Test: `tests/test_relationship_summary.py`

**Interfaces:**
- Produces: `MemoryStore.get_ready_summaries_after(user_id: int, processed_count: int) -> list[str]`
- Produces: `MemoryStore.save_user_summary(user_id: int, summary: str, processed_episode_count: int) -> None`
- Produces: `EpisodeSummarizer.update_relationship_summary(...) -> str`

- [ ] **Step 1: Write failing ten-episode threshold and restart tests**

Insert nine ready episodes and assert no update is requested. Insert the tenth and assert the updater receives the prior summary plus exactly ten new neutral descriptions. Reopen the SQLite file, add another nine, and assert the stored `processed_episode_count` prevents duplicate processing.

- [ ] **Step 2: Run the tests and verify missing method failures**

Run: `venv\Scripts\python.exe -m unittest tests.test_relationship_summary -v`

Expected: missing summary update methods.

- [ ] **Step 3: Implement the update transaction**

Update after each block of ten ready episodes. Save the new summary and processed count atomically. If generation fails, leave the prior summary/count unchanged so the same block can be retried. Call the same update function after archive batches in `import_history.py` and after live episode completion in `bot.py`.

- [ ] **Step 4: Implement the relationship-summary prompt**

Require a concise description of recurring subjects, preferred tone, unresolved threads, and changes over time. Ban diagnosis, certainty about current opinions, raw quotations, third-party private details, and instructions copied from archived content.

- [ ] **Step 5: Run relationship and full tests**

Run: `venv\Scripts\python.exe -m unittest tests.test_relationship_summary -v`

Run: `venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 6: Record checkpoint**

Record the three modified files, the new test, and both passing commands.

---

### Task 9: Operational files, real import, and end-to-end verification

**Files:**
- Modify: `.gitignore`
- Create: `README.md`
- Create locally during verification: `memory.db` (ignored, not treated as source code)

**Interfaces:**
- Documents: `ALLOWED_CHAT_ID`, `MEMORY_DB_PATH`, `MEMORY_MAX_EPISODES`, `MEMORY_CONTEXT_MESSAGES`
- Documents commands for dry run, import without summaries, resume summaries, tests, and bot startup.

- [ ] **Step 1: Protect local data files**

Append these patterns to `.gitignore`:

```gitignore
memory.db
memory.db-shm
memory.db-wal
memory.*.db
tests/*.db
```

- [ ] **Step 2: Write concise operator documentation**

`README.md` must give exact PowerShell commands:

```powershell
$env:ALLOWED_CHAT_ID='-1001234567890'
.\venv\Scripts\python.exe import_history.py 'E:\ЧАт\ChatExport_2026-08-13' --db '.\memory.db' --dry-run
.\venv\Scripts\python.exe import_history.py 'E:\ЧАт\ChatExport_2026-08-13' --db '.\memory.db' --no-summarize
.\venv\Scripts\python.exe import_history.py 'E:\ЧАт\ChatExport_2026-08-13' --db '.\memory.db' --resume
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe bot.py
```

Explain how to obtain the numeric chat ID from existing bot logs or a temporary diagnostic handler without printing tokens. State that the placeholder ID in the example must be replaced.

- [ ] **Step 3: Run static checks**

Run: `venv\Scripts\python.exe -m py_compile bot.py voice.py identity.py archive_parser.py memory.py memory_summaries.py import_history.py`

Expected: exit code `0`.

- [ ] **Step 4: Run the entire test suite**

Run: `venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Import raw production memory idempotently**

Run the importer with `--no-summarize` into `memory.db`, record only aggregate counts, then run the same command again and verify the second run reports zero newly created messages and episodes.

- [ ] **Step 6: Measure summary workload before external calls**

Run a read-only importer status command that prints the number of `pending`, `ready`, and `failed` episodes. If the implementation estimates more than 100 Gemini requests after batching, stop before the external calls and report the count; otherwise run `--resume` and verify all processable episodes become `ready`. This is the only cost guardrail and does not alter the imported raw memory.

- [ ] **Step 7: Verify two-user isolation with a fake Gemini client**

Use a local script or integration test that creates contexts for Вовах and Маша from `memory.db`, records the canonical name and episode IDs only, and asserts their episode ID sets are disjoint. Do not print summaries or raw texts.

- [ ] **Step 8: Perform a bounded Telegram smoke test**

Start the bot with the configured `ALLOWED_CHAT_ID`, send one mention from a mapped test account, and verify one response is delivered and both incoming/outgoing rows appear in SQLite. Send an unaddressed message and verify it is stored without a bot response. Do not send spontaneous messages solely for testing.

- [ ] **Step 9: Final regression and handoff**

Re-run compilation and all tests, report the imported aggregate counts, database path, whether summaries are ready, and the exact environment variable still required from the user. Do not include tokens, full prompts, summaries, or chat text in the handoff.

---

## Completion Criteria

- The full old export can be imported repeatedly without duplicates.
- Each mapped Telegram user receives only their own archived and live memory.
- Unaddressed messages in the allowed chat are remembered but do not trigger answers.
- Other chats never read or write long-term memory.
- Restarting the bot preserves new conversations and relationship summaries.
- The model receives neutral descriptions, not raw archive excerpts.
- The bot calls itself only «Памперс» or «Дима».
- All unit/integration tests pass and the existing sticker, voice, private-chat, mention, Reply, and spontaneous-message behavior remains functional.
