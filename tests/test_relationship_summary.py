import sqlite3
import tempfile
import unittest
import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bot
import memory_summaries
from import_history import import_archive
from memory import MemoryStore


class RecordingSummarizer:
    def __init__(self, *, fail_relationship: bool = False):
        self.fail_relationship = fail_relationship
        self.relationship_calls = []

    async def summarize_batch(self, items):
        return {
            item.episode_id: f"нейтральная сводка {item.episode_id}"
            for item in items
        }

    async def summarize_episode(self, canonical_name, messages):
        return "нейтральная сводка живого эпизода"

    async def update_relationship_summary(
        self, canonical_name, previous_summary, episode_summaries
    ):
        self.relationship_calls.append(
            (canonical_name, previous_summary, tuple(episode_summaries))
        )
        if self.fail_relationship:
            raise RuntimeError("relationship model unavailable")
        return f"сводка отношений {len(self.relationship_calls)}"


class RelationshipSummaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()
        self.user_id = self.store.upsert_user("Вовах")

    def tearDown(self):
        self.tempdir.cleanup()

    def add_ready_episode(self, number: int, summary: str | None = None) -> int:
        when = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(
            minutes=number
        )
        episode_id = self.store.store_episode(
            user_id=self.user_id,
            source="archive",
            started_at=when,
            ended_at=when,
            search_text=f"тема {number}",
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint=f"relationship-{number}",
        )
        self.store.mark_episode_ready(
            episode_id, summary or f"эпизод {number}"
        )
        return episode_id

    def seed_summary(self, summary: str, processed_count: int) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO user_summaries"
                "(user_id, summary, processed_episode_count, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    self.user_id,
                    summary,
                    processed_count,
                    "2026-08-01T00:00:00.000000Z",
                ),
            )
            connection.commit()

    def read_summary(self) -> tuple[str, int] | None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            return connection.execute(
                "SELECT summary, processed_episode_count "
                "FROM user_summaries WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()

    async def run_updates(self, summarizer) -> int:
        updater = getattr(
            memory_summaries, "update_relationship_summary_blocks", None
        )
        self.assertIsNotNone(updater)
        return await updater(
            self.store, summarizer, self.user_id, "Вовах"
        )

    async def test_nine_ready_episodes_do_not_request_an_update(self):
        for number in range(1, 10):
            self.add_ready_episode(number)
        summarizer = RecordingSummarizer()

        updates = await self.run_updates(summarizer)

        self.assertEqual(updates, 0)
        self.assertEqual(summarizer.relationship_calls, [])
        self.assertIsNone(self.read_summary())

    async def test_tenth_episode_uses_prior_summary_and_exact_next_ten(self):
        self.seed_summary("прежняя сводка", 0)
        for number in range(1, 11):
            self.add_ready_episode(number)
        summarizer = RecordingSummarizer()

        updates = await self.run_updates(summarizer)

        self.assertEqual(updates, 1)
        self.assertEqual(
            summarizer.relationship_calls,
            [
                (
                    "Вовах",
                    "прежняя сводка",
                    tuple(f"эпизод {number}" for number in range(1, 11)),
                )
            ],
        )
        self.assertEqual(self.read_summary(), ("сводка отношений 1", 10))

    async def test_restart_resumes_after_processed_count_without_duplicates(self):
        for number in range(1, 11):
            self.add_ready_episode(number)
        first_summarizer = RecordingSummarizer()
        await self.run_updates(first_summarizer)
        self.assertEqual(self.read_summary(), ("сводка отношений 1", 10))

        self.store = MemoryStore(self.db_path)
        for number in range(11, 20):
            self.add_ready_episode(number)
        restarted_summarizer = RecordingSummarizer()

        self.assertEqual(await self.run_updates(restarted_summarizer), 0)
        self.assertEqual(restarted_summarizer.relationship_calls, [])
        self.assertEqual(self.read_summary(), ("сводка отношений 1", 10))

        self.add_ready_episode(20)
        self.assertEqual(await self.run_updates(restarted_summarizer), 1)
        self.assertEqual(
            restarted_summarizer.relationship_calls[0][2],
            tuple(f"эпизод {number}" for number in range(11, 21)),
        )
        self.assertEqual(self.read_summary(), ("сводка отношений 1", 20))

    async def test_late_ready_episode_is_processed_once_without_skipping(self):
        late_episode_id = self.store.store_episode(
            user_id=self.user_id,
            source="archive",
            started_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
            ended_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
            search_text="поздняя тема",
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint="relationship-late",
        )
        for number in range(2, 12):
            self.add_ready_episode(number)
        summarizer = RecordingSummarizer()
        self.assertEqual(await self.run_updates(summarizer), 1)

        self.store.mark_episode_ready(late_episode_id, "поздний эпизод 1")
        for number in range(12, 21):
            self.add_ready_episode(number)

        self.assertEqual(await self.run_updates(summarizer), 1)
        self.assertEqual(
            summarizer.relationship_calls[1][2],
            ("поздний эпизод 1",)
            + tuple(f"эпизод {number}" for number in range(12, 21)),
        )

    async def test_revised_processed_episode_is_included_again_after_ready(self):
        when = datetime(2026, 8, 1, tzinfo=timezone.utc)
        revised_episode_id, created = self.store.store_episode_with_status(
            user_id=self.user_id,
            source="archive",
            started_at=when,
            ended_at=when,
            search_text="первая версия",
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint="relationship-revised-v1",
            overlap_keys=("shared-anchor",),
        )
        self.assertTrue(created)
        self.store.mark_episode_ready(revised_episode_id, "эпизод 1")
        for number in range(2, 11):
            self.add_ready_episode(number)
        summarizer = RecordingSummarizer()
        self.assertEqual(await self.run_updates(summarizer), 1)

        same_episode_id, created = self.store.store_episode_with_status(
            user_id=self.user_id,
            source="archive",
            started_at=when,
            ended_at=when + timedelta(minutes=1),
            search_text="вторая версия",
            direct_exchange_count=2,
            message_row_ids=(),
            fingerprint="relationship-revised-v2",
            overlap_keys=("shared-anchor", "new-anchor"),
        )
        self.assertFalse(created)
        self.assertEqual(same_episode_id, revised_episode_id)
        self.store.mark_episode_ready(revised_episode_id, "обновлённый эпизод 1")
        for number in range(11, 20):
            self.add_ready_episode(number)

        self.assertEqual(await self.run_updates(summarizer), 1)
        self.assertEqual(
            summarizer.relationship_calls[1][2],
            ("обновлённый эпизод 1",)
            + tuple(f"эпизод {number}" for number in range(11, 20)),
        )

    async def test_repeated_ready_is_idempotent_but_changed_summary_is_reprocessed(self):
        episode_ids = [self.add_ready_episode(number) for number in range(1, 11)]
        summarizer = RecordingSummarizer()
        self.assertEqual(await self.run_updates(summarizer), 1)

        self.store.mark_episode_ready(episode_ids[0], "эпизод 1")
        for number in range(11, 20):
            self.add_ready_episode(number)
        self.assertEqual(await self.run_updates(summarizer), 0)

        self.store.mark_episode_ready(episode_ids[0], "уточнённый эпизод 1")
        self.assertEqual(await self.run_updates(summarizer), 1)
        self.assertEqual(
            summarizer.relationship_calls[1][2],
            tuple(f"эпизод {number}" for number in range(11, 20))
            + ("уточнённый эпизод 1",),
        )

    async def test_identity_promotion_keeps_processed_membership_from_both_users(self):
        temporary_user_id = self.store.upsert_user("telegram:991")
        self.store.bind_telegram_identity(
            temporary_user_id, 991, "temporary_name", "Новый участник"
        )

        def add_for(user_id: int, number: int, prefix: str) -> int:
            when = datetime(2026, 8, 2, tzinfo=timezone.utc) + timedelta(
                minutes=number
            )
            episode_id = self.store.store_episode(
                user_id=user_id,
                source="live",
                started_at=when,
                ended_at=when,
                search_text=f"{prefix} {number}",
                direct_exchange_count=1,
                message_row_ids=(),
                fingerprint=f"promotion-{prefix}-{number}",
            )
            self.store.mark_episode_ready(
                episode_id, f"{prefix} эпизод {number}"
            )
            return episode_id

        target_summarizer = RecordingSummarizer()
        for number in range(1, 11):
            add_for(self.user_id, number, "target")
        await memory_summaries.update_relationship_summary_blocks(
            self.store, target_summarizer, self.user_id, "Вовах"
        )
        for number in range(11, 20):
            add_for(self.user_id, number, "target")

        source_summarizer = RecordingSummarizer()
        for number in range(1, 11):
            add_for(temporary_user_id, number, "source")
        await memory_summaries.update_relationship_summary_blocks(
            self.store,
            source_summarizer,
            temporary_user_id,
            "telegram:991",
        )

        promoted_id = self.store.promote_telegram_identity(
            991, "Вовах", "testuser_a", "Вовах"
        )
        self.assertEqual(promoted_id, self.user_id)
        add_for(self.user_id, 20, "target")
        after_promotion = RecordingSummarizer()

        self.assertEqual(
            await memory_summaries.update_relationship_summary_blocks(
                self.store, after_promotion, self.user_id, "Вовах"
            ),
            1,
        )
        self.assertEqual(
            after_promotion.relationship_calls[0][2],
            tuple(f"target эпизод {number}" for number in range(11, 21)),
        )

    async def test_more_than_twenty_ready_episodes_process_repeated_blocks(self):
        for number in range(1, 26):
            self.add_ready_episode(number)
        summarizer = RecordingSummarizer()

        self.assertEqual(await self.run_updates(summarizer), 2)

        self.assertEqual([len(call[2]) for call in summarizer.relationship_calls], [10, 10])
        self.assertEqual(self.read_summary(), ("сводка отношений 2", 20))
        remaining = self.store.get_ready_summaries_after(self.user_id, 20)
        self.assertEqual(remaining, [f"эпизод {number}" for number in range(21, 26)])

    async def test_generation_failure_keeps_old_summary_and_count(self):
        self.seed_summary("старая сводка", 0)
        for number in range(1, 11):
            self.add_ready_episode(number)

        with self.assertRaises(RuntimeError):
            await self.run_updates(RecordingSummarizer(fail_relationship=True))

        self.assertEqual(self.read_summary(), ("старая сводка", 0))

    async def test_concurrent_live_updates_do_not_process_the_same_block_twice(self):
        for number in range(1, 11):
            self.add_ready_episode(number)

        class YieldingSummarizer(RecordingSummarizer):
            async def update_relationship_summary(
                self, canonical_name, previous_summary, episode_summaries
            ):
                await asyncio.sleep(0)
                return await super().update_relationship_summary(
                    canonical_name, previous_summary, episode_summaries
                )

        summarizer = YieldingSummarizer()

        results = await asyncio.gather(
            self.run_updates(summarizer), self.run_updates(summarizer)
        )

        self.assertEqual(sorted(results), [0, 1])
        self.assertEqual(len(summarizer.relationship_calls), 1)
        self.assertEqual(self.read_summary(), ("сводка отношений 1", 10))

    async def test_ready_reordering_during_generation_commits_exact_seen_block(self):
        episode_ids = [self.add_ready_episode(number) for number in range(1, 12)]
        model_started = asyncio.Event()
        let_model_finish = asyncio.Event()

        class PausingSummarizer(RecordingSummarizer):
            async def update_relationship_summary(
                self, canonical_name, previous_summary, episode_summaries
            ):
                self.relationship_calls.append(
                    (
                        canonical_name,
                        previous_summary,
                        tuple(episode_summaries),
                    )
                )
                model_started.set()
                await let_model_finish.wait()
                return "сводка точного блока"

        summarizer = PausingSummarizer()
        update_task = asyncio.create_task(self.run_updates(summarizer))
        await model_started.wait()
        self.store.mark_episode_ready(episode_ids[0], "эпизод 1")
        let_model_finish.set()

        self.assertEqual(await update_task, 1)
        with closing(sqlite3.connect(self.db_path)) as connection:
            processed = connection.execute(
                "SELECT id FROM episodes "
                "WHERE relationship_processed = 1 ORDER BY id"
            ).fetchall()
        self.assertEqual(
            [row[0] for row in processed], episode_ids[:10]
        )
        self.assertEqual(
            self.store.get_ready_summaries_after(self.user_id, 10),
            ["эпизод 11"],
        )

    async def test_changed_episode_during_generation_rejects_stale_block(self):
        episode_ids = [self.add_ready_episode(number) for number in range(1, 11)]
        model_started = asyncio.Event()
        let_model_finish = asyncio.Event()

        class PausingSummarizer(RecordingSummarizer):
            async def update_relationship_summary(
                self, canonical_name, previous_summary, episode_summaries
            ):
                model_started.set()
                await let_model_finish.wait()
                return "устаревший вывод"

        update_task = asyncio.create_task(
            self.run_updates(PausingSummarizer())
        )
        await model_started.wait()
        self.store.mark_episode_ready(
            episode_ids[0], "изменённый эпизод 1"
        )
        let_model_finish.set()

        with self.assertRaises(ValueError):
            await update_task
        self.assertIsNone(self.read_summary())
        with closing(sqlite3.connect(self.db_path)) as connection:
            processed_count = connection.execute(
                "SELECT COUNT(*) FROM episodes "
                "WHERE relationship_processed = 1"
            ).fetchone()[0]
        self.assertEqual(processed_count, 0)

    def test_save_user_summary_replaces_text_and_count_together(self):
        save = getattr(self.store, "save_user_summary", None)
        self.assertIsNotNone(save)
        for number in range(1, 21):
            self.add_ready_episode(number)

        save(self.user_id, "первая", 10)
        save(self.user_id, "вторая", 20)

        self.assertEqual(self.read_summary(), ("вторая", 20))

    def test_save_user_summary_never_regresses_processed_count(self):
        for number in range(1, 11):
            self.add_ready_episode(number)
        self.store.save_user_summary(self.user_id, "новая", 10)

        with self.assertRaises(ValueError):
            self.store.save_user_summary(self.user_id, "устаревшая", 0)

        self.assertEqual(self.read_summary(), ("новая", 10))

    def test_relationship_prompt_contains_scope_and_privacy_rules(self):
        prompt = memory_summaries.build_relationship_summary_prompt(
            "Вовах", "раньше обсуждали игры", ["потом обсуждали музыку"]
        ).casefold()

        for phrase in (
            "повторяющиеся темы",
            "предпочтительный тон",
            "незавершённые темы",
            "изменения со временем",
            "диагноз",
            "текущих мнениях",
            "дословных цитат",
            "третьих лиц",
            "инструкции",
        ):
            self.assertIn(phrase, prompt)

    async def test_episode_summarizer_generates_and_sanitizes_relationship_summary(self):
        prompts = []

        async def generate(prompt):
            prompts.append(prompt)
            return "  Аллан предпочитал короткие ответы  "

        summarizer = memory_summaries.EpisodeSummarizer(generate)
        result = await summarizer.update_relationship_summary(
            "Вовах", "старая", ["эпизод"] * 10
        )

        self.assertEqual(result, "другой человек предпочитал короткие ответы")
        self.assertEqual(len(prompts), 1)


class RelationshipSummaryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()
        self.user_id = self.store.upsert_user("Вовах")
        self.fixture_dir = Path(__file__).parent / "fixtures" / "telegram_export"

    def tearDown(self):
        self.tempdir.cleanup()

    def add_episode(self, number: int, *, ready: bool) -> int:
        when = datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(
            minutes=number
        )
        episode_id = self.store.store_episode(
            user_id=self.user_id,
            source="archive" if ready else "live",
            started_at=when,
            ended_at=when,
            search_text=f"тема {number}",
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint=f"integration-relationship-{number}",
        )
        if ready:
            self.store.mark_episode_ready(episode_id, f"эпизод {number}")
        return episode_id

    async def test_archive_ready_batch_triggers_same_relationship_updater(self):
        for number in range(1, 10):
            self.add_episode(number, ready=True)
        summarizer = RecordingSummarizer()

        await import_archive(
            self.fixture_dir, self.store, summarizer, summarize=True
        )

        self.assertEqual(len(summarizer.relationship_calls), 1)
        self.assertEqual(len(summarizer.relationship_calls[0][2]), 10)

    async def test_live_episode_completion_triggers_same_relationship_updater(self):
        for number in range(1, 10):
            self.add_episode(number, ready=True)
        pending_episode_id = self.add_episode(10, ready=False)
        summarizer = RecordingSummarizer()

        await bot._summarize_live_episode(
            self.store,
            summarizer,
            pending_episode_id,
            "Вовах",
            ("Вовах: вопрос", "Памперс: ответ"),
        )

        self.assertEqual(len(summarizer.relationship_calls), 1)
        self.assertEqual(len(summarizer.relationship_calls[0][2]), 10)


class RelationshipSummaryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "legacy.db"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_initialize_backfills_legacy_processed_episode_membership(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO schema_meta VALUES ('schema_version', '1');
                INSERT INTO schema_meta VALUES ('summary_privacy_policy_version', '1');
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    canonical_name TEXT NOT NULL UNIQUE,
                    telegram_user_id INTEGER UNIQUE,
                    username TEXT,
                    display_name TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE aliases (
                    user_id INTEGER NOT NULL,
                    alias_type TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    PRIMARY KEY (alias_type, normalized_value)
                );
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY,
                    source TEXT NOT NULL,
                    chat_key TEXT NOT NULL,
                    external_message_id INTEGER NOT NULL,
                    user_id INTEGER,
                    author_label TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    text TEXT NOT NULL,
                    media_description TEXT NOT NULL,
                    reply_to_external_id INTEGER,
                    UNIQUE (source, chat_key, external_message_id)
                );
                CREATE TABLE episodes (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    search_text TEXT NOT NULL DEFAULT '',
                    direct_exchange_count INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    last_error TEXT NOT NULL DEFAULT '',
                    fingerprint TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE episode_messages (
                    episode_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (episode_id, message_id)
                );
                CREATE TABLE user_summaries (
                    user_id INTEGER PRIMARY KEY,
                    summary TEXT NOT NULL,
                    processed_episode_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO users VALUES (
                    1, 'Вовах', NULL, NULL, NULL, '2026-08-01T00:00:00Z',
                    '2026-08-01T00:00:00Z'
                );
                INSERT INTO user_summaries VALUES (
                    1, 'старая сводка', 10, '2026-08-01T01:00:00Z'
                );
                """
            )
            for number in range(1, 13):
                status = "ready" if number <= 11 else "pending"
                connection.execute(
                    "INSERT INTO episodes VALUES "
                    "(?, 1, 'archive', ?, ?, ?, ?, 1, ?, '', ?, ?, ?)",
                    (
                        number,
                        f"2026-08-01T00:{number:02d}:00Z",
                        f"2026-08-01T00:{number:02d}:00Z",
                        f"эпизод {number}" if status == "ready" else "",
                        f"тема {number}",
                        status,
                        f"legacy-{number}",
                        "2026-08-01T00:00:00Z",
                        f"2026-08-01T00:{number:02d}:00Z",
                    ),
                )
            connection.commit()

        store = MemoryStore(self.db_path)
        store.initialize()

        with closing(sqlite3.connect(self.db_path)) as connection:
            processed = connection.execute(
                "SELECT id FROM episodes "
                "WHERE relationship_processed = 1 ORDER BY id"
            ).fetchall()
        self.assertEqual([row[0] for row in processed], list(range(1, 11)))
        self.assertEqual(store.get_ready_summaries_after(1, 10), ["эпизод 11"])

if __name__ == "__main__":
    unittest.main()
