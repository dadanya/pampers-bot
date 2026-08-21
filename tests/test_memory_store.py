import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory import EpisodeRecord, MemoryContext, MemoryStore, MessageRecord


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()

    def tearDown(self):
        self.tempdir.cleanup()

    def _archive_message(self, user_id: int | None = None) -> MessageRecord:
        return MessageRecord(
            source="archive",
            chat_key="old-chat",
            external_message_id=42,
            user_id=user_id,
            author_label="V0VAH?",
            sent_at=datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc),
            text="Привет",
            media_description="",
            reply_to_external_id=None,
        )

    def test_message_is_persistent_and_idempotent(self):
        user_id = self.store.upsert_user("Вовах")
        record = self._archive_message(user_id)

        first = self.store.store_message(record)
        second = self.store.store_message(record)

        self.assertEqual(first, second)
        reopened = MemoryStore(self.db_path)
        reopened.initialize()
        self.assertEqual(reopened.get_recent_messages("old-chat", 10), [record])

    def test_recent_messages_are_ordered_by_instant_across_utc_offsets(self):
        earlier = MessageRecord(
            source="archive",
            chat_key="old-chat",
            external_message_id=1,
            user_id=None,
            author_label="Арина",
            sent_at=datetime(
                2026, 8, 1, 12, 0, tzinfo=timezone(timedelta(hours=3))
            ),
            text="Раньше",
            media_description="",
            reply_to_external_id=None,
        )
        later = MessageRecord(
            source="archive",
            chat_key="old-chat",
            external_message_id=2,
            user_id=None,
            author_label="Арина",
            sent_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
            text="Позже",
            media_description="",
            reply_to_external_id=None,
        )
        self.store.store_message(earlier)
        self.store.store_message(later)

        recent = self.store.get_recent_messages("old-chat", 1)

        self.assertEqual([message.external_message_id for message in recent], [2])

    def test_message_without_timezone_is_rejected(self):
        record = MessageRecord(
            source="archive",
            chat_key="old-chat",
            external_message_id=3,
            user_id=None,
            author_label="Арина",
            sent_at=datetime(2026, 8, 1, 10, 0),
            text="Нет часового пояса",
            media_description="",
            reply_to_external_id=None,
        )

        with self.assertRaises(ValueError):
            self.store.store_message(record)

    def test_known_user_cannot_be_bound_to_a_second_telegram_id(self):
        user_id = self.store.upsert_user("Вовах")
        self.store.bind_telegram_identity(
            user_id, 1001, "testuser_a", "Вова"
        )

        with self.assertRaises(ValueError):
            self.store.bind_telegram_identity(user_id, 2002, "other", "Другой")

    def test_telegram_id_cannot_be_reused_for_another_known_user(self):
        first_user_id = self.store.upsert_user("Вовах")
        second_user_id = self.store.upsert_user("Маша")
        self.store.bind_telegram_identity(
            first_user_id, 1001, "testuser_a", "Вова"
        )

        with self.assertRaises(ValueError):
            self.store.bind_telegram_identity(
                second_user_id, 1001, "different", "Другой"
            )

    def test_normalized_alias_conflict_rolls_back_the_whole_binding(self):
        first_user_id = self.store.upsert_user("Вовах")
        second_user_id = self.store.upsert_user("Маша")
        self.store.bind_telegram_identity(
            first_user_id, 1001, "first-user", "  Общее   Имя  "
        )

        with self.assertRaises(ValueError):
            self.store.bind_telegram_identity(
                second_user_id, 2002, "temporary-alias", "общее имя"
            )

        with closing(sqlite3.connect(self.db_path)) as connection:
            second_user = connection.execute(
                "SELECT telegram_user_id, username, display_name "
                "FROM users WHERE id = ?",
                (second_user_id,),
            ).fetchone()
            partial_alias = connection.execute(
                "SELECT 1 FROM aliases WHERE normalized_value = 'temporary-alias'"
            ).fetchone()
        self.assertEqual(second_user, (None, None, None))
        self.assertIsNone(partial_alias)

    def test_repeating_the_same_binding_updates_last_known_names(self):
        user_id = self.store.upsert_user("Вовах")
        first = self.store.bind_telegram_identity(
            user_id, 1001, "testuser_a", "Вова"
        )
        second = self.store.bind_telegram_identity(
            user_id, 1001, "TESTUSER_A", "Вовах"
        )

        self.assertEqual((first, second), (user_id, user_id))
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT telegram_user_id, username, display_name FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            aliases = connection.execute(
                "SELECT alias_type, normalized_value FROM aliases WHERE user_id = ? "
                "ORDER BY alias_type, normalized_value",
                (user_id,),
            ).fetchall()
        self.assertEqual(row, (1001, "TESTUSER_A", "Вовах"))
        self.assertEqual(
            aliases,
            [
                ("display_name", "вова"),
                ("display_name", "вовах"),
                ("username", "testuser_a"),
            ],
        )

    def test_get_user_by_telegram_id_returns_only_the_bound_canonical_user(self):
        user_id = self.store.upsert_user("Вовах")
        self.assertIsNone(self.store.get_user_by_telegram_id(1001))
        self.store.bind_telegram_identity(
            user_id, 1001, "testuser_a", "Вовах"
        )

        self.assertEqual(
            self.store.get_user_by_telegram_id(1001),
            (user_id, "Вовах"),
        )
        self.assertIsNone(self.store.get_user_by_telegram_id(2002))

    def test_promote_temporary_telegram_user_atomically_merges_all_memory(self):
        temporary_id = self.store.upsert_user("telegram:1001")
        self.store.bind_telegram_identity(
            temporary_id, 1001, "temporary-name", "Вова"
        )
        target_id = self.store.upsert_user("Вовах")
        self.store.bind_archive_alias(target_id, "V0VAH?")
        message_id = self.store.store_message(self._archive_message(temporary_id))
        when = datetime(2026, 8, 1, 13, 0, tzinfo=timezone.utc)
        episode_id = self.store.store_episode(
            user_id=temporary_id,
            source="live",
            started_at=when,
            ended_at=when,
            search_text="ранний разговор",
            direct_exchange_count=1,
            message_row_ids=(message_id,),
            fingerprint="temporary-live-episode",
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.executemany(
                "INSERT INTO user_summaries(user_id, summary, processed_episode_count, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    (temporary_id, "временная сводка", 2, "2026-08-01T10:00:00+00:00"),
                    (target_id, "архивная сводка", 3, "2026-08-02T10:00:00+00:00"),
                ),
            )
            connection.commit()

        promoted_id = self.store.promote_telegram_identity(
            telegram_user_id=1001,
            canonical_name="Вовах",
            username="testuser_a",
            display_name="Вовах",
        )
        repeated_id = self.store.promote_telegram_identity(
            telegram_user_id=1001,
            canonical_name="Вовах",
            username="testuser_a",
            display_name="Вовах",
        )

        self.assertEqual((promoted_id, repeated_id), (target_id, target_id))
        with closing(sqlite3.connect(self.db_path)) as connection:
            users = connection.execute(
                "SELECT id, canonical_name, telegram_user_id FROM users ORDER BY id"
            ).fetchall()
            message_user = connection.execute(
                "SELECT user_id FROM messages WHERE id = ?", (message_id,)
            ).fetchone()[0]
            episode_user = connection.execute(
                "SELECT user_id FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()[0]
            aliases = connection.execute(
                "SELECT alias_type, normalized_value FROM aliases WHERE user_id = ? "
                "ORDER BY alias_type, normalized_value",
                (target_id,),
            ).fetchall()
            summary = connection.execute(
                "SELECT summary, processed_episode_count FROM user_summaries WHERE user_id = ?",
                (target_id,),
            ).fetchone()
        self.assertEqual(users, [(target_id, "Вовах", 1001)])
        self.assertEqual((message_user, episode_user), (target_id, target_id))
        self.assertEqual(
            aliases,
            [
                ("archive_name", "v0vah?"),
                ("display_name", "вова"),
                ("display_name", "вовах"),
                ("username", "testuser_a"),
                ("username", "temporary-name"),
            ],
        )
        self.assertEqual(summary, ("архивная сводка\nвременная сводка", 5))

    def test_promote_never_merges_two_different_telegram_ids(self):
        temporary_id = self.store.upsert_user("telegram:1001")
        self.store.bind_telegram_identity(temporary_id, 1001, "temp", "Первый")
        target_id = self.store.upsert_user("Вовах")
        self.store.bind_telegram_identity(
            target_id, 2002, "testuser_a", "Второй"
        )

        with self.assertRaises(ValueError):
            self.store.promote_telegram_identity(
                telegram_user_id=1001,
                canonical_name="Вовах",
                username="testuser_a",
                display_name="Вовах",
            )

        with closing(sqlite3.connect(self.db_path)) as connection:
            bindings = connection.execute(
                "SELECT canonical_name, telegram_user_id FROM users ORDER BY canonical_name"
            ).fetchall()
        self.assertEqual(bindings, [("telegram:1001", 1001), ("Вовах", 2002)])

    def test_foreign_keys_reject_a_message_for_a_missing_user(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.store_message(self._archive_message(user_id=999999))

    def test_schema_version_and_wal_mode_persist_after_reopening(self):
        expected_tables = {
            "aliases",
            "episode_messages",
            "episodes",
            "messages",
            "schema_meta",
            "user_summaries",
            "users",
        }

        reopened = MemoryStore(self.db_path)
        reopened.initialize()
        with closing(sqlite3.connect(self.db_path)) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            schema_version = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0]

        self.assertEqual(journal_mode, "wal")
        self.assertEqual(tables, expected_tables)
        self.assertEqual(schema_version, "1")

    def test_legacy_summary_privacy_migration_invalidates_derived_data_once(self):
        user_id = self.store.upsert_user("Вовах")
        when = datetime(2026, 8, 1, 13, 0, tzinfo=timezone.utc)
        episode_id = self.store.store_episode(
            user_id=user_id,
            source="archive",
            started_at=when,
            ended_at=when,
            search_text="сырой локальный текст",
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint="legacy-policy-episode",
        )
        self.store.mark_episode_ready(episode_id, "СТАРАЯ_НЕБЕЗОПАСНАЯ_СВОДКА")
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO user_summaries(user_id, summary, processed_episode_count, updated_at) "
                "VALUES (?, ?, 1, ?)",
                (user_id, "СТАРАЯ_СВОДКА_ОТНОШЕНИЙ", when.isoformat()),
            )
            connection.execute(
                "UPDATE episodes SET relationship_processed = 1 WHERE id = ?",
                (episode_id,),
            )
            connection.execute(
                "DELETE FROM schema_meta WHERE key = 'summary_privacy_policy_version'"
            )
            connection.commit()

        reopened = MemoryStore(self.db_path)
        reopened.initialize()

        with closing(sqlite3.connect(self.db_path)) as connection:
            episode = connection.execute(
                "SELECT status, summary, relationship_processed FROM episodes WHERE id = ?",
                (episode_id,),
            ).fetchone()
            user_summary_count = connection.execute(
                "SELECT COUNT(*) FROM user_summaries"
            ).fetchone()[0]
            marker = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'summary_privacy_policy_version'"
            ).fetchone()
            fts_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'episode_fts'"
            ).fetchone()
            fts_count = (
                connection.execute("SELECT COUNT(*) FROM episode_fts").fetchone()[0]
                if fts_table is not None
                else 0
            )
        self.assertEqual(episode, ("pending", "", 0))
        self.assertEqual(user_summary_count, 0)
        self.assertEqual(marker, ("1",))
        self.assertEqual(fts_count, 0)

        reopened.mark_episode_ready(episode_id, "НОВАЯ_БЕЗОПАСНАЯ_СВОДКА")
        MemoryStore(self.db_path).initialize()
        with closing(sqlite3.connect(self.db_path)) as connection:
            after_second_reopen = connection.execute(
                "SELECT status, summary FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
        self.assertEqual(after_second_reopen, ("ready", "НОВАЯ_БЕЗОПАСНАЯ_СВОДКА"))

    def test_fresh_database_is_marked_current_without_future_invalidation(self):
        user_id = self.store.upsert_user("Вовах")
        when = datetime(2026, 8, 1, 14, 0, tzinfo=timezone.utc)
        episode_id = self.store.store_episode(
            user_id=user_id,
            source="live",
            started_at=when,
            ended_at=when,
            search_text="безопасный текст",
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint="fresh-policy-episode",
        )
        self.store.mark_episode_ready(episode_id, "БЕЗОПАСНАЯ_СВОДКА")

        MemoryStore(self.db_path).initialize()

        with closing(sqlite3.connect(self.db_path)) as connection:
            result = connection.execute(
                "SELECT status, summary FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            marker = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'summary_privacy_policy_version'"
            ).fetchone()
        self.assertEqual(result, ("ready", "БЕЗОПАСНАЯ_СВОДКА"))
        self.assertEqual(marker, ("1",))

    def test_public_context_records_are_frozen_value_objects(self):
        episode = EpisodeRecord(
            id=7,
            user_id=3,
            summary="Обсуждали планы",
            started_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
            ended_at=datetime(2026, 7, 2, tzinfo=timezone.utc),
            direct_exchange_count=2,
        )
        context = MemoryContext(
            canonical_name="Вовах",
            relationship_summary="Давно общаются",
            episodes=(episode,),
            recent_messages=(self._archive_message(user_id=3),),
        )

        self.assertEqual(context.episodes, (episode,))
        with self.assertRaises(AttributeError):
            context.canonical_name = "Маша"


if __name__ == "__main__":
    unittest.main()
