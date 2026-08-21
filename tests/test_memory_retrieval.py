import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from identity import normalize_alias
from memory import (
    EpisodeRecord,
    MemoryContext,
    MemoryStore,
    MessageRecord,
    render_memory_instruction,
)


class MemoryRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()
        self.base_time = datetime(2026, 8, 1, tzinfo=timezone.utc)
        self._episode_number = 0

    def tearDown(self):
        self.tempdir.cleanup()

    def add_pending_episode(
        self,
        user_id: int,
        search_text: str,
        *,
        direct_exchange_count: int = 1,
        ended_at: datetime | None = None,
        message_row_ids=(),
        fingerprint: str | None = None,
    ) -> int:
        self._episode_number += 1
        ended_at = ended_at or self.base_time + timedelta(
            minutes=self._episode_number
        )
        return self.store.store_episode(
            user_id=user_id,
            source="archive",
            started_at=ended_at - timedelta(minutes=1),
            ended_at=ended_at,
            search_text=search_text,
            direct_exchange_count=direct_exchange_count,
            message_row_ids=message_row_ids,
            fingerprint=fingerprint or f"episode-{self._episode_number}",
        )

    def add_ready_episode(
        self,
        user_id: int,
        summary: str,
        search_text: str,
        *,
        direct_exchange_count: int = 1,
        ended_at: datetime | None = None,
    ) -> int:
        episode_id = self.add_pending_episode(
            user_id,
            search_text,
            direct_exchange_count=direct_exchange_count,
            ended_at=ended_at,
        )
        self.store.mark_episode_ready(episode_id, summary)
        return episode_id

    def store_live_message(
        self,
        user_id: int,
        external_message_id: int,
        text: str,
        *,
        chat_key: str = "live:1",
    ) -> int:
        return self.store.store_message(
            MessageRecord(
                source="live",
                chat_key=chat_key,
                external_message_id=external_message_id,
                user_id=user_id,
                author_label="user",
                sent_at=self.base_time + timedelta(seconds=external_message_id),
                text=text,
                media_description="",
                reply_to_external_id=None,
            )
        )

    def set_relationship_summary(self, user_id: int, summary: str) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                INSERT INTO user_summaries(
                    user_id, summary, processed_episode_count, updated_at
                ) VALUES (?, ?, 1, ?)
                """,
                (user_id, summary, self.base_time.isoformat()),
            )
            connection.commit()

    def disable_fts(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "UPDATE schema_meta SET value = '0' WHERE key = 'fts5_available'"
            )
            connection.commit()

    def test_episode_storage_is_idempotent_and_links_messages_once(self):
        user_id = self.store.upsert_user("Вовах")
        message_id = self.store_live_message(user_id, 101, "привет")
        ended_at = self.base_time + timedelta(minutes=20)

        first = self.add_pending_episode(
            user_id,
            "привет",
            ended_at=ended_at,
            message_row_ids=(message_id,),
            fingerprint="same-exchange",
        )
        second = self.add_pending_episode(
            user_id,
            "привет",
            ended_at=ended_at,
            message_row_ids=(message_id,),
            fingerprint="same-exchange",
        )

        self.assertEqual(first, second)
        with closing(sqlite3.connect(self.db_path)) as connection:
            episode_count = connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE fingerprint = ?",
                ("same-exchange",),
            ).fetchone()[0]
            link_count = connection.execute(
                "SELECT COUNT(*) FROM episode_messages WHERE episode_id = ?",
                (first,),
            ).fetchone()[0]
        self.assertEqual(episode_count, 1)
        self.assertEqual(link_count, 1)

    def test_episode_fingerprint_cannot_resolve_to_another_users_episode(self):
        vovah = self.store.upsert_user("Вовах")
        maria = self.store.upsert_user("Маша")
        vovah_message = self.store_live_message(vovah, 111, "своя реплика")
        maria_message = self.store_live_message(maria, 112, "чужая реплика")
        self.add_pending_episode(
            vovah,
            "своя тема",
            message_row_ids=(vovah_message,),
            fingerprint="shared-fingerprint",
        )

        with self.assertRaises(ValueError):
            self.add_pending_episode(
                maria,
                "чужая тема",
                message_row_ids=(maria_message,),
                fingerprint="shared-fingerprint",
            )

    def test_episode_fingerprint_rejects_different_linked_message_ids(self):
        user_id = self.store.upsert_user("Вовах")
        first_message = self.store_live_message(user_id, 121, "первая реплика")
        second_message = self.store_live_message(user_id, 122, "вторая реплика")
        ended_at = self.base_time + timedelta(minutes=30)
        self.add_pending_episode(
            user_id,
            "одна тема",
            ended_at=ended_at,
            message_row_ids=(first_message,),
            fingerprint="immutable-exchange",
        )

        with self.assertRaises(ValueError):
            self.add_pending_episode(
                user_id,
                "одна тема",
                ended_at=ended_at,
                message_row_ids=(second_message,),
                fingerprint="immutable-exchange",
            )

    def test_episode_fingerprint_rejects_changed_immutable_fields(self):
        user_id = self.store.upsert_user("Вовах")
        ended_at = self.base_time + timedelta(minutes=40)
        immutable_values = {
            "source": "archive",
            "started_at": ended_at - timedelta(minutes=1),
            "ended_at": ended_at,
            "search_text": "одна тема",
            "direct_exchange_count": 2,
        }
        for changed_field, changed_value in (
            ("source", "live"),
            ("started_at", ended_at - timedelta(minutes=2)),
            ("ended_at", ended_at + timedelta(minutes=1)),
            ("search_text", "другая тема"),
            ("direct_exchange_count", 3),
        ):
            with self.subTest(field=changed_field):
                fingerprint = f"immutable-{changed_field}"
                self.store.store_episode(
                    user_id=user_id,
                    message_row_ids=(),
                    fingerprint=fingerprint,
                    **immutable_values,
                )
                changed_values = dict(immutable_values)
                changed_values[changed_field] = changed_value
                with self.assertRaises(ValueError):
                    self.store.store_episode(
                        user_id=user_id,
                        message_row_ids=(),
                        fingerprint=fingerprint,
                        **changed_values,
                    )

    def test_episode_timestamps_must_include_a_timezone(self):
        user_id = self.store.upsert_user("Вовах")
        aware = self.base_time
        naive = aware.replace(tzinfo=None)
        for started_at, ended_at in ((naive, aware), (aware, naive)):
            with self.subTest(started_at=started_at, ended_at=ended_at):
                with self.assertRaises(ValueError):
                    self.store.store_episode(
                        user_id=user_id,
                        source="archive",
                        started_at=started_at,
                        ended_at=ended_at,
                        search_text="тема",
                        direct_exchange_count=1,
                        message_row_ids=(),
                        fingerprint=f"naive-{started_at!r}-{ended_at!r}",
                    )

    def test_search_never_returns_another_users_episode(self):
        vovah = self.store.upsert_user("Вовах")
        maria = self.store.upsert_user("Маша")
        own_id = self.add_ready_episode(
            vovah, "обсуждали переезд и язык", "переезд язык"
        )
        self.add_ready_episode(
            maria, "обсуждали переезд и работу", "переезд работа"
        )

        context = self.store.get_memory_context(
            vovah, "переезд", "live:1", 6, 6
        )

        self.assertEqual([item.id for item in context.episodes], [own_id])
        self.assertEqual([item.user_id for item in context.episodes], [vovah])

    def test_pending_episode_is_not_model_context(self):
        user_id = self.store.upsert_user("Вовах")
        self.add_pending_episode(user_id, "СЫРАЯ СЕКРЕТНАЯ РЕПЛИКА")

        context = self.store.get_memory_context(
            user_id, "секретная", "live:1", 6, 6
        )
        instruction = render_memory_instruction(context)

        self.assertEqual(context.episodes, ())
        self.assertNotIn("СЫРАЯ СЕКРЕТНАЯ РЕПЛИКА", instruction)

    def test_episode_summary_input_excludes_linked_third_party_messages(self):
        target = self.store.upsert_user("Вовах")
        other = self.store.upsert_user("Маша")
        records = (
            MessageRecord(
                source="archive", chat_key="old-chat", external_message_id=301,
                user_id=target, author_label="V0VAH?", sent_at=self.base_time,
                text="МОЯ_ТЕМА_ШАХМАТЫ", media_description="", reply_to_external_id=None,
            ),
            MessageRecord(
                source="archive", chat_key="old-chat", external_message_id=302,
                user_id=other, author_label="Мария", sent_at=self.base_time + timedelta(seconds=1),
                text="ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_А", media_description="", reply_to_external_id=None,
            ),
            MessageRecord(
                source="archive", chat_key="old-chat", external_message_id=303,
                user_id=None, author_label="Незнакомец", sent_at=self.base_time + timedelta(seconds=2),
                text="ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_Б", media_description="", reply_to_external_id=None,
            ),
            MessageRecord(
                source="archive", chat_key="old-chat", external_message_id=304,
                user_id=None, author_label="Памперс2004", sent_at=self.base_time + timedelta(seconds=3),
                text="ОТВЕТ_ПАМПЕРСА", media_description="", reply_to_external_id=301,
            ),
            MessageRecord(
                source="archive", chat_key="old-chat", external_message_id=305,
                user_id=other, author_label="Памперс2004", sent_at=self.base_time + timedelta(seconds=4),
                text="ТРЕТЬЯ_СТОРОНА_ПОД_ИМЕНЕМ_ПАМПЕРСА", media_description="", reply_to_external_id=None,
            ),
        )
        row_ids = tuple(self.store.store_message(record) for record in records)
        self.add_pending_episode(
            target,
            "локальный индекс может содержать весь контекст",
            message_row_ids=row_ids,
        )

        candidate = self.store.get_episodes_for_summary(("pending",))[0]
        model_input = "\n".join(candidate.messages)

        self.assertIn("МОЯ_ТЕМА_ШАХМАТЫ", model_input)
        self.assertIn("ОТВЕТ_ПАМПЕРСА", model_input)
        self.assertNotIn("ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_А", model_input)
        self.assertNotIn("ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_Б", model_input)
        self.assertNotIn("ТРЕТЬЯ_СТОРОНА_ПОД_ИМЕНЕМ_ПАМПЕРСА", model_input)

    def test_retrieval_uses_safe_name_and_only_own_bounded_recent_messages(self):
        target = self.store.upsert_user("telegram:987654321")
        other = self.store.upsert_user("Маша")
        relevant_id = self.add_ready_episode(
            target,
            "раньше обсуждали шахматы",
            "шахматы",
            ended_at=self.base_time,
        )
        for number in range(4):
            self.add_ready_episode(
                target,
                f"более новый разговор {number}",
                f"другая тема {number}",
                ended_at=self.base_time + timedelta(minutes=number + 1),
            )
        self.store_live_message(target, 401, "продолжим шахматы")
        self.store_live_message(other, 402, "ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_ПОИСКА")

        context = self.store.get_memory_context(
            target, "продолжим", "live:1", 6, 1
        )
        instruction = render_memory_instruction(context)

        self.assertIn(relevant_id, [episode.id for episode in context.episodes])
        self.assertEqual(
            [message.text for message in context.recent_messages],
            ["продолжим шахматы"],
        )
        self.assertIn("неизвестный собеседник", instruction.casefold())
        self.assertNotIn("telegram:", instruction.casefold())
        self.assertNotIn("987654321", instruction)
        self.assertNotIn("ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_ПОИСКА", instruction)

    def test_recent_messages_and_relationship_summary_are_user_isolated(self):
        vovah = self.store.upsert_user("Вовах")
        maria = self.store.upsert_user("Маша")
        self.set_relationship_summary(vovah, "сводка Воваха")
        self.set_relationship_summary(maria, "чужая сводка Маши")
        self.store_live_message(vovah, 201, "своя недавняя реплика")
        self.store_live_message(maria, 202, "чужая недавняя реплика")

        context = self.store.get_memory_context(
            vovah, "нет совпадений", "live:1", 6, 6
        )

        self.assertEqual(context.canonical_name, "Вовах")
        self.assertEqual(context.relationship_summary, "сводка Воваха")
        self.assertEqual(
            [message.text for message in context.recent_messages],
            ["своя недавняя реплика"],
        )

    def test_shared_token_returns_at_most_six_best_direct_exchanges(self):
        user_id = self.store.upsert_user("Вовах")
        for number in range(8):
            self.add_ready_episode(
                user_id,
                f"общая тема, эпизод {number}",
                "общая тема",
                direct_exchange_count=number,
            )

        context = self.store.get_memory_context(
            user_id, "общая", "live:1", 6, 0
        )

        self.assertEqual(len(context.episodes), 6)
        self.assertEqual(
            [item.direct_exchange_count for item in context.episodes],
            [7, 6, 5, 4, 3, 2],
        )

    def test_no_lexical_match_returns_latest_three_ready_episodes(self):
        user_id = self.store.upsert_user("Вовах")
        episode_ids = [
            self.add_ready_episode(user_id, f"разговор {number}", f"тема {number}")
            for number in range(5)
        ]

        context = self.store.get_memory_context(
            user_id, "несуществующее-совпадение", "live:1", 6, 0
        )

        self.assertEqual(
            [item.id for item in context.episodes],
            list(reversed(episode_ids[-3:])),
        )

    def test_recency_ranking_uses_the_instant_across_utc_offsets(self):
        user_id = self.store.upsert_user("Вовах")
        earlier_id = self.add_ready_episode(
            user_id,
            "общая тема раньше",
            "общая",
            direct_exchange_count=1,
            ended_at=datetime(
                2026, 8, 1, 12, 0, tzinfo=timezone(timedelta(hours=3))
            ),
        )
        later_id = self.add_ready_episode(
            user_id,
            "общая тема позже",
            "общая",
            direct_exchange_count=1,
            ended_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
        )

        context = self.store.get_memory_context(
            user_id, "общая", "live:1", 6, 0
        )

        self.assertEqual(
            [item.id for item in context.episodes], [later_id, earlier_id]
        )

    def test_duplicate_summaries_are_removed_after_ranking(self):
        user_id = self.store.upsert_user("Вовах")
        self.add_ready_episode(
            user_id,
            "Обсуждали   переезд",
            "переезд",
            direct_exchange_count=1,
        )
        preferred_id = self.add_ready_episode(
            user_id,
            "  обсуждали переезд  ",
            "переезд",
            direct_exchange_count=5,
        )
        other_id = self.add_ready_episode(
            user_id,
            "Переезд и язык",
            "переезд язык",
            direct_exchange_count=2,
        )

        context = self.store.get_memory_context(
            user_id, "переезд", "live:1", 6, 0
        )

        self.assertEqual([item.id for item in context.episodes], [preferred_id, other_id])
        self.assertEqual(
            len({normalize_alias(item.summary) for item in context.episodes}),
            len(context.episodes),
        )

    def test_python_fallback_is_ranked_and_user_isolated(self):
        vovah = self.store.upsert_user("Вовах")
        maria = self.store.upsert_user("Маша")
        weaker_id = self.add_ready_episode(
            vovah,
            "переезд обсуждали кратко",
            "переезд",
            direct_exchange_count=1,
        )
        stronger_id = self.add_ready_episode(
            vovah,
            "переезд и язык обсуждали подробно",
            "переезд язык",
            direct_exchange_count=4,
        )
        self.add_ready_episode(
            maria,
            "переезд и чужая работа",
            "переезд работа",
            direct_exchange_count=99,
        )
        self.disable_fts()

        context = self.store.get_memory_context(
            vovah, "переезд язык", "live:1", 6, 0
        )

        self.assertEqual(
            [item.id for item in context.episodes],
            [stronger_id, weaker_id],
        )
        self.assertEqual({item.user_id for item in context.episodes}, {vovah})

    def test_rebuilt_fts_index_is_committed_and_reused(self):
        user_id = self.store.upsert_user("Вовах")
        relevant_id = self.add_ready_episode(
            user_id, "говорили о переезде", "переезд"
        )
        self.add_ready_episode(user_id, "говорили о музыке", "музыка")
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("DROP TABLE episode_fts")
            connection.commit()

        first = self.store.get_memory_context(
            user_id, "переезд", "live:1", 6, 0
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            indexed_rows = connection.execute(
                "SELECT rowid FROM episode_fts ORDER BY rowid"
            ).fetchall()
        second = self.store.get_memory_context(
            user_id, "переезд", "live:1", 6, 0
        )

        self.assertEqual([item.id for item in first.episodes], [relevant_id])
        self.assertEqual(
            [item.id for item in second.episodes],
            [item.id for item in first.episodes],
        )
        self.assertEqual(
            [row[0] for row in indexed_rows],
            sorted([relevant_id, relevant_id + 1]),
        )

    def test_render_instruction_contains_only_approved_summary_fields(self):
        context = MemoryContext(
            canonical_name="Вовах",
            relationship_summary="предпочитает короткий дружеский тон",
            episodes=(
                EpisodeRecord(
                    id=918273,
                    user_id=12345,
                    summary="интересовался изучением языка",
                    started_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
                    ended_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
                    direct_exchange_count=77,
                ),
            ),
            recent_messages=(
                MessageRecord(
                    source="live",
                    chat_key="C:/private/memory.db",
                    external_message_id=555666,
                    user_id=12345,
                    author_label="raw-author",
                    sent_at=datetime(2025, 4, 5, tzinfo=timezone.utc),
                    text="НЕОБРАБОТАННАЯ ЛИЧНАЯ РЕПЛИКА",
                    media_description="секретное описание медиа",
                    reply_to_external_id=444333,
                ),
            ),
        )

        instruction = render_memory_instruction(context)

        self.assertIn("Вовах", instruction)
        self.assertIn("предпочитает короткий дружеский тон", instruction)
        self.assertIn("интересовался изучением языка", instruction)
        self.assertIn("наблюден", instruction.casefold())
        self.assertIn("не цитируй", instruction.casefold())
        self.assertIn("не раскрывай", instruction.casefold())
        for forbidden in (
            "НЕОБРАБОТАННАЯ ЛИЧНАЯ РЕПЛИКА",
            "секретное описание медиа",
            "raw-author",
            "C:/private/memory.db",
            "918273",
            "12345",
            "555666",
            "444333",
            "2024-01-02",
            "2024-01-03",
            "2025-04-05",
            "77",
        ):
            self.assertNotIn(forbidden, instruction)

    def test_render_instruction_scrubs_technical_keys_from_saved_summaries(self):
        context = MemoryContext(
            canonical_name="Вовах",
            relationship_summary="служебный telegram:123456789",
            episodes=(
                EpisodeRecord(
                    id=1,
                    user_id=1,
                    summary="эпизод TELEGRAM:987654321",
                    started_at=self.base_time,
                    ended_at=self.base_time,
                    direct_exchange_count=1,
                ),
            ),
            recent_messages=(),
        )

        instruction = render_memory_instruction(context)

        self.assertNotIn("telegram:", instruction.casefold())
        self.assertNotIn("123456789", instruction)
        self.assertNotIn("987654321", instruction)


if __name__ == "__main__":
    unittest.main()
