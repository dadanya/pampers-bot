import asyncio
import io
import json
import sqlite3
import tempfile
import unittest
import os
from contextlib import closing, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from archive_parser import ArchiveMessage
from import_history import (
    ImportStats,
    build_argument_parser,
    import_archive,
    main,
    read_import_status,
    _build_summarizer_from_environment,
)
from memory_summaries import SummaryInput
from memory import MemoryStore, MessageRecord


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    async def summarize_batch(self, items):
        self.calls.append(tuple(items))
        return {
            item.episode_id: f"нейтральная сводка {item.episode_id}"
            for item in items
        }


class PartialSummarizer(FakeSummarizer):
    async def summarize_batch(self, items):
        self.calls.append(tuple(items))
        return {items[0].episode_id: "только один результат"}


class SecretArchiveError(RuntimeError):
    pass


class FailingSummarizer(FakeSummarizer):
    async def summarize_batch(self, items):
        self.calls.append(tuple(items))
        raise SecretArchiveError("raw archive text must never be stored")


class ImportHistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.fixture_dir = (
            Path(__file__).parent / "fixtures" / "telegram_export"
        )
        self.db_path = self.root / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.summarizer = FakeSummarizer()

    def tearDown(self):
        self.tempdir.cleanup()

    async def test_second_import_does_not_duplicate_rows_or_summaries(self):
        first = await import_archive(
            self.fixture_dir, self.store, self.summarizer, summarize=True
        )
        second = await import_archive(
            self.fixture_dir, self.store, self.summarizer, summarize=True
        )

        self.assertIsInstance(first, ImportStats)
        self.assertEqual(first.parsed_messages, 5)
        self.assertEqual(first.created_messages, 5)
        self.assertEqual(first.created_episodes, 1)
        self.assertEqual(first.ready_episodes, 1)
        self.assertEqual(first.summary_requests, 1)
        self.assertEqual(second.created_messages, 0)
        self.assertEqual(second.created_episodes, 0)
        self.assertEqual(second.summary_requests, 0)
        self.assertEqual(len(self.summarizer.calls), 1)

        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                10,
            )

    async def test_larger_overlapping_export_reuses_episode_anchor(self):
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)

        def message(mid, author, minute, text, reply_to=None):
            return ArchiveMessage(
                id=mid,
                author=author,
                sent_at=base + timedelta(minutes=minute),
                text=text,
                media_description="",
                reply_to=reply_to,
                page="messages.html",
            )

        narrow = [
            message(1, "V0VAH?", 1, "вопрос"),
            message(2, "Памперс2004", 2, "ответ", reply_to=1),
        ]
        expanded = [
            message(0, "V0VAH?", 0, "контекст до"),
            *narrow,
            message(3, "V0VAH?", 3, "контекст после"),
        ]

        with patch("import_history.parse_export", side_effect=[narrow, expanded]):
            first = await import_archive(
                self.fixture_dir, self.store, self.summarizer, summarize=True
            )
            second = await import_archive(
                self.fixture_dir, self.store, self.summarizer, summarize=True
            )

        self.assertEqual(first.created_episodes, 1)
        self.assertEqual(second.created_messages, 2)
        self.assertEqual(second.created_episodes, 0)
        self.assertEqual(second.ready_episodes, 1)
        self.assertEqual(second.summary_requests, 0)
        self.assertEqual(len(self.summarizer.calls), 1)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                1,
            )

    async def test_new_reply_in_overlapping_window_resummarizes_merged_episode(self):
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)

        def message(mid, author, minute, text, reply_to=None):
            return ArchiveMessage(
                id=mid,
                author=author,
                sent_at=base + timedelta(minutes=minute),
                text=text,
                media_description="",
                reply_to=reply_to,
                page="messages.html",
            )

        first_exchange = [
            message(1, "V0VAH?", 1, "вопрос 1"),
            message(2, "Памперс2004", 2, "ответ 1", reply_to=1),
        ]
        merged_exchange = [
            message(0, "V0VAH?", 0, "более ранний контекст"),
            *first_exchange,
            message(3, "V0VAH?", 3, "вопрос 2", reply_to=2),
            message(4, "Памперс2004", 4, "ответ 2", reply_to=3),
        ]

        with patch(
            "import_history.parse_export",
            side_effect=[first_exchange, merged_exchange],
        ):
            await import_archive(
                self.fixture_dir, self.store, self.summarizer, summarize=True
            )
            second = await import_archive(
                self.fixture_dir, self.store, self.summarizer, summarize=True
            )

        self.assertEqual(second.created_episodes, 0)
        self.assertEqual(second.summary_requests, 1)
        self.assertEqual(len(self.summarizer.calls), 2)
        self.assertTrue(
            self.summarizer.calls[1][0].messages[0].endswith(
                "более ранний контекст"
            )
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            count, exchanges, status, search_text = connection.execute(
                """
                SELECT COUNT(*), direct_exchange_count, status, search_text
                FROM episodes
                """
            ).fetchone()
            linked_count = connection.execute(
                "SELECT COUNT(*) FROM episode_messages"
            ).fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(exchanges, 3)
        self.assertEqual(status, "ready")
        self.assertIn("вопрос 2", search_text)
        self.assertEqual(linked_count, 5)

    async def test_pre_anchor_database_backfills_before_expanded_import(self):
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)

        def message(mid, author, minute, text, reply_to=None):
            return ArchiveMessage(
                id=mid,
                author=author,
                sent_at=base + timedelta(minutes=minute),
                text=text,
                media_description="",
                reply_to=reply_to,
                page="messages.html",
            )

        narrow = [
            message(1, "V0VAH?", 1, "вопрос"),
            message(2, "Памперс2004", 2, "ответ", reply_to=1),
        ]
        expanded = [
            message(0, "V0VAH?", 0, "контекст"),
            *narrow,
            message(3, "V0VAH?", 3, "после"),
        ]
        with patch("import_history.parse_export", return_value=narrow):
            await import_archive(
                self.fixture_dir, self.store, self.summarizer, summarize=True
            )
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "DELETE FROM schema_meta "
                "WHERE key LIKE 'episode_anchor:%' "
                "OR key = 'archive_anchor_backfill_v1'"
            )
            connection.commit()

        with patch("import_history.parse_export", return_value=expanded):
            second = await import_archive(
                self.fixture_dir, self.store, self.summarizer, summarize=True
            )

        self.assertEqual(second.created_episodes, 0)
        self.assertEqual(second.summary_requests, 0)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                1,
            )

    async def test_pre_anchor_duplicate_episodes_are_reconciled(self):
        with patch("import_history.parse_export") as parse:
            from archive_parser import parse_export as real_parse_export

            parsed = real_parse_export(self.fixture_dir)
            parse.return_value = parsed
            await import_archive(
                self.fixture_dir, self.store, None, summarize=False
            )
        with closing(sqlite3.connect(self.db_path)) as connection:
            user_id, started_at, ended_at, search_text = connection.execute(
                "SELECT user_id, started_at, ended_at, search_text FROM episodes"
            ).fetchone()
            message_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT message_id FROM episode_messages ORDER BY position"
                ).fetchall()
            )
        self.store.store_episode(
            user_id=user_id,
            source="archive",
            started_at=datetime.fromisoformat(started_at),
            ended_at=datetime.fromisoformat(ended_at),
            search_text=search_text,
            direct_exchange_count=1,
            message_row_ids=message_ids,
            fingerprint="legacy-duplicate",
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "DELETE FROM schema_meta "
                "WHERE key LIKE 'episode_anchor:%' "
                "OR key = 'archive_anchor_backfill_v1'"
            )
            connection.commit()

        with patch("import_history.parse_export", return_value=parsed):
            stats = await import_archive(
                self.fixture_dir, self.store, None, summarize=False
            )

        self.assertEqual(stats.created_episodes, 0)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM schema_meta "
                    "WHERE key LIKE 'episode_anchor:%'"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM aliases WHERE alias_type='archive_name'"
                ).fetchone()[0],
                10,
            )

    async def test_no_summarize_leaves_episode_pending_and_calls_no_model(self):
        stats = await import_archive(
            self.fixture_dir, self.store, None, summarize=False
        )

        self.assertEqual(stats.pending_episodes, 1)
        self.assertEqual(stats.ready_episodes, 0)
        self.assertEqual(stats.summary_requests, 0)

    async def test_resume_retries_pending_and_failed_in_batches_of_twenty(self):
        self.store.initialize()
        user_id = self.store.upsert_user("Вовах")
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)
        episode_ids = []
        for number in range(23):
            message_id = self.store.store_message(
                MessageRecord(
                    source="archive",
                    chat_key="archive:test-seed",
                    external_message_id=10_000 + number,
                    user_id=user_id,
                    author_label="V0VAH?",
                    sent_at=base + timedelta(minutes=number),
                    text=f"текст {number}",
                    media_description="",
                    reply_to_external_id=None,
                )
            )
            episode_ids.append(
                self.store.store_episode(
                    user_id=user_id,
                    source="archive",
                    started_at=base + timedelta(minutes=number),
                    ended_at=base + timedelta(minutes=number),
                    search_text=f"текст {number}",
                    direct_exchange_count=1,
                    message_row_ids=(message_id,),
                    fingerprint=f"seed-{number}",
                )
            )
        self.store.mark_episode_ready(episode_ids[0], "готовая сводка")
        self.store.mark_episodes_failed((episode_ids[1],), "PriorError")
        empty_export = self.root / "empty-export"
        empty_export.mkdir()

        stats = await import_archive(
            empty_export,
            self.store,
            self.summarizer,
            summarize=True,
            resume=True,
        )

        self.assertEqual([len(call) for call in self.summarizer.calls], [20, 2])
        sent_ids = {
            item.episode_id for call in self.summarizer.calls for item in call
        }
        self.assertNotIn(episode_ids[0], sent_ids)
        self.assertIn(episode_ids[1], sent_ids)
        self.assertEqual(stats.summary_requests, 2)
        self.assertEqual(stats.ready_episodes, 23)
        self.assertEqual(stats.pending_episodes, 0)
        self.assertEqual(stats.failed_episodes, 0)

    async def test_invalid_batch_result_marks_whole_batch_failed(self):
        await import_archive(
            self.fixture_dir, self.store, None, summarize=False
        )
        # A second pending episode proves that a partial result is never committed.
        user_id = self.store.upsert_user("Вовах")
        first_message = self.store.get_recent_messages(
            "archive:telegram-export", 1
        )[0]
        message_id = self.store.store_message(
            MessageRecord(
                source="archive",
                chat_key="archive:test-extra",
                external_message_id=999,
                user_id=user_id,
                author_label="V0VAH?",
                sent_at=first_message.sent_at + timedelta(hours=1),
                text="ещё один эпизод",
                media_description="",
                reply_to_external_id=None,
            )
        )
        self.store.store_episode(
            user_id=user_id,
            source="archive",
            started_at=first_message.sent_at + timedelta(hours=1),
            ended_at=first_message.sent_at + timedelta(hours=1),
            search_text="ещё один эпизод",
            direct_exchange_count=1,
            message_row_ids=(message_id,),
            fingerprint="extra-pending",
        )

        stats = await import_archive(
            self.fixture_dir,
            self.store,
            PartialSummarizer(),
            summarize=True,
        )

        self.assertEqual(stats.ready_episodes, 0)
        self.assertEqual(stats.failed_episodes, 2)
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(
                "SELECT summary, last_error FROM episodes ORDER BY id"
            ).fetchall()
        self.assertEqual(rows, [("", "ValueError"), ("", "ValueError")])

    async def test_failed_batch_stores_exception_class_only(self):
        stats = await import_archive(
            self.fixture_dir,
            self.store,
            FailingSummarizer(),
            summarize=True,
        )

        self.assertEqual(stats.failed_episodes, 1)
        with closing(sqlite3.connect(self.db_path)) as connection:
            status, summary, last_error = connection.execute(
                "SELECT status, summary, last_error FROM episodes"
            ).fetchone()
        self.assertEqual((status, summary, last_error), (
            "failed", "", "SecretArchiveError"
        ))
        self.assertNotIn("raw archive text", last_error)


class ImportHistoryCliTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.export_dir = self.root / "Чат архив"
        self.export_dir.mkdir()
        self.db_path = self.root / "память.db"

    def tearDown(self):
        self.tempdir.cleanup()

    def seed_pending_episodes(self, count, *, user_name="Вовах", offset=0):
        store = MemoryStore(self.db_path)
        store.initialize()
        user_id = store.upsert_user(user_name)
        when = datetime(2026, 8, 1, tzinfo=timezone.utc)
        for number in range(count):
            store.store_episode(
                user_id=user_id,
                source="archive",
                started_at=when,
                ended_at=when,
                search_text="x",
                direct_exchange_count=1,
                message_row_ids=(),
                fingerprint=f"budget-{offset + number}",
            )

    def seed_1421_distributed_pending_episodes(self):
        # Ten users leave 51 episodes outside complete relationship blocks:
        # 7 * 146 + 3 * 133 = 1421, while 7 * 14 + 3 * 13 = 137.
        offset = 0
        for user_number, count in enumerate([146] * 7 + [133] * 3):
            self.seed_pending_episodes(
                count,
                user_name=f"synthetic-user-{user_number}",
                offset=offset,
            )
            offset += count

    def test_parser_defaults_and_cyrillic_windows_paths(self):
        parser = build_argument_parser()
        args = parser.parse_args(
            [str(self.export_dir), "--db", str(self.db_path)]
        )

        self.assertEqual(args.export_dir, self.export_dir)
        self.assertEqual(args.db, self.db_path)
        self.assertFalse(args.no_summarize)
        self.assertFalse(args.resume)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.status)
        self.assertFalse(args.allow_over_budget)

    def test_dry_run_does_not_create_database_or_build_summarizer(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [str(self.export_dir), "--db", str(self.db_path), "--dry-run"]
            )

        self.assertEqual(exit_code, 0)
        self.assertFalse(self.db_path.exists())
        self.assertIn("parsed_messages=0", stdout.getvalue())
        self.assertIn("episode_drafts=0", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_first_summarized_run_reports_rows_created_by_raw_import_phase(self):
        fixture_dir = Path(__file__).parent / "fixtures" / "telegram_export"
        output = io.StringIO()

        with patch(
            "import_history._build_summarizer_from_environment",
            return_value=FakeSummarizer(),
        ), redirect_stdout(output):
            exit_code = main([str(fixture_dir), "--db", str(self.db_path)])

        self.assertEqual(exit_code, 0)
        self.assertIn("created_messages=5", output.getvalue())
        self.assertIn("created_episodes=1", output.getvalue())
        self.assertIn("summary_requests=1", output.getvalue())

    def test_nonexistent_export_directory_is_rejected_with_exit_two(self):
        missing = self.root / "нет-такого-чата"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                main([str(missing), "--db", str(self.db_path), "--dry-run"])

        self.assertEqual(caught.exception.code, 2)
        self.assertIn("export directory does not exist", stderr.getvalue())
        self.assertNotIn("GEMINI_API_KEY", stderr.getvalue())

    def test_status_is_read_only_and_reports_estimated_batched_requests(self):
        store = MemoryStore(self.db_path)
        store.initialize()
        user_id = store.upsert_user("Вовах")
        base = datetime(2026, 8, 1, tzinfo=timezone.utc)
        for number in range(21):
            episode_id = store.store_episode(
                user_id=user_id,
                source="archive",
                started_at=base,
                ended_at=base,
                search_text="x",
                direct_exchange_count=1,
                message_row_ids=(),
                fingerprint=f"status-{number}",
            )
            if number == 0:
                store.mark_episode_ready(episode_id, "ready")
            elif number == 1:
                store.mark_episodes_failed((episode_id,), "OldError")
        before = self.db_path.stat().st_mtime_ns

        status = read_import_status(self.db_path)
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [str(self.export_dir), "--db", str(self.db_path), "--status"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            status,
            {
                "pending": 19,
                "ready": 1,
                "failed": 1,
                "episode_summary_requests": 1,
                "relationship_summary_requests": 2,
                "estimated_requests": 3,
            },
        )
        self.assertEqual(self.db_path.stat().st_mtime_ns, before)
        lines = output.getvalue().splitlines()
        self.assertEqual(
            lines,
            [
                "pending=19",
                "ready=1",
                "failed=1",
                "episode_summary_requests=1",
                "relationship_summary_requests=2",
                "estimated_requests=3",
            ],
        )

    def test_status_counts_1421_pending_episode_and_relationship_requests(self):
        self.seed_1421_distributed_pending_episodes()

        status = read_import_status(self.db_path)

        self.assertEqual(status["episode_summary_requests"], 72)
        self.assertEqual(status["relationship_summary_requests"], 137)
        self.assertEqual(status["estimated_requests"], 209)

    def test_cli_aborts_over_budget_before_building_model_client(self):
        self.seed_1421_distributed_pending_episodes()
        stderr = io.StringIO()

        with patch("import_history._build_summarizer_from_environment") as build:
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                main([str(self.export_dir), "--db", str(self.db_path)])

        self.assertEqual(caught.exception.code, 2)
        build.assert_not_called()
        self.assertIn("209", stderr.getvalue())
        self.assertIn("--allow-over-budget", stderr.getvalue())

    def test_cli_override_allows_over_budget_summarization(self):
        self.seed_1421_distributed_pending_episodes()
        summarizer = FakeSummarizer()

        with patch(
            "import_history._build_summarizer_from_environment",
            return_value=summarizer,
        ) as build:
            exit_code = main(
                [
                    str(self.export_dir),
                    "--db",
                    str(self.db_path),
                    "--allow-over-budget",
                ]
            )

        self.assertEqual(exit_code, 0)
        build.assert_called_once_with()
        self.assertEqual(len(summarizer.calls), 72)


class ImportHistoryGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_episode_batches_use_json_but_relationship_uses_plain_text(self):
        calls = []

        class FakeModels:
            async def generate_content(self, **kwargs):
                calls.append(kwargs)
                config = kwargs.get("config")
                mime_type = getattr(config, "response_mime_type", None)
                text = (
                    '[{"episode_id": 7, "summary": "сводка"}]'
                    if mime_type == "application/json"
                    else "обычная сводка отношений"
                )
                return type("Response", (), {"text": text})()

        fake_client = type(
            "Client",
            (),
            {"aio": type("Aio", (), {"models": FakeModels()})()},
        )()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "dotenv.load_dotenv"
        ), patch("google.genai.Client", return_value=fake_client):
            summarizer = _build_summarizer_from_environment()

        batch_result = await summarizer.summarize_batch(
            [SummaryInput(7, "Вовах", ("сообщение",))]
        )
        relationship_result = await summarizer.update_relationship_summary(
            "Вовах", "", ["сводка"] * 10
        )

        self.assertEqual(batch_result, {7: "сводка"})
        self.assertEqual(relationship_result, "обычная сводка отношений")
        self.assertEqual(
            getattr(calls[0]["config"], "response_mime_type", None),
            "application/json",
        )
        self.assertNotIn("config", calls[1])


if __name__ == "__main__":
    unittest.main()
