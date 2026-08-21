from __future__ import annotations

import tempfile
import unittest
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from memory import MemoryStore
from startup_check import (
    StartupStatus,
    _gemini_health_probe,
    _telegram_health_probe,
    check_local_configuration,
    run_online_checks,
)


class StartupCheckTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "memory.db"
        self.store = MemoryStore(self.db_path)
        self.store.initialize()

    async def asyncTearDown(self):
        self.tempdir.cleanup()

    def add_pending_episode(self) -> None:
        user_id = self.store.upsert_user("Тестовый пользователь")
        when = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        self.store.store_episode(
            user_id=user_id,
            source="live",
            started_at=when,
            ended_at=when,
            search_text="synthetic searchable text",
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint="startup-check-pending",
        )

    def test_missing_chat_id_fails_closed_and_report_hides_secrets(self):
        secret = "SUPER_SECRET_TELEGRAM_TOKEN"

        status = check_local_configuration(
            {"TELEGRAM_BOT_TOKEN": secret},
            self.db_path,
        )

        self.assertFalse(status.allowed_chat_id_valid)
        self.assertFalse(status.memory_db_ready)
        self.assertNotIn(secret, status.public_report())
        self.assertNotIn("TELEGRAM_BOT_TOKEN", status.public_report())

    def test_pending_memory_is_reported_with_import_estimate(self):
        self.add_pending_episode()

        status = check_local_configuration(
            {"ALLOWED_CHAT_ID": "-100123"},
            self.db_path,
        )

        self.assertTrue(status.allowed_chat_id_valid)
        self.assertFalse(status.memory_db_ready)
        self.assertEqual(status.pending_episodes, 1)
        self.assertEqual(status.ready_episodes, 0)
        self.assertEqual(status.failed_episodes, 0)
        self.assertEqual(status.estimated_import_calls, 1)

    def test_valid_chat_and_fully_ready_database_pass_local_check(self):
        status = check_local_configuration(
            {"ALLOWED_CHAT_ID": " -100123 "},
            self.db_path,
        )

        self.assertTrue(status.allowed_chat_id_valid)
        self.assertTrue(status.memory_db_ready)
        self.assertEqual(status.errors, ())

    def test_missing_database_reports_only_exception_class(self):
        missing = Path(self.tempdir.name) / "missing.db"

        status = check_local_configuration(
            {"ALLOWED_CHAT_ID": "-100123"},
            missing,
        )

        self.assertFalse(status.memory_db_ready)
        self.assertIn("FileNotFoundError", status.errors)
        self.assertNotIn(str(missing), status.public_report())

    async def test_online_check_calls_only_injected_health_probes(self):
        calls = []
        local = check_local_configuration(
            {"ALLOWED_CHAT_ID": "-100123"},
            self.db_path,
        )

        async def telegram_probe(token: str) -> bool:
            calls.append(("telegram", token))
            return True

        async def gemini_probe(api_key: str, model: str) -> bool:
            calls.append(("gemini", api_key, model))
            return True

        result = await run_online_checks(
            local,
            {
                "TELEGRAM_BOT_TOKEN": "telegram-secret",
                "GEMINI_API_KEY": "gemini-secret",
                "GEMINI_MODEL": "test-model",
            },
            telegram_probe=telegram_probe,
            gemini_probe=gemini_probe,
        )

        self.assertEqual(
            calls,
            [
                ("telegram", "telegram-secret"),
                ("gemini", "gemini-secret", "test-model"),
            ],
        )
        self.assertTrue(result.online_checked)
        self.assertTrue(result.telegram_ok)
        self.assertTrue(result.gemini_ok)
        self.assertNotIn("secret", result.public_report())

    async def test_online_check_skips_probes_when_local_requirements_fail(self):
        calls = []
        local = StartupStatus(
            allowed_chat_id_valid=False,
            memory_db_ready=True,
            pending_episodes=0,
            ready_episodes=0,
            failed_episodes=0,
            estimated_import_calls=0,
        )

        async def telegram_probe(_token: str) -> bool:
            calls.append("telegram")
            return True

        async def gemini_probe(_api_key: str, _model: str) -> bool:
            calls.append("gemini")
            return True

        result = await run_online_checks(
            local,
            {},
            telegram_probe=telegram_probe,
            gemini_probe=gemini_probe,
        )

        self.assertEqual(calls, [])
        self.assertFalse(result.online_checked)
        self.assertIn("LocalPrerequisitesError", result.errors)

    async def test_online_failure_records_exception_class_not_message(self):
        local = check_local_configuration(
            {"ALLOWED_CHAT_ID": "-100123"},
            self.db_path,
        )

        async def telegram_probe(_token: str) -> bool:
            raise RuntimeError("PRIVATE TELEGRAM FAILURE DETAILS")

        async def gemini_probe(_api_key: str, _model: str) -> bool:
            return True

        result = await run_online_checks(
            local,
            {
                "TELEGRAM_BOT_TOKEN": "telegram-secret",
                "GEMINI_API_KEY": "gemini-secret",
            },
            telegram_probe=telegram_probe,
            gemini_probe=gemini_probe,
        )

        self.assertIn("RuntimeError", result.errors)
        self.assertNotIn("PRIVATE TELEGRAM FAILURE DETAILS", result.public_report())

    async def test_requested_online_check_without_credentials_is_not_ready(self):
        local = check_local_configuration(
            {"ALLOWED_CHAT_ID": "-100123"},
            self.db_path,
        )

        result = await run_online_checks(local, {})

        self.assertFalse(result.fully_ready)
        self.assertIn("MissingTelegramToken", result.errors)
        self.assertIn("MissingGeminiKey", result.errors)

    async def test_real_telegram_probe_uses_only_get_me_and_closes_session(self):
        calls = []

        class FakeSession:
            async def close(self):
                calls.append("close")

        class FakeBot:
            def __init__(self, token):
                calls.append(("init", token))
                self.session = FakeSession()

            async def get_me(self):
                calls.append("get_me")
                return SimpleNamespace(id=123)

        fake_aiogram = SimpleNamespace(Bot=FakeBot)
        with patch.dict(sys.modules, {"aiogram": fake_aiogram}):
            result = await _telegram_health_probe("telegram-secret")

        self.assertTrue(result)
        self.assertEqual(
            calls,
            [("init", "telegram-secret"), "get_me", "close"],
        )

    async def test_real_gemini_probe_has_a_small_output_cap(self):
        generate = AsyncMock(return_value=SimpleNamespace(text="OK"))
        client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate)
            )
        )

        with patch("google.genai.Client", return_value=client):
            result = await _gemini_health_probe("gemini-secret", "test-model")

        self.assertTrue(result)
        kwargs = generate.await_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertLessEqual(kwargs["config"].max_output_tokens, 8)

    async def test_online_health_probes_have_a_timeout(self):
        local = check_local_configuration(
            {"ALLOWED_CHAT_ID": "-100123"},
            self.db_path,
        )

        async def hanging_telegram(_token: str) -> bool:
            await __import__("asyncio").Event().wait()
            return True

        async def healthy_gemini(_api_key: str, _model: str) -> bool:
            return True

        with patch("startup_check._ONLINE_PROBE_TIMEOUT_SECONDS", 0.01):
            result = await run_online_checks(
                local,
                {
                    "TELEGRAM_BOT_TOKEN": "telegram-secret",
                    "GEMINI_API_KEY": "gemini-secret",
                },
                telegram_probe=hanging_telegram,
                gemini_probe=healthy_gemini,
            )

        self.assertFalse(result.telegram_ok)
        self.assertTrue(result.gemini_ok)
        self.assertIn("TimeoutError", result.errors)


if __name__ == "__main__":
    unittest.main()
