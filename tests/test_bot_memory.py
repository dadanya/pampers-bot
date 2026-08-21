from __future__ import annotations

import asyncio
import copy
import os
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


os.environ["TELEGRAM_BOT_TOKEN"] = "123456:TEST_TOKEN_FOR_UNIT_TESTS"
os.environ["GEMINI_API_KEY"] = ""
os.environ["ALLOWED_CHAT_ID"] = ""

import bot
from memory import MemoryStore, MessageRecord


class FakeTelegramMessage:
    def __init__(
        self,
        *,
        chat_id: int = 100,
        chat_type: str = "group",
        chat_title: str = "Чат Одесского маньяка",
        message_id: int = 10,
        user_id: int = 55,
        username: str | None = "testuser_a",
        first_name: str = "Вова",
        last_name: str | None = None,
        text: str | None = "",
        caption: str | None = None,
        sticker=None,
        reply_to_message=None,
    ):
        self.chat = SimpleNamespace(
            id=chat_id,
            type=chat_type,
            title=chat_title,
        )
        self.message_id = message_id
        self.from_user = SimpleNamespace(
            id=user_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            full_name=" ".join(part for part in (first_name, last_name) if part),
        )
        self.text = text
        self.caption = caption
        self.sticker = sticker
        self.reply_to_message = reply_to_message
        self.date = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        self.photo = None
        self.video = None
        self.animation = None
        self.audio = None
        self.voice = None
        self.video_note = None
        self.document = None
        self.reply = AsyncMock(side_effect=self._reply)
        self.reply_voice = AsyncMock(side_effect=self._reply_voice)
        self.reply_sticker = AsyncMock(side_effect=self._reply_sticker)
        self.answer = AsyncMock(side_effect=self._reply)

    def _sent(self, kind: str):
        return SimpleNamespace(
            message_id=self.message_id + 1000,
            date=self.date + timedelta(seconds=1),
            kind=kind,
        )

    async def _reply(self, _text):
        return self._sent("text")

    async def _reply_voice(self, _voice):
        return self._sent("voice")

    async def _reply_sticker(self, _sticker):
        return self._sent("sticker")


class BotMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tempdir.name) / "memory.db")
        self.store.initialize()

        self.saved_globals = {
            "ALLOWED_CHAT_ID": bot.ALLOWED_CHAT_ID,
            "memory_store": bot.memory_store,
            "episode_summarizer": bot.episode_summarizer,
            "gemini_client": bot.gemini_client,
            "BOT_USERNAME": bot.BOT_USERNAME,
            "BOT_ID": bot.BOT_ID,
            "STICKER_PROBABILITY": bot.STICKER_PROBABILITY,
            "VOICE_REPLY_PROBABILITY": bot.VOICE_REPLY_PROBABILITY,
            "CAPS_PROBABILITY": bot.CAPS_PROBABILITY,
            "WORD_CAPS_PROBABILITY": bot.WORD_CAPS_PROBABILITY,
            "MEMORY_MAX_EPISODES": bot.MEMORY_MAX_EPISODES,
            "MEMORY_CONTEXT_MESSAGES": bot.MEMORY_CONTEXT_MESSAGES,
            "sticker_file_ids": bot.sticker_file_ids,
        }
        bot.ALLOWED_CHAT_ID = 100
        bot.memory_store = self.store
        bot.episode_summarizer = None
        bot.gemini_client = object()
        bot.BOT_USERNAME = "bot"
        bot.BOT_ID = 999
        bot.STICKER_PROBABILITY = 0
        bot.VOICE_REPLY_PROBABILITY = 0
        bot.CAPS_PROBABILITY = 0
        bot.WORD_CAPS_PROBABILITY = 0
        bot.sticker_file_ids = []
        bot.history.clear()
        bot.chat_locks.clear()
        bot.known_group_chats.clear()
        bot.summary_tasks.clear()

    async def asyncTearDown(self):
        if bot.summary_tasks:
            await asyncio.gather(*tuple(bot.summary_tasks), return_exceptions=True)
        for name, value in self.saved_globals.items():
            setattr(bot, name, value)
        self.tempdir.cleanup()

    def test_parse_allowed_chat_id_is_fail_closed(self):
        for value in (None, "", "   ", "not-a-number", "12.5"):
            with self.subTest(value=value):
                self.assertIsNone(bot.parse_allowed_chat_id(value))
        self.assertEqual(bot.parse_allowed_chat_id(" -100123 "), -100123)

    def test_memory_limits_are_positive_and_fail_closed_to_defaults(self):
        self.assertEqual(bot.parse_memory_limit(None, 6), 6)
        self.assertEqual(bot.parse_memory_limit("", 6), 6)
        self.assertEqual(bot.parse_memory_limit("bad", 6), 6)
        self.assertEqual(bot.parse_memory_limit("0", 6), 6)
        self.assertEqual(bot.parse_memory_limit("-2", 6), 6)
        self.assertEqual(bot.parse_memory_limit(" 9 ", 6), 9)

    async def test_configured_memory_limits_are_used_for_retrieval(self):
        bot.MEMORY_MAX_EPISODES = 4
        bot.MEMORY_CONTEXT_MESSAGES = 3
        message = FakeTelegramMessage(text="@bot переезд")

        with (
            patch.object(bot, "call_gemini", AsyncMock(return_value="помню")),
            patch.object(
                self.store,
                "get_memory_context",
                wraps=self.store.get_memory_context,
            ) as get_context,
        ):
            await bot.handle_message(message)

        self.assertEqual(get_context.call_args.args[-2:], (4, 3))

    def test_memory_is_enabled_only_for_the_configured_chat_and_live_store(self):
        self.assertTrue(bot.memory_enabled_for(100))
        self.assertFalse(bot.memory_enabled_for(200))
        bot.memory_store = None
        self.assertFalse(bot.memory_enabled_for(100))

    def test_memory_initialization_failure_disables_store_without_logging_details(self):
        failure = OSError("private database path")
        with (
            patch.object(bot, "MemoryStore", side_effect=failure),
            self.assertLogs("pampers-bot", level="ERROR") as captured,
        ):
            result = bot._initialize_memory_store(
                100, Path(self.tempdir.name) / "unavailable.db"
            )

        self.assertIsNone(result)
        output = "\n".join(captured.output)
        self.assertIn("OSError", output)
        self.assertNotIn("private database path", output)

    async def test_unaddressed_message_in_allowed_chat_is_saved_but_not_answered(self):
        message = FakeTelegramMessage(text="обычная реплика")
        generate = AsyncMock(return_value="не должно вызываться")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        saved = self.store.get_recent_messages("live:100", 10)
        self.assertEqual([item.text for item in saved], ["обычная реплика"])
        generate.assert_not_awaited()
        message.reply.assert_not_awaited()

    async def test_unaddressed_aggression_does_not_change_response_frequency(self):
        message = FakeTelegramMessage(text="ты тупой, закрой рот")
        generate = AsyncMock(return_value="не должно вызываться")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        self.assertEqual(
            [item.text for item in self.store.get_recent_messages("live:100", 10)],
            ["ты тупой, закрой рот"],
        )
        generate.assert_not_awaited()
        message.reply.assert_not_awaited()

    async def test_other_chat_uses_no_long_term_memory(self):
        message = FakeTelegramMessage(chat_id=200, text="@bot привет")
        generate = AsyncMock(return_value="привет")

        with (
            patch.object(bot, "call_gemini", generate),
            patch.object(
                self.store,
                "get_memory_context",
                wraps=self.store.get_memory_context,
            ) as get_context,
        ):
            await bot.handle_message(message)

        self.assertEqual(self.store.get_recent_messages("live:200", 10), [])
        get_context.assert_not_called()
        generate.assert_awaited_once()

    def test_request_prompt_orders_memory_before_aggression(self):
        prompt = bot.build_request_system_prompt(
            "MEMORY_SENTINEL",
            "AGGRESSION_SENTINEL",
        )

        self.assertTrue(prompt.startswith(bot.SYSTEM_PROMPT))
        self.assertLess(
            prompt.index("MEMORY_SENTINEL"),
            prompt.index("AGGRESSION_SENTINEL"),
        )

    def test_request_prompt_keeps_question_answer_priority_before_conditional_aggression(self):
        prompt = bot.build_request_system_prompt("", "АГРЕССИЯ")

        self.assertIn("сначала коротко и по существу", prompt.casefold())
        self.assertLess(
            prompt.casefold().index("сначала коротко и по существу"),
            prompt.index("АГРЕССИЯ"),
        )

    def test_question_like_text_recognizes_questions_without_changing_regular_messages(self):
        for text in (
            "Какая погода",
            "Ты можешь помочь",
            "Подскажи время",
            "Это правда?",
            "Памперс как дела",
            "Дима, ответишь",
            "посоветуй фильм",
        ):
            with self.subTest(text=text):
                self.assertTrue(bot.is_question_like(text))

        self.assertFalse(bot.is_question_like("рассказываю анекдот"))

    def test_strip_mention_is_case_insensitive(self):
        self.assertEqual(bot.strip_mention("@BOT как дела"), "как дела")

    async def test_neutral_message_gets_a_fail_neutral_conditional_instruction(self):
        message = FakeTelegramMessage(text="@BOT как дела")
        generate = AsyncMock(return_value="норм")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        request_prompt = generate.await_args.args[1]
        self.assertTrue(request_prompt.startswith(bot.SYSTEM_PROMPT))
        self.assertIn("УСЛОВНЫЙ", request_prompt)
        self.assertIn("Если смысл неоднозначен, не усиливай агрессию", request_prompt)

    async def test_aggression_instruction_failure_falls_back_to_neutral_prompt(self):
        message = FakeTelegramMessage(text="@bot как дела")
        generate = AsyncMock(return_value="норм")

        with (
            patch.object(
                bot,
                "build_conditional_aggression_instruction",
                side_effect=RuntimeError("PRIVATE DETECTOR DETAILS"),
            ),
            patch.object(bot, "call_gemini", generate),
            self.assertLogs("pampers-bot", level="ERROR") as captured,
        ):
            await bot.handle_message(message)

        self.assertEqual(generate.await_args.args[1], bot.SYSTEM_PROMPT)
        self.assertNotIn("PRIVATE DETECTOR DETAILS", "\n".join(captured.output))

    async def test_direct_insult_appends_one_local_arrow_instruction(self):
        message = FakeTelegramMessage(text="@bot ты тупой, закрой рот")
        generate = AsyncMock(return_value="сам мысль сначала собери")
        original_prompt = bot.SYSTEM_PROMPT

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        request_prompt = generate.await_args.args[1]
        self.assertTrue(request_prompt.startswith(original_prompt))
        self.assertIn("ситуационную стрелку", request_prompt)
        self.assertIn("примерно на 30% жёстче", request_prompt)
        self.assertIn("не используй скрытую память", request_prompt.casefold())
        self.assertEqual(generate.await_count, 1)
        self.assertEqual(bot.SYSTEM_PROMPT, original_prompt)

    async def test_aggression_instruction_follows_memory_and_hides_technical_user_key(self):
        unknown = FakeTelegramMessage(
            user_id=808,
            username="new_person",
            first_name="Новый",
            text="@bot ты идиот",
        )
        generate = AsyncMock(return_value="сам разберись")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(unknown)

        request_prompt = generate.await_args.args[1]
        self.assertIn("ситуационную стрелку", request_prompt)
        self.assertNotIn("telegram:808", request_prompt)

    async def test_aggression_instruction_is_after_ready_memory_and_forbids_using_it_as_a_weapon(self):
        vovah_id = self.store.upsert_user("Вовах")
        self._add_ready_episode(
            vovah_id,
            "PRIVATE_HISTORY_SENTINEL обсуждал шахматы",
            9,
        )
        message = FakeTelegramMessage(text="@bot ты тупой")
        generate = AsyncMock(return_value="сначала сам думать научись")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        request_prompt = generate.await_args.args[1]
        self.assertIn("PRIVATE_HISTORY_SENTINEL", request_prompt)
        self.assertLess(
            request_prompt.index("PRIVATE_HISTORY_SENTINEL"),
            request_prompt.index("РЕЖИМ ОТВЕТА НА АДРЕСНУЮ АГРЕССИЮ"),
        )
        self.assertIn("не используй скрытую память", request_prompt.casefold())

    async def test_family_sexual_attack_gets_the_conditional_mirror_boundary(self):
        message = FakeTelegramMessage(
            text="@bot твоя мать шлюха и сексуально обслуживала весь двор"
        )
        generate = AsyncMock(return_value="риторическая встречная стрелка")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        request_prompt = generate.await_args.args[1]
        self.assertIn("примерно на 30% жёстче", request_prompt)
        self.assertIn("семейном или сексуальном оскорблении", request_prompt)
        self.assertIn("без выдачи выдумки за факт", request_prompt)
        self.assertEqual(generate.await_count, 1)

    async def test_memory_instruction_contains_only_the_current_user(self):
        vovah_id = self.store.upsert_user("Вовах")
        maria_id = self.store.upsert_user("Маша")
        self._add_ready_episode(vovah_id, "Вовах раньше обсуждал переезд", 1)
        self._add_ready_episode(maria_id, "Маша раньше обсуждала отпуск", 2)
        message = FakeTelegramMessage(text="@bot переезд")
        generate = AsyncMock(return_value="помню")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        instruction = generate.await_args.args[1]
        self.assertIn("Вовах", instruction)
        self.assertIn("обсуждал переезд", instruction)
        self.assertNotIn("Маша", instruction)
        self.assertNotIn("обсуждала отпуск", instruction)
        self.assertEqual(bot.SYSTEM_PROMPT, bot.build_system_prompt(bot.PERSONA))

    def test_unknown_user_gets_telegram_key_and_current_display_name(self):
        message = FakeTelegramMessage(
            user_id=808,
            username="new_person",
            first_name="Новый",
            last_name="Человек",
            text="привет",
        )

        user_id, message_row_id = bot.persist_incoming_message(message)

        self.assertIsInstance(message_row_id, int)
        context = self.store.get_memory_context(
            user_id, "привет", "live:100", 6, 6
        )
        self.assertEqual(context.canonical_name, "telegram:808")
        self.assertEqual(context.recent_messages[0].author_label, "Новый Человек")

    def test_conflicting_known_identity_is_not_saved_under_the_wrong_person(self):
        first = FakeTelegramMessage(user_id=55, text="первая")
        second = FakeTelegramMessage(user_id=56, text="подмена")

        self.assertIsNotNone(bot.persist_incoming_message(first)[0])
        self.assertEqual(bot.persist_incoming_message(second), (None, None))
        saved = self.store.get_recent_messages("live:100", 10)
        self.assertEqual([item.text for item in saved], ["первая"])

    def test_bound_telegram_id_survives_a_later_username_change(self):
        first = FakeTelegramMessage(user_id=55, text="с известным юзернеймом")
        changed = FakeTelegramMessage(
            message_id=11,
            user_id=55,
            username=None,
            first_name="Новое отображаемое имя",
            text="после смены юзернейма",
        )

        original_user_id = bot.persist_incoming_message(first)[0]
        rebound_user_id = bot.persist_incoming_message(changed)[0]

        self.assertEqual(rebound_user_id, original_user_id)
        context = self.store.get_memory_context(
            original_user_id, "смены", "live:100", 6, 6
        )
        self.assertEqual(context.canonical_name, "Вовах")
        self.assertEqual(
            [item.text for item in context.recent_messages],
            ["с известным юзернеймом", "после смены юзернейма"],
        )

    def test_bound_telegram_id_wins_over_username_mapped_to_another_person(self):
        first = FakeTelegramMessage(user_id=55, text="первая реплика Воваха")
        conflicting = FakeTelegramMessage(
            message_id=12,
            user_id=55,
            username="testuser_b",
            first_name="Подменённое имя",
            text="реплика после конфликтного юзернейма",
        )

        original_user_id = bot.persist_incoming_message(first)[0]
        persisted_user_id, persisted_message_id = bot.persist_incoming_message(
            conflicting
        )

        self.assertEqual(persisted_user_id, original_user_id)
        self.assertIsInstance(persisted_message_id, int)
        context = self.store.get_memory_context(
            original_user_id, "реплика", "live:100", 6, 6
        )
        self.assertEqual(context.canonical_name, "Вовах")
        self.assertEqual(
            [item.text for item in context.recent_messages],
            ["первая реплика Воваха", "реплика после конфликтного юзернейма"],
        )
        with closing(sqlite3.connect(self.store.db_path)) as connection:
            conflicting_alias = connection.execute(
                "SELECT 1 FROM aliases WHERE alias_type = 'username' "
                "AND normalized_value = 'testuser_b'"
            ).fetchone()
        self.assertIsNone(conflicting_alias)

    def test_temporary_user_is_promoted_when_username_becomes_confirmed(self):
        confirmed_archive_user_id = self.store.upsert_user("Вовах")
        unknown = FakeTelegramMessage(
            user_id=808,
            username="temporary_name",
            first_name="Вова",
            text="ранний разговор",
        )
        confirmed = FakeTelegramMessage(
            message_id=11,
            user_id=808,
            username="testuser_a",
            first_name="Вова",
            text="теперь меня узнали",
        )

        temporary_id = bot.persist_incoming_message(unknown)[0]
        promoted_id = bot.persist_incoming_message(confirmed)[0]

        self.assertNotEqual(temporary_id, promoted_id)
        self.assertEqual(promoted_id, confirmed_archive_user_id)
        context = self.store.get_memory_context(
            promoted_id, "разговор", "live:100", 6, 6
        )
        self.assertEqual(context.canonical_name, "Вовах")
        self.assertEqual(
            [item.text for item in context.recent_messages],
            ["ранний разговор", "теперь меня узнали"],
        )
        with closing(sqlite3.connect(self.store.db_path)) as connection:
            temporary_count = connection.execute(
                "SELECT COUNT(*) FROM users WHERE canonical_name = 'telegram:808'"
            ).fetchone()[0]
        self.assertEqual(temporary_count, 0)

    async def test_deliver_returns_confirmed_text_voice_and_sticker_messages(self):
        message = FakeTelegramMessage(text="@bot привет")

        sent_text = await bot.deliver("ответ", reply_to=message, voice=False)
        with patch.object(bot, "generate_voice_ogg", AsyncMock(return_value=b"ogg")):
            sent_voice = await bot.deliver("ответ", reply_to=message, voice=True)
        sent_sticker = await bot.deliver(
            None, reply_to=message, sticker_file_id="sticker-id"
        )

        self.assertEqual(sent_text.kind, "text")
        self.assertEqual(sent_voice.kind, "voice")
        self.assertEqual(sent_sticker.kind, "sticker")

    async def test_deliver_returns_all_three_standalone_bot_sends(self):
        text_message = SimpleNamespace(message_id=1, kind="text")
        voice_message = SimpleNamespace(message_id=2, kind="voice")
        sticker_message = SimpleNamespace(message_id=3, kind="sticker")
        with (
            patch.object(
                bot.bot, "send_message", AsyncMock(return_value=text_message)
            ) as send_message,
            patch.object(
                bot.bot, "send_voice", AsyncMock(return_value=voice_message)
            ) as send_voice,
            patch.object(
                bot.bot, "send_sticker", AsyncMock(return_value=sticker_message)
            ) as send_sticker,
            patch.object(bot, "generate_voice_ogg", AsyncMock(return_value=b"ogg")),
        ):
            delivered_text = await bot.deliver("текст", chat_id=100, voice=False)
            delivered_voice = await bot.deliver("голос", chat_id=100, voice=True)
            delivered_sticker = await bot.deliver(
                None, chat_id=100, sticker_file_id="sticker-id"
            )

        self.assertIs(delivered_text, text_message)
        self.assertIs(delivered_voice, voice_message)
        self.assertIs(delivered_sticker, sticker_message)
        send_message.assert_awaited_once()
        send_voice.assert_awaited_once()
        send_sticker.assert_awaited_once()

    async def test_wrong_chat_title_is_ignored_before_memory_and_response(self):
        message = FakeTelegramMessage(
            chat_title="Совсем другой чат",
            text="@bot привет",
        )
        generate = AsyncMock(return_value="ответ")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        generate.assert_not_awaited()
        message.reply.assert_not_awaited()
        self.assertEqual(self.store.get_recent_messages("live:100", 10), [])

    async def test_approved_name_and_reply_to_bot_both_trigger_responses(self):
        named = FakeTelegramMessage(message_id=20, text="Памперс привет")
        replied = FakeTelegramMessage(
            message_id=21,
            text="продолжай",
            reply_to_message=SimpleNamespace(
                message_id=19,
                from_user=SimpleNamespace(id=bot.BOT_ID),
            ),
        )
        generate = AsyncMock(side_effect=("первый ответ", "второй ответ"))

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(named)
            await bot.handle_message(replied)

        self.assertEqual(generate.await_count, 2)
        named.reply.assert_awaited_once()
        replied.reply.assert_awaited_once()

    async def test_skip_sentinel_saves_only_incoming_and_sends_nothing(self):
        message = FakeTelegramMessage(message_id=25, text="@bot вопрос")

        with patch.object(
            bot, "call_gemini", AsyncMock(return_value=bot.SKIP_SENTINEL)
        ):
            await bot.handle_message(message)

        message.reply.assert_not_awaited()
        self.assertEqual(
            [item.text for item in self.store.get_recent_messages("live:100", 10)],
            ["@bot вопрос"],
        )
        self.assertEqual(self.store.get_episode_status_counts()["pending"], 0)

    async def test_empty_model_reply_cleans_transient_history_and_sends_nothing(self):
        message = FakeTelegramMessage(message_id=26, text="@bot вопрос")

        with patch.object(bot, "call_gemini", AsyncMock(return_value="")):
            await bot.handle_message(message)

        self.assertEqual(list(bot.history[100]), [])
        message.reply.assert_not_awaited()

    async def test_long_first_reply_is_regenerated_once_and_only_valid_reply_is_saved(self):
        message = FakeTelegramMessage(message_id=90, text="@bot ответь")
        snapshots = []

        async def generate(chat_history, system_instruction):
            snapshots.append((list(chat_history), system_instruction))
            if len(snapshots) == 1:
                return (
                    "раз два три четыре пять шесть семь восемь девять десять "
                    "одиннадцать двенадцать"
                )
            return "короче уже некуда"

        with patch.object(bot, "call_gemini", AsyncMock(side_effect=generate)) as call:
            await bot.handle_message(message)

        self.assertEqual(call.await_count, 2)
        self.assertEqual(
            snapshots[1][0],
            [{"role": "user", "content": "ответь", "sender": "Вовах (м)"}],
        )
        self.assertIn("Предыдущий вариант отклонён", snapshots[1][1])
        message.reply.assert_awaited_once_with("короче уже некуда")
        saved = self.store.get_recent_messages("live:100", 10)
        self.assertEqual([item.text for item in saved], ["@bot ответь", "короче уже некуда"])
        self.assertNotIn("двенадцать", " ".join(item.text for item in saved))

    async def test_generic_reaction_and_recent_repetition_are_regenerated(self):
        bot.history[100].append(
            {"role": "assistant", "content": "отвали уже отсюда"}
        )
        first = FakeTelegramMessage(message_id=91, text="@bot ну")
        with patch.object(
            bot,
            "call_gemini",
            AsyncMock(side_effect=("Ты серьёзно?", "новая конкретная мысль")),
        ) as first_call:
            await bot.handle_message(first)
        self.assertEqual(first_call.await_count, 2)
        first.reply.assert_awaited_once_with("новая конкретная мысль")

        second = FakeTelegramMessage(message_id=92, text="@bot ещё")
        with patch.object(
            bot,
            "call_gemini",
            AsyncMock(side_effect=("отвали уже отсюда", "другая короткая реплика")),
        ) as second_call:
            await bot.handle_message(second)
        self.assertEqual(second_call.await_count, 2)
        second.reply.assert_awaited_once_with("другая короткая реплика")

    async def test_two_invalid_variants_send_and_save_no_bot_reply(self):
        message = FakeTelegramMessage(message_id=93, text="@bot ответь")
        generate = AsyncMock(side_effect=("Ты серьёзно?", "И это всё?"))

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        self.assertEqual(generate.await_count, 2)
        message.reply.assert_not_awaited()
        saved = self.store.get_recent_messages("live:100", 10)
        self.assertEqual([item.text for item in saved], ["@bot ответь"])
        self.assertFalse(
            any(item["role"] == "assistant" for item in bot.history[100])
        )

    async def test_failed_retry_is_silent(self):
        retry_message = FakeTelegramMessage(message_id=94, text="@bot ответь")
        with (
            patch.object(
                bot,
                "call_gemini",
                AsyncMock(
                    side_effect=("Ты серьёзно?", RuntimeError("retry failed"))
                ),
            ) as retry_call,
            self.assertLogs("pampers-bot", level="WARNING"),
        ):
            await bot.handle_message(retry_message)

        self.assertEqual(retry_call.await_count, 2)
        retry_message.reply.assert_not_awaited()
        self.assertEqual(list(bot.history[100]), [])

    async def test_first_api_error_keeps_error_reply(self):
        first_error_message = FakeTelegramMessage(message_id=95, text="@bot ответь")
        with (
            patch.object(
                bot,
                "call_gemini",
                AsyncMock(side_effect=RuntimeError("first failed")),
            ) as first_call,
            self.assertLogs("pampers-bot", level="ERROR"),
        ):
            await bot.handle_message(first_error_message)

        self.assertEqual(first_call.await_count, 1)
        first_error_message.reply.assert_awaited_once_with(
            "Что-то сломалось, попробуй ещё раз."
        )
        self.assertEqual(list(bot.history[100]), [])

    def test_recent_bot_replies_uses_only_last_ten_assistant_items_from_argument(self):
        current_chat = []
        for index in range(12):
            current_chat.append({"role": "user", "content": f"вопрос {index}"})
            current_chat.append({"role": "assistant", "content": f"ответ {index}"})
        bot.history[200].append(
            {"role": "assistant", "content": "ответ из другого чата"}
        )

        self.assertEqual(
            bot.recent_bot_replies(current_chat),
            tuple(f"ответ {index}" for index in range(2, 12)),
        )

    async def test_spontaneous_loop_still_delivers_to_known_group(self):
        bot.known_group_chats.add(100)
        delivered = SimpleNamespace(message_id=99, kind="text")
        sleeps = 0

        async def sleep_once(_delay):
            nonlocal sleeps
            sleeps += 1
            if sleeps > 1:
                raise asyncio.CancelledError

        generate = AsyncMock(
            side_effect=(
                "раз два три четыре пять шесть семь восемь девять десять "
                "одиннадцать двенадцать",
                "новая короткая мысль для чата",
            )
        )
        with (
            patch.object(asyncio, "sleep", side_effect=sleep_once),
            patch.object(bot, "call_gemini", generate),
            patch.object(bot, "deliver", AsyncMock(return_value=delivered)) as deliver,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await bot.spontaneous_loop()

        self.assertEqual(generate.await_count, 2)
        deliver.assert_awaited_once_with("новая короткая мысль для чата", chat_id=100)
        spontaneous_prompt = generate.await_args.args[1]
        self.assertNotIn("РЕЖИМ ОТВЕТА НА АДРЕСНУЮ АГРЕССИЮ", spontaneous_prompt)

    async def test_two_invalid_spontaneous_variants_are_not_delivered(self):
        bot.known_group_chats.add(100)
        sleeps = 0

        async def sleep_once(_delay):
            nonlocal sleeps
            sleeps += 1
            if sleeps > 1:
                raise asyncio.CancelledError

        generate = AsyncMock(side_effect=("Ты реально?", "Ну давай"))
        with (
            patch.object(asyncio, "sleep", side_effect=sleep_once),
            patch.object(bot, "call_gemini", generate),
            patch.object(bot, "deliver", AsyncMock()) as deliver,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await bot.spontaneous_loop()

        self.assertEqual(generate.await_count, 2)
        deliver.assert_not_awaited()
        self.assertEqual(list(bot.history[100]), [])

    async def test_spontaneous_reply_replaces_retired_self_names_before_delivery(self):
        bot.known_group_chats.add(100)
        delivered = SimpleNamespace(message_id=99, kind="text")
        sleeps = 0

        async def sleep_once(_delay):
            nonlocal sleeps
            sleeps += 1
            if sleeps > 1:
                raise asyncio.CancelledError

        generate = AsyncMock(return_value="Я Аллан и Дмитрием не называюсь")
        with (
            patch.object(asyncio, "sleep", side_effect=sleep_once),
            patch.object(bot, "call_gemini", generate),
            patch.object(bot, "deliver", AsyncMock(return_value=delivered)) as deliver,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await bot.spontaneous_loop()

        deliver.assert_awaited_once_with("Я Памперс и Дима не называюсь", chat_id=100)

    async def test_addressed_reply_replaces_retired_self_names_before_chat_and_memory(self):
        message = FakeTelegramMessage(message_id=32, text="@bot кто ты")

        with patch.object(
            bot,
            "call_gemini",
            AsyncMock(return_value="Я @example_handle_0000, а не Дмитрий."),
        ):
            await bot.handle_message(message)

        message.reply.assert_awaited_once_with("Я Памперс, а не Дима")
        records = self.store.get_recent_messages("live:100", 10)
        self.assertEqual(records[-1].text, "Я Памперс, а не Дима")

    async def test_bot_reply_and_episode_are_saved_only_after_confirmed_delivery(self):
        delivered = FakeTelegramMessage(message_id=30, text="@bot вопрос")
        generate = AsyncMock(return_value="ответ")
        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(delivered)

        records = self.store.get_recent_messages("live:100", 10)
        self.assertEqual([record.text for record in records], ["@bot вопрос", "ответ"])
        self.assertEqual(records[-1].author_label, "Памперс")
        self.assertEqual(records[-1].reply_to_external_id, 30)
        self.assertEqual(self.store.get_episode_status_counts()["pending"], 1)

        failed = FakeTelegramMessage(message_id=31, text="@bot ещё вопрос")
        failed.reply.side_effect = RuntimeError("telegram unavailable")
        with patch.object(bot, "call_gemini", AsyncMock(return_value="не доставлено")):
            with self.assertRaises(RuntimeError):
                await bot.handle_message(failed)

        records = self.store.get_recent_messages("live:100", 10)
        self.assertNotIn("не доставлено", [record.text for record in records])
        self.assertEqual(self.store.get_episode_status_counts()["pending"], 1)

    def test_busy_chat_episode_keeps_direct_exchange_in_summary_and_search(self):
        incoming = FakeTelegramMessage(
            message_id=70,
            text="@bot прямой вопрос",
        )
        user_id, incoming_row_id = bot.persist_incoming_message(incoming)
        for offset in range(6):
            self.store.store_message(
                MessageRecord(
                    source="live",
                    chat_key="live:100",
                    external_message_id=800 + offset,
                    user_id=None,
                    author_label="Другой",
                    sent_at=incoming.date + timedelta(minutes=offset + 1),
                    text=f"ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_LIVE_{offset}",
                    media_description="",
                    reply_to_external_id=None,
                )
            )
        other_user_id = self.store.upsert_user("Маша")
        self.store.store_message(
            MessageRecord(
                source="live",
                chat_key="live:100",
                external_message_id=900,
                user_id=other_user_id,
                author_label="Памперс2004",
                sent_at=incoming.date + timedelta(minutes=8),
                text="ТРЕТЬЯ_СТОРОНА_ПОД_ИМЕНЕМ_ПАМПЕРСА_LIVE",
                media_description="",
                reply_to_external_id=None,
            )
        )
        sent = SimpleNamespace(
            message_id=1070,
            date=incoming.date + timedelta(seconds=1),
            kind="text",
        )

        episode_id, summary_messages = bot._persist_delivered_reply(
            incoming,
            incoming_row_id,
            user_id,
            "Вовах",
            "прямой ответ",
            sent,
            False,
        )

        self.assertTrue(any("прямой вопрос" in item for item in summary_messages))
        self.assertTrue(any("прямой ответ" in item for item in summary_messages))
        self.assertFalse(
            any("ТРЕТЬЯ_СТОРОНА_СЕКРЕТ_LIVE" in item for item in summary_messages)
        )
        self.assertFalse(
            any(
                "ТРЕТЬЯ_СТОРОНА_ПОД_ИМЕНЕМ_ПАМПЕРСА_LIVE" in item
                for item in summary_messages
            )
        )
        with closing(sqlite3.connect(self.store.db_path)) as connection:
            search_text = connection.execute(
                "SELECT search_text FROM episodes WHERE id = ?", (episode_id,)
            ).fetchone()[0]
            linked = connection.execute(
                "SELECT messages.external_message_id, messages.sent_at "
                "FROM episode_messages JOIN messages "
                "ON messages.id = episode_messages.message_id "
                "WHERE episode_messages.episode_id = ? "
                "ORDER BY episode_messages.position",
                (episode_id,),
            ).fetchall()
        self.assertIn("@bot прямой вопрос", search_text)
        self.assertIn("прямой ответ", search_text)
        self.assertIn(70, [item[0] for item in linked])
        self.assertIn(1070, [item[0] for item in linked])
        self.assertEqual([item[1] for item in linked], sorted(item[1] for item in linked))

    async def test_background_summary_failure_does_not_change_the_sent_reply(self):
        class FailingSummarizer:
            async def summarize_episode(self, canonical_name, messages):
                raise RuntimeError("model unavailable")

        bot.episode_summarizer = FailingSummarizer()
        message = FakeTelegramMessage(message_id=40, text="@bot вопрос")
        with patch.object(bot, "call_gemini", AsyncMock(return_value="доставленный ответ")):
            await bot.handle_message(message)
        await asyncio.gather(*tuple(bot.summary_tasks), return_exceptions=True)

        self.assertEqual(message.reply.await_count, 1)
        self.assertEqual(
            self.store.get_episode_status_counts(),
            {"pending": 0, "ready": 0, "failed": 1},
        )

    async def test_successful_background_summary_makes_the_live_episode_ready(self):
        class SuccessfulSummarizer:
            async def summarize_episode(self, canonical_name, messages):
                self.canonical_name = canonical_name
                self.messages = messages
                return "обсуждали доставленный ответ"

        summarizer = SuccessfulSummarizer()
        bot.episode_summarizer = summarizer
        message = FakeTelegramMessage(message_id=41, text="@bot вопрос")
        with patch.object(bot, "call_gemini", AsyncMock(return_value="ответ")):
            await bot.handle_message(message)
        if bot.summary_tasks:
            await asyncio.gather(*tuple(bot.summary_tasks), return_exceptions=True)

        self.assertEqual(summarizer.canonical_name, "Вовах")
        self.assertTrue(any("ответ" in item for item in summarizer.messages))
        self.assertEqual(
            self.store.get_episode_status_counts(),
            {"pending": 0, "ready": 1, "failed": 0},
        )

    async def test_private_message_keeps_rejection_and_never_enters_memory(self):
        message = FakeTelegramMessage(
            chat_id=100,
            chat_type="private",
            text="@bot привет",
        )

        await bot.handle_message(message)

        message.reply.assert_awaited_once_with(bot.DM_REJECTION_TEXT)
        self.assertEqual(self.store.get_recent_messages("live:100", 10), [])

    async def test_sticker_only_response_creates_no_textual_episode(self):
        bot.sticker_file_ids = ["sticker-id"]
        bot.STICKER_PROBABILITY = 1
        message = FakeTelegramMessage(text="@bot привет")

        with patch.object(bot.random, "random", return_value=0):
            await bot.handle_message(message)

        message.reply_sticker.assert_awaited_once()
        self.assertEqual(self.store.get_episode_status_counts()["pending"], 0)
        saved = self.store.get_recent_messages("live:100", 10)
        self.assertEqual([item.text for item in saved], ["@bot привет"])

    async def test_question_skips_sticker_and_gets_a_text_reply(self):
        bot.sticker_file_ids = ["sticker-id"]
        bot.STICKER_PROBABILITY = 1
        message = FakeTelegramMessage(text="@bot как дела")
        generate = AsyncMock(return_value="у меня всё нормально")

        with (
            patch.object(bot, "call_gemini", generate),
            patch.object(bot, "generate_voice_ogg", AsyncMock(return_value=None)),
            patch.object(bot.random, "random", return_value=0.5),
        ):
            await bot.handle_message(message)

        generate.assert_awaited_once()
        message.reply.assert_awaited_once_with("у меня всё нормально")
        message.reply_sticker.assert_not_awaited()

    async def test_named_question_skips_sticker_and_gets_a_text_reply(self):
        bot.sticker_file_ids = ["sticker-id"]
        bot.STICKER_PROBABILITY = 1
        message = FakeTelegramMessage(text="Памперс как дела")
        generate = AsyncMock(return_value="у меня всё нормально")

        with (
            patch.object(bot, "call_gemini", generate),
            patch.object(bot, "generate_voice_ogg", AsyncMock(return_value=None)),
            patch.object(bot.random, "random", return_value=0.5),
        ):
            await bot.handle_message(message)

        generate.assert_awaited_once()
        message.reply.assert_awaited_once_with("у меня всё нормально")
        message.reply_sticker.assert_not_awaited()

    def test_generated_retired_self_names_are_replaced_before_delivery(self):
        fake_pattern = re.compile(
            r"(?<!\w)@?(?:example_handle_0000|allan(?:'s)?|аллан(?:а|у|ом|е)?)(?!\w)",
            re.IGNORECASE,
        )
        generated = "Я Аллан, Allan, @example_handle_0000 и Дмитрием не называюсь"

        with patch.object(bot, "RETIRED_SELF_NAME_PATTERN", fake_pattern):
            normalized = bot.normalize_bot_self_names(generated)

        self.assertEqual(
            normalized,
            "Я Памперс, Памперс, Памперс и Дима не называюсь",
        )

    def test_system_prompt_uses_only_the_two_approved_self_names(self):
        persona = copy.deepcopy(bot.PERSONA)
        persona["persona_name"] = "Аллан"
        persona["persona_aliases"] = ["Allan"]

        prompt = bot.build_system_prompt(persona).casefold()

        self.assertIn("ты памперс, также тебя называют дима.", prompt)
        self.assertIn("памперс", prompt)
        self.assertIn("дима", prompt)
        self.assertNotIn("аллан", prompt)
        self.assertNotIn("allan", prompt)
        self.assertNotIn("дмитрий", prompt)

    def _add_ready_episode(self, user_id: int, summary: str, number: int) -> None:
        when = datetime(2026, 8, 1, 10, number, tzinfo=timezone.utc)
        episode_id = self.store.store_episode(
            user_id=user_id,
            source="archive",
            started_at=when,
            ended_at=when,
            search_text=summary,
            direct_exchange_count=1,
            message_row_ids=(),
            fingerprint=f"fixture-{number}",
        )
        self.store.mark_episode_ready(episode_id, summary)


if __name__ == "__main__":
    unittest.main()
