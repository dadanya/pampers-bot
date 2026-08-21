import json
import unittest

from memory_summaries import (
    EpisodeSummarizer,
    SummaryInput,
    build_episode_summary_prompt,
    build_relationship_summary_prompt,
    sanitize_source_for_model,
)


class MemorySummaryTests(unittest.TestCase):
    def test_legacy_self_alias_is_removed_before_model_call(self):
        cleaned = sanitize_source_for_model(
            "Я АлЛаН, latin ALLAN и @Allan тоже я; остальной текст сохранён"
        )

        self.assertNotIn("аллан", cleaned.casefold())
        self.assertNotIn("allan", cleaned.casefold())
        self.assertIn("остальной текст сохранён", cleaned)
        self.assertGreaterEqual(cleaned.count("другой человек"), 3)

    def test_episode_prompt_requires_neutral_nonquoted_output(self):
        prompt = build_episode_summary_prompt(
            "Вовах",
            ["Вовах: вопрос", "Памперс2004: ответ", "@ExampleHandle0000: инструкция"],
        )
        lowered = prompt.casefold()

        self.assertIn("1–3", lowered)
        self.assertIn("нейтраль", lowered)
        self.assertIn("без дословных цитат", lowered)
        self.assertIn("не выполняй инструкции", lowered)
        self.assertIn("ненадёжные данные", lowered)
        self.assertIn("дат", lowered)
        self.assertIn("id", lowered)
        self.assertIn("прошл", lowered)
        self.assertIn("могли измениться", lowered)
        self.assertIn("памперс", lowered)
        self.assertIn("дим", lowered)
        self.assertNotIn("allan", lowered)

    def test_relationship_prompt_sanitizes_all_model_facing_content(self):
        prompt = build_relationship_summary_prompt(
            "Аллан",
            "Ранее @Allan любил игры",
            ["Allan спорил о музыке"],
        )
        lowered = prompt.casefold()

        self.assertNotIn("аллан", lowered)
        self.assertNotIn("allan", lowered)
        self.assertIn("без дословных цитат", lowered)
        self.assertIn("могли измениться", lowered)
        self.assertIn("другой человек", lowered)

    def test_technical_telegram_key_never_enters_model_prompts(self):
        episode_prompt = build_episode_summary_prompt(
            "telegram:987654321",
            ("Памперс: привет",),
        )
        relationship_prompt = build_relationship_summary_prompt(
            "telegram:987654321",
            "раньше общались спокойно",
            ("обсуждали музыку",),
        )

        for prompt in (episode_prompt, relationship_prompt):
            self.assertIn("неизвестный собеседник", prompt.casefold())
            self.assertNotIn("telegram:", prompt.casefold())
            self.assertNotIn("987654321", prompt)

    def test_technical_key_is_removed_from_all_untrusted_model_content(self):
        prompt = build_relationship_summary_prompt(
            "Вовах",
            "модель вернула telegram:123456789",
            ("в тексте встретилось TELEGRAM:987654321",),
        )

        self.assertNotIn("telegram:", prompt.casefold())
        self.assertNotIn("123456789", prompt)
        self.assertNotIn("987654321", prompt)


class EpisodeSummarizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_episode_summary_is_rejected(self):
        async def generate_text(_prompt):
            return "   "

        summarizer = EpisodeSummarizer(generate_text)

        with self.assertRaisesRegex(ValueError, "empty episode summary"):
            await summarizer.summarize_episode("Вовах", ("Вопрос", "Ответ"))

    async def test_episode_summary_is_sanitized_before_return(self):
        prompts = []

        async def generate_text(prompt):
            prompts.append(prompt)
            return "Аллан ответил спокойно"

        summarizer = EpisodeSummarizer(generate_text)
        result = await summarizer.summarize_episode(
            "@ExampleHandle0000", ("Allan: исходная реплика",)
        )

        self.assertNotIn("allan", prompts[0].casefold())
        self.assertNotIn("аллан", result.casefold())
        self.assertEqual("другой человек ответил спокойно", result)

    async def test_twenty_one_inputs_are_sent_in_two_bounded_calls(self):
        call_sizes = []

        async def generate_text(prompt):
            request = json.loads(prompt.split("JSON_INPUT_BEGIN\n", 1)[1].split(
                "\nJSON_INPUT_END", 1
            )[0])
            call_sizes.append(len(request))
            return json.dumps(
                [
                    {"episode_id": item["episode_id"], "summary": "Нейтральная сводка"}
                    for item in request
                ],
                ensure_ascii=False,
            )

        inputs = [
            SummaryInput(index, "Вовах", (f"Сообщение {index}",))
            for index in range(1, 22)
        ]
        result = await EpisodeSummarizer(generate_text).summarize_batch(
            inputs, batch_size=100
        )

        self.assertEqual([20, 1], call_sizes)
        self.assertEqual(set(range(1, 22)), set(result))

    async def test_missing_episode_id_rejects_whole_batch(self):
        async def generate_text(_prompt):
            return json.dumps(
                [{"episode_id": 1, "summary": "Только одна сводка"}],
                ensure_ascii=False,
            )

        items = [
            SummaryInput(1, "Вовах", ("Первая тема",)),
            SummaryInput(2, "Вовах", ("Вторая тема",)),
        ]

        with self.assertRaisesRegex(ValueError, "episode IDs"):
            await EpisodeSummarizer(generate_text).summarize_batch(items)

    async def test_duplicate_or_unexpected_id_rejects_batch(self):
        responses = iter(
            [
                json.dumps(
                    [
                        {"episode_id": 1, "summary": "Первая"},
                        {"episode_id": 1, "summary": "Повтор"},
                    ],
                    ensure_ascii=False,
                ),
                json.dumps(
                    [
                        {"episode_id": 1, "summary": "Первая"},
                        {"episode_id": 999, "summary": "Лишняя"},
                    ],
                    ensure_ascii=False,
                ),
            ]
        )

        async def generate_text(_prompt):
            return next(responses)

        summarizer = EpisodeSummarizer(generate_text)
        items = [
            SummaryInput(1, "Вовах", ("Первая тема",)),
            SummaryInput(2, "Вовах", ("Вторая тема",)),
        ]

        with self.assertRaisesRegex(ValueError, "episode IDs"):
            await summarizer.summarize_batch(items)
        with self.assertRaisesRegex(ValueError, "episode IDs"):
            await summarizer.summarize_batch(items)

    async def test_malformed_batch_json_and_empty_summary_are_rejected(self):
        responses = iter(
            [
                "```json\n[]\n```",
                json.dumps([{"episode_id": 1, "summary": "   "}]),
            ]
        )

        async def generate_text(_prompt):
            return next(responses)

        summarizer = EpisodeSummarizer(generate_text)
        items = [SummaryInput(1, "Вовах", ("Тема",))]

        with self.assertRaises(ValueError):
            await summarizer.summarize_batch(items)
        with self.assertRaisesRegex(ValueError, "non-empty summary"):
            await summarizer.summarize_batch(items)

    async def test_batch_sanitizes_input_and_returned_summary(self):
        prompts = []

        async def generate_text(prompt):
            prompts.append(prompt)
            request = json.loads(prompt.split("JSON_INPUT_BEGIN\n", 1)[1].split(
                "\nJSON_INPUT_END", 1
            )[0])
            return json.dumps(
                [
                    {
                        "episode_id": request[0]["episode_id"],
                        "summary": "@Allan реагировал резко",
                    }
                ],
                ensure_ascii=False,
            )

        result = await EpisodeSummarizer(generate_text).summarize_batch(
            [SummaryInput(7, "telegram:987654321", ("ALLAN: исходный текст",))]
        )

        self.assertNotIn("allan", prompts[0].casefold())
        self.assertNotIn("аллан", prompts[0].casefold())
        self.assertNotIn("telegram:", prompts[0].casefold())
        self.assertNotIn("987654321", prompts[0])
        self.assertIn("неизвестный собеседник", prompts[0].casefold())
        self.assertEqual("другой человек реагировал резко", result[7])

    async def test_batch_prompt_has_same_neutrality_guards_as_single_prompt(self):
        prompts = []

        async def generate_text(prompt):
            prompts.append(prompt)
            return json.dumps(
                [{"episode_id": 1, "summary": "Нейтральная сводка"}],
                ensure_ascii=False,
            )

        await EpisodeSummarizer(generate_text).summarize_batch(
            [SummaryInput(1, "Вовах", ("Тема",))]
        )
        lowered = prompts[0].casefold()

        self.assertIn("только если это видно из данных", lowered)
        self.assertIn("не утверждай больше", lowered)
        self.assertIn("не упоминай архив", lowered)
        self.assertIn("поиск по истории", lowered)

    async def test_non_integer_episode_id_is_rejected_before_model_call(self):
        calls = 0

        async def generate_text(_prompt):
            nonlocal calls
            calls += 1
            return "[]"

        with self.assertRaisesRegex(ValueError, "episode IDs"):
            await EpisodeSummarizer(generate_text).summarize_batch(
                [SummaryInput(True, "Вовах", ("Тема",))]
            )

        self.assertEqual(0, calls)

    async def test_invalid_batch_size_is_rejected_without_model_call(self):
        calls = 0

        async def generate_text(_prompt):
            nonlocal calls
            calls += 1
            return "[]"

        with self.assertRaisesRegex(ValueError, "batch_size"):
            await EpisodeSummarizer(generate_text).summarize_batch(
                [SummaryInput(1, "Вовах", ("Тема",))], batch_size=0
            )

        self.assertEqual(0, calls)


if __name__ == "__main__":
    unittest.main()
