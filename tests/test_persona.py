import json
import os
import unittest
from pathlib import Path


os.environ.setdefault(
    "TELEGRAM_BOT_TOKEN",
    "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
)

from bot import build_system_prompt  # noqa: E402


PERSONA_PATH = Path(__file__).resolve().parents[1] / "persona.json"


class PersonaTests(unittest.TestCase):
    def setUp(self):
        self.persona = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))

    def test_persona_configuration_drives_the_approved_primary_identity_prompt(self):
        prompt = build_system_prompt(self.persona)

        self.assertEqual(self.persona["persona_name"], "Памперс")
        self.assertIn("по имени Памперс", prompt)
        self.assertNotIn("Аллан", prompt)
        self.assertNotIn("Allan", prompt)

    def test_persona_alias_schema_has_only_the_approved_conversational_name(self):
        serialized = json.dumps(self.persona, ensure_ascii=False).casefold()

        self.assertEqual(self.persona["persona_aliases"], ["Дима"])
        self.assertNotIn("аллан", serialized)
        self.assertNotIn("allan", serialized)

    def test_production_persona_contains_only_abstract_style_guidance(self):
        prompt = build_system_prompt(self.persona)

        self.assertNotIn("example_reactions", self.persona)
        self.assertNotIn("example_aggressive_reactions", self.persona)
        for raw_example in (
            "Иди нахуй",
            "Закрой рот",
            "Скажи мне это в лицо!",
            "Арина",
            "Вован",
            "Танака",
        ):
            with self.subTest(raw_example=raw_example):
                self.assertNotIn(raw_example.casefold(), prompt.casefold())

    def test_prompt_requires_short_specific_replies_without_generic_openers(self):
        prompt = build_system_prompt(self.persona).casefold()

        self.assertIn("3–11 слов", prompt)
        for generic_opener in (
            "ты серьёзно",
            "ты реально",
            "и это всё",
            "ну давай",
            "опять ты",
        ):
            with self.subTest(generic_opener=generic_opener):
                self.assertIn(generic_opener, prompt)
        self.assertNotIn("1-10 слов", prompt)

    def test_prompt_answers_questions_before_any_optional_humor(self):
        prompt = build_system_prompt(self.persona).casefold()

        self.assertIn("сначала коротко и по существу", prompt)
        self.assertIn("не подменяй ответ шуткой", prompt)
        self.assertIn("если не знаешь", prompt)
        self.assertIn("не выдумывай факт", prompt)


if __name__ == "__main__":
    unittest.main()
