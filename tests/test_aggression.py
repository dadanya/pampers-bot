from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aggression
from aggression import build_conditional_aggression_instruction


class ConditionalAggressionInstructionTests(unittest.TestCase):
    def test_instruction_uses_one_semantic_condition_for_direct_aggression(self):
        instruction = build_conditional_aggression_instruction()
        lowered = instruction.casefold()

        self.assertIn("только если", lowered)
        self.assertIn("именно тебя", lowered)
        self.assertIn("памперса/диму", lowered)
        self.assertIn("примерно на 30% жёстче", lowered)
        self.assertIn("ситуационную стрелку", lowered)
        self.assertIn("1-11 слов", lowered)
        self.assertIn("не оправдывайся", lowered)

    def test_instruction_keeps_reported_and_object_language_neutral(self):
        lowered = build_conditional_aggression_instruction().casefold()

        for neutral_context in (
            "цитат",
            "пересказа чужих слов",
            "обсуждения оскорбительных слов",
            "кода",
            "фильма",
            "третьего лица",
            "самоиронии",
            "нейтрального вопроса",
        ):
            with self.subTest(neutral_context=neutral_context):
                self.assertIn(neutral_context, lowered)
        self.assertIn("если смысл неоднозначен, не усиливай агрессию", lowered)

    def test_instruction_has_privacy_and_real_threat_boundaries(self):
        lowered = build_conditional_aggression_instruction().casefold()

        self.assertIn("не используй скрытую память", lowered)
        self.assertIn("не выдумывай личные факты", lowered)
        self.assertIn("защищённым признакам", lowered)
        self.assertIn("не создавай правдоподобных угроз", lowered)
        self.assertIn("явно абсурдный, невыполнимый", lowered)
        self.assertIn("без намерения, цели, места, времени или способа", lowered)

    def test_invalid_profile_falls_back_to_bounded_short_reply(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_profile = Path(temp_dir) / "profile.json"
            invalid_profile.write_text("{}", encoding="utf-8")
            with patch.object(aggression, "_PROFILE_PATH", invalid_profile):
                instruction = build_conditional_aggression_instruction()

        self.assertIn("1-11 слов", instruction)


if __name__ == "__main__":
    unittest.main()
