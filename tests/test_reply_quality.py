import unittest

from reply_quality import (
    count_reply_words,
    normalize_reply_for_comparison,
    validate_reply,
)


class ReplyQualityTests(unittest.TestCase):
    def test_word_count_ignores_punctuation_and_emoji(self):
        self.assertEqual(count_reply_words("Ну... давай 😂 123!"), 3)

    def test_comparison_normalization_folds_case_yo_and_punctuation(self):
        self.assertEqual(
            normalize_reply_for_comparison("  ТЫ, серьЁзно?!  "),
            "ты серьезно",
        )

    def test_rejects_more_than_eleven_words(self):
        result = validate_reply(
            "раз два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать",
            (),
        )
        self.assertEqual((result.is_valid, result.reason), (False, "too_long"))

    def test_rejects_generic_reaction_families(self):
        samples = (
            "Ты серьёзно?",
            "СЕРЬЕЗНО?!",
            "Ты реально, опять начинаешь",
            "И это всё?",
            "Ну давай, рассказывай",
            "Опять ты со своим бредом",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(validate_reply(sample, ()).reason, "generic_reaction")

    def test_rejects_standalone_realno_reaction_family(self):
        samples = (
            "реально?",
            "РЕАЛЬНО?!",
            "ну реально, это твой ответ",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(validate_reply(sample, ()).reason, "generic_reaction")

    def test_allows_same_words_inside_a_specific_statement(self):
        result = validate_reply("Он спросил, ты серьёзно это сказал", ())
        self.assertTrue(result.is_valid)

    def test_rejects_exact_and_near_recent_replies(self):
        recent = ("иди отсюда и не позорься",)
        self.assertEqual(
            validate_reply("Иди отсюда и не позорься!", recent).reason,
            "repetition",
        )
        self.assertEqual(
            validate_reply("иди отсюда, не позорься", recent).reason,
            "repetition",
        )

    def test_only_ten_most_recent_bot_replies_are_compared(self):
        recent = ("исчезни уже отсюда",) + tuple(
            f"уникальная недавняя реплика номер {number}" for number in range(10)
        )
        self.assertTrue(validate_reply("исчезни уже отсюда", recent).is_valid)
        self.assertEqual(validate_reply(recent[1], recent).reason, "repetition")

    def test_accepts_exactly_eleven_words(self):
        result = validate_reply(
            "раз два три четыре пять шесть семь восемь девять десять одиннадцать",
            (),
        )
        self.assertTrue(result.is_valid)

    def test_similarity_threshold_includes_exactly_zero_point_eighty_two(self):
        candidate = "a" * 41 + "b" * 9
        recent = "a" * 41 + "c" * 9

        self.assertEqual(validate_reply(candidate, (recent,)).reason, "repetition")
