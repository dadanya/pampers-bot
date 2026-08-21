from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_aggression_profile import (
    build_aggression_profile,
    extract_pampers_responses,
    write_aggression_profile,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "aggression_replies.md"
)
PROFILE_PATH = Path(__file__).resolve().parents[1] / "aggression_profile.json"


class AggressionProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    def test_extracts_only_pampers_response_lines(self):
        responses = extract_pampers_responses(self.fixture_text)

        self.assertEqual(
            responses,
            (
                "Рот закрой",
                "Сам свою чушь сначала разгреби, бля",
                "Я просто объяснил почему не отвечал раньше",
            ),
        )
        self.assertNotIn("synthetic incoming insult", responses)

    def test_profile_contains_hand_checked_aggregates_but_no_raw_replies(self):
        responses = extract_pampers_responses(self.fixture_text)

        profile = build_aggression_profile(responses)

        self.assertEqual(profile["source_response_count"], 3)
        self.assertEqual(profile["median_words"], 6)
        self.assertAlmostEqual(profile["up_to_5_words_rate"], 1 / 3)
        self.assertEqual(profile["up_to_10_words_rate"], 1.0)
        self.assertAlmostEqual(profile["profanity_rate"], 1 / 3)
        self.assertAlmostEqual(profile["justification_rate"], 1 / 3)
        self.assertEqual(profile["target_word_range"], [1, 10])

        serialized = json.dumps(profile, ensure_ascii=False)
        for response in responses:
            self.assertNotIn(response, serialized)
        self.assertNotIn("messages", profile)
        self.assertNotIn("replies", profile)

    def test_writer_persists_only_the_aggregate_profile(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "profile.json"

            profile = write_aggression_profile(FIXTURE_PATH, output_path)
            stored = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(stored, profile)
        self.assertEqual(stored["source_response_count"], 3)
        self.assertNotIn("Рот закрой", json.dumps(stored, ensure_ascii=False))

    def test_real_profile_represents_all_responses_without_source_records(self):
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(profile["source_response_count"], 181)
        self.assertEqual(profile["median_words"], 5)
        self.assertAlmostEqual(profile["up_to_5_words_rate"], 0.5635359)
        self.assertAlmostEqual(profile["up_to_10_words_rate"], 0.8232044)
        self.assertAlmostEqual(profile["profanity_rate"], 0.3038674)
        self.assertAlmostEqual(profile["justification_rate"], 0.1160221)

        serialized = json.dumps(profile, ensure_ascii=False).casefold()
        for forbidden in (
            "Памперс2004",
            "Аллан",
            "Allan",
            "telegram",
            "message_id",
            "reply_text",
            "author",
            "date",
        ):
            self.assertNotIn(forbidden.casefold(), serialized)

        forbidden_key_terms = {
            "id",
            "message",
            "reply",
            "text",
            "author",
            "date",
        }

        def walk_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield str(key).casefold()
                    yield from walk_keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk_keys(child)

        for key in walk_keys(profile):
            with self.subTest(key=key):
                self.assertFalse(
                    any(term in key for term in forbidden_key_terms)
                )


if __name__ == "__main__":
    unittest.main()
