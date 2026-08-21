import unittest
from datetime import datetime, timezone
from pathlib import Path

from archive_parser import parse_export


class ArchiveParserTests(unittest.TestCase):
    def setUp(self):
        self.fixture_dir = (
            Path(__file__).parent / "fixtures" / "telegram_export"
        )

    def test_parses_real_telegram_message_fields(self):
        messages = parse_export(self.fixture_dir)

        self.assertEqual([message.id for message in messages], [1, 2, 3, 4, 5])
        self.assertEqual(messages[1].author, "V0VAH?")
        self.assertEqual(
            messages[2].sent_at,
            datetime(2026, 8, 1, 12, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(messages[2].text, "ответ & реакция")
        self.assertEqual(messages[2].reply_to, 2)
        self.assertEqual(messages[2].page, "messages.html")
        self.assertIn("photos/photo_1.jpg", messages[2].media_description)

    def test_inline_markup_preserves_word_boundaries_and_breaks(self):
        messages = parse_export(self.fixture_dir)

        self.assertEqual(messages[3].text, "невероятно! два слова")


if __name__ == "__main__":
    unittest.main()
