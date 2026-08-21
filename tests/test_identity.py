import unittest
from unittest.mock import patch

import identity
from identity import (
    archive_aliases_for,
    canonical_from_archive,
    canonical_from_telegram,
    normalize_alias,
)

FAKE_ARCHIVE_TO_CANONICAL = {
    "alice_display": "Алиса",
    "BOB99": "Боб",
}
FAKE_USERNAME_TO_CANONICAL = {
    "testuser_a": "Боб",
}
FAKE_DISPLAY_TO_CANONICAL = {"мария": "Маша"}


@patch("identity.ARCHIVE_TO_CANONICAL", FAKE_ARCHIVE_TO_CANONICAL)
@patch("identity.USERNAME_TO_CANONICAL", FAKE_USERNAME_TO_CANONICAL)
@patch("identity.DISPLAY_TO_CANONICAL", FAKE_DISPLAY_TO_CANONICAL)
class IdentityTests(unittest.TestCase):
    def test_all_confirmed_archive_names_map_to_their_canonical_users(self):
        actual = {
            name: canonical_from_archive(name) for name in FAKE_ARCHIVE_TO_CANONICAL
        }
        self.assertEqual(actual, FAKE_ARCHIVE_TO_CANONICAL)

    def test_live_username_takes_priority_and_display_name_maps_maria(self):
        self.assertEqual(
            canonical_from_telegram("testuser_a", "Неважно"),
            "Боб",
        )
        self.assertEqual(canonical_from_telegram(None, "Мария"), "Маша")

    def test_unknown_identity_is_not_guessed(self):
        self.assertIsNone(canonical_from_archive("Похожее имя"))
        self.assertIsNone(canonical_from_telegram("new_user", "Алиса2"))

    def test_normalization_is_case_and_space_insensitive(self):
        self.assertEqual(normalize_alias("  BOB99  "), normalize_alias("bob99"))

    def test_archive_alias_lookup_returns_only_the_requested_user_aliases(self):
        self.assertEqual(archive_aliases_for("Боб"), ("BOB99",))
        self.assertEqual(archive_aliases_for("Неизвестный"), ())


if __name__ == "__main__":
    unittest.main()
