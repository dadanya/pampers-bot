import unittest

from identity import (
    archive_aliases_for,
    canonical_from_archive,
    canonical_from_telegram,
    normalize_alias,
)


class IdentityTests(unittest.TestCase):
    def test_all_confirmed_archive_names_map_to_their_canonical_users(self):
        expected = {
            ".": "Арина",
            "tanaka": "Андрей",
            ". Sür Маленький": "Сюр",
            "V0VAH?": "Вовах",
            "Чернобыль": "Чернобыль",
            "ꀘꍏ꓄ꃅꍟꋪꀤꈤꍟ": "Катя",
            "Denis": "Денис",
            "Анастасия": "Настя",
            "fiya zimmerman🎚️": "Фия",
            "Мария": "Маша",
        }

        actual = {name: canonical_from_archive(name) for name in expected}

        self.assertEqual(actual, expected)

    def test_live_username_takes_priority_and_display_name_maps_maria(self):
        self.assertEqual(
            canonical_from_telegram("evilgeniusforever", "Неважно"),
            "Вовах",
        )
        self.assertEqual(canonical_from_telegram(None, "Мария"), "Маша")

    def test_unknown_identity_is_not_guessed(self):
        self.assertIsNone(canonical_from_archive("Похожее имя"))
        self.assertIsNone(canonical_from_telegram("new_user", "Арина2"))

    def test_normalization_is_case_and_space_insensitive(self):
        self.assertEqual(normalize_alias("  V0VAH?  "), normalize_alias("v0vah?"))

    def test_archive_alias_lookup_returns_only_the_requested_user_aliases(self):
        self.assertEqual(archive_aliases_for("Вовах"), ("V0VAH?",))
        self.assertEqual(archive_aliases_for("Неизвестный"), ())


if __name__ == "__main__":
    unittest.main()
