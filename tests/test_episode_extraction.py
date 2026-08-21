import unittest
from datetime import datetime, timezone

from archive_parser import ArchiveMessage
from import_history import build_episode_drafts


def msg(mid, author, minute, text, reply_to=None):
    return ArchiveMessage(
        id=mid,
        author=author,
        sent_at=datetime(2026, 8, 1, 12, minute, tzinfo=timezone.utc),
        text=text,
        media_description="",
        reply_to=reply_to,
        page="messages.html",
    )


class EpisodeExtractionTests(unittest.TestCase):
    def test_direct_reply_gets_three_message_window_but_not_late_message(self):
        messages = [
            msg(1, "V0VAH?", 0, "контекст"),
            msg(2, "V0VAH?", 1, "вопрос"),
            msg(3, "Памперс2004", 2, "ответ", reply_to=2),
            msg(4, "V0VAH?", 3, "продолжение"),
            msg(5, "V0VAH?", 30, "другая тема"),
        ]

        episodes = build_episode_drafts(messages)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].canonical_name, "Вовах")
        self.assertEqual(episodes[0].message_ids, (1, 2, 3, 4))

    def test_overlapping_reply_windows_merge(self):
        messages = [
            msg(1, "V0VAH?", 0, "один"),
            msg(2, "Памперс2004", 1, "два", reply_to=1),
            msg(3, "V0VAH?", 2, "три", reply_to=2),
        ]

        episodes = build_episode_drafts(messages)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].direct_exchange_count, 2)

    def test_same_user_window_merges_with_any_prior_overlap(self):
        messages = [
            msg(1, "V0VAH?", 0, "общий родитель"),
            msg(2, "tanaka", 1, "разделитель"),
            msg(3, "Памперс2004", 2, "первый ответ", reply_to=1),
            msg(4, "V0VAH?", 3, "отдельный родитель"),
            msg(5, "Памперс2004", 4, "отдельный ответ", reply_to=4),
            msg(6, "tanaka", 5, "ещё разделитель"),
            msg(7, "Памперс2004", 6, "второй ответ", reply_to=1),
        ]

        episodes = build_episode_drafts(messages, window_size=0)

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].message_ids, (1, 3, 7))
        self.assertEqual(episodes[0].direct_exchange_count, 2)
        self.assertEqual(episodes[1].message_ids, (4, 5))

    def test_each_anchor_includes_only_three_indices_on_each_side(self):
        messages = [msg(mid, "V0VAH?", mid - 1, str(mid)) for mid in range(1, 9)]
        messages[4] = msg(5, "Памперс2004", 4, "ответ", reply_to=4)

        episodes = build_episode_drafts(messages)

        self.assertEqual(episodes[0].message_ids, (1, 2, 3, 4, 5, 6, 7, 8))

    def test_fifteen_minute_boundary_is_inclusive(self):
        messages = [
            msg(1, "V0VAH?", 0, "вопрос"),
            msg(2, "Памперс2004", 15, "ответ", reply_to=1),
            msg(3, "V0VAH?", 30, "ровно на границе"),
            msg(4, "V0VAH?", 31, "за границей"),
        ]

        episodes = build_episode_drafts(messages)

        self.assertEqual(episodes[0].message_ids, (1, 2, 3))

    def test_reply_and_distant_parent_are_both_anchors_with_bounded_context(self):
        messages = [
            msg(1, "V0VAH?", 0, "исходный вопрос"),
            msg(2, "V0VAH?", 1, "контекст вопроса 1"),
            msg(3, "V0VAH?", 2, "контекст вопроса 2"),
            msg(4, "tanaka", 30, "несвязанный разрыв"),
            msg(5, "V0VAH?", 58, "контекст ответа до"),
            msg(6, "Памперс2004", 59, "поздний ответ", reply_to=1),
            msg(7, "V0VAH?", 59, "контекст ответа после"),
        ]

        episodes = build_episode_drafts(messages)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].message_ids, (1, 2, 3, 5, 6, 7))

    def test_only_exact_bot_author_and_confirmed_archive_user_form_anchor(self):
        messages = [
            msg(1, "V0VAH?", 0, "вопрос"),
            msg(2, "Памперс", 1, "не точное имя", reply_to=1),
            msg(3, "Неизвестный", 2, "другой вопрос"),
            msg(4, "Памперс2004", 3, "неизвестный адресат", reply_to=3),
        ]

        self.assertEqual(build_episode_drafts(messages), [])

    def test_overlapping_windows_for_different_users_stay_separate(self):
        messages = [
            msg(1, "V0VAH?", 0, "вопрос Воваха"),
            msg(2, "Памперс2004", 1, "ответ Воваху", reply_to=1),
            msg(3, "tanaka", 2, "вопрос Андрея"),
            msg(4, "Памперс2004", 3, "ответ Андрею", reply_to=3),
        ]

        episodes = build_episode_drafts(messages)

        self.assertEqual(
            [episode.canonical_name for episode in episodes],
            ["Вовах", "Андрей"],
        )
        self.assertEqual(
            [episode.direct_exchange_count for episode in episodes],
            [1, 1],
        )


if __name__ == "__main__":
    unittest.main()
