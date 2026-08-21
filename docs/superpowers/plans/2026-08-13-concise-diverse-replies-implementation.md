# Concise Diverse Replies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ограничить ответы Памперса одиннадцатью словами, исключить дежурные реакции недоверия и не допускать полного или почти полного повторения десяти последних реплик.

**Architecture:** Чистый модуль `reply_quality.py` выполняет детерминированную нормализацию и проверку текста без Telegram, Gemini и базы данных. `bot.py` использует один общий генератор для адресованных и самостоятельных реплик: первая недопустимая генерация получает одну попытку переформулировки, а второй недопустимый вариант приводит к молчанию.

**Tech Stack:** Python 3.11, стандартные `re`, `dataclasses` и `difflib`, aiogram, Google Gen AI SDK, `unittest`.

## Global Constraints

- Обычная генерация получает целевой диапазон 3–11 слов.
- Агрессивная генерация получает целевой диапазон 1–11 слов.
- Любой доставляемый текст или текст для TTS содержит от 1 до 11 слов.
- Сравнение выполняется с десятью последними репликами бота в текущем чате; порог `SequenceMatcher` равен `0.82`.
- Разрешена не более чем одна дополнительная генерация Gemini на входящее сообщение.
- Недопустимые варианты не попадают в историю, Telegram, TTS или SQLite.
- Стикерный путь, идентификация пользователей, память, условная агрессия и ограничения безопасности не меняются.
- `E:\pampers-bot` не является Git-репозиторием: вместо невозможных `git commit` каждый Task завершается тестовым контрольным пунктом и списком изменённых файлов.

---

### Task 1: Детерминированный валидатор качества ответа

**Files:**
- Create: `reply_quality.py`
- Create: `tests/test_reply_quality.py`

**Interfaces:**
- Produces: `MAX_REPLY_WORDS: int = 11`
- Produces: `RECENT_REPLY_LIMIT: int = 10`
- Produces: `SIMILARITY_THRESHOLD: float = 0.82`
- Produces: `ReplyValidation(is_valid: bool, reason: str | None)`
- Produces: `count_reply_words(text: str) -> int`
- Produces: `normalize_reply_for_comparison(text: str) -> str`
- Produces: `validate_reply(text: str, recent_replies: Sequence[str]) -> ReplyValidation`

- [ ] **Step 1: Добавить RED-тесты подсчёта и нормализации**

Создать `tests/test_reply_quality.py`:

```python
import unittest

from reply_quality import (
    count_reply_words,
    normalize_reply_for_comparison,
)


class ReplyQualityTests(unittest.TestCase):
    def test_word_count_ignores_punctuation_and_emoji(self):
        self.assertEqual(count_reply_words("Ну... давай 😂 123!"), 3)

    def test_comparison_normalization_folds_case_yo_and_punctuation(self):
        self.assertEqual(
            normalize_reply_for_comparison("  ТЫ, серьЁзно?!  "),
            "ты серьезно",
        )
```

- [ ] **Step 2: Запустить тест и подтвердить RED**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_reply_quality -v
```

Expected: `ModuleNotFoundError: No module named 'reply_quality'`.

- [ ] **Step 3: Добавить RED-тесты запрещённых вступлений, лимита и повторов**

Дополнить `ReplyQualityTests`:

```python
from reply_quality import validate_reply


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
```

- [ ] **Step 4: Реализовать минимальный `reply_quality.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence


MAX_REPLY_WORDS = 11
RECENT_REPLY_LIMIT = 10
SIMILARITY_THRESHOLD = 0.82
_WORD_PATTERN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_GENERIC_OPENINGS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:ну )?ты (?:серьезно|реально)(?: что ли)?(?: |$)",
        r"^(?:ну )?серьезно(?: что ли)?(?: |$)",
        r"^(?:и )?это все(?: |$)",
        r"^ну давай(?: |$)",
        r"^опять ты(?: |$)",
    )
)


@dataclass(frozen=True)
class ReplyValidation:
    is_valid: bool
    reason: str | None = None


def count_reply_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def normalize_reply_for_comparison(text: str) -> str:
    return " ".join(_WORD_PATTERN.findall(text.casefold().replace("ё", "е")))


def validate_reply(
    text: str,
    recent_replies: Sequence[str],
) -> ReplyValidation:
    normalized = normalize_reply_for_comparison(text)
    word_count = count_reply_words(text)
    if not normalized or word_count == 0:
        return ReplyValidation(False, "empty")
    if word_count > MAX_REPLY_WORDS:
        return ReplyValidation(False, "too_long")
    if any(pattern.match(normalized) for pattern in _GENERIC_OPENINGS):
        return ReplyValidation(False, "generic_reaction")
    for recent in recent_replies[-RECENT_REPLY_LIMIT:]:
        recent_normalized = normalize_reply_for_comparison(recent)
        if not recent_normalized:
            continue
        if normalized == recent_normalized:
            return ReplyValidation(False, "repetition")
        if SequenceMatcher(None, normalized, recent_normalized).ratio() >= SIMILARITY_THRESHOLD:
            return ReplyValidation(False, "repetition")
    return ReplyValidation(True)
```

- [ ] **Step 5: Запустить профильные тесты и подтвердить GREEN**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_reply_quality -v
```

Expected: все тесты `OK`.

- [ ] **Step 6: Контрольный пункт Task 1**

Run:

```powershell
.\venv\Scripts\python.exe -m py_compile reply_quality.py tests\test_reply_quality.py
```

Expected: exit code `0`. Зафиксировать изменённые файлы: `reply_quality.py`, `tests/test_reply_quality.py`.

---

### Task 2: Одна повторная генерация для адресованных и самостоятельных ответов

**Files:**
- Modify: `bot.py:170-320`
- Modify: `bot.py:780-890`
- Modify: `persona.json:8-24`
- Modify: `aggression.py:8`
- Modify: `aggression_profile.json:15-18`
- Modify: `tests/test_bot_memory.py`
- Modify: `tests/test_aggression.py`
- Modify: `tests/test_persona.py`

**Interfaces:**
- Consumes: `reply_quality.validate_reply(text, recent_replies)`
- Produces: `recent_bot_replies(chat_history) -> tuple[str, ...]`
- Produces: `prepare_model_reply(text: str) -> str`
- Produces: `generate_validated_reply(chat_history, system_instruction: str) -> str | None`
- Keeps: `call_gemini(chat_history, system_instruction: str) -> str`

- [ ] **Step 1: Добавить RED-тест повторной генерации длинного ответа**

В `tests/test_bot_memory.py` добавить:

```python
    async def test_long_first_reply_is_regenerated_once_and_only_valid_reply_is_saved(self):
        message = FakeTelegramMessage(message_id=90, text="@bot ответь")
        generate = AsyncMock(side_effect=(
            "раз два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать",
            "короче уже некуда",
        ))

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        self.assertEqual(generate.await_count, 2)
        message.reply.assert_awaited_once_with("короче уже некуда")
        saved = self.store.get_recent_messages("live:100", 10)
        self.assertEqual(saved[-1].text, "короче уже некуда")
        self.assertNotIn("двенадцать", " ".join(item.text for item in saved))
        self.assertIn("Предыдущий вариант отклонён", generate.await_args.args[1])
```

- [ ] **Step 2: Добавить RED-тесты дежурной реакции, повтора и двойного отказа**

```python
    async def test_generic_reaction_and_recent_repetition_are_regenerated(self):
        bot.history[100].append({"role": "assistant", "content": "отвали уже отсюда"})
        first = FakeTelegramMessage(message_id=91, text="@bot ну")
        with patch.object(
            bot,
            "call_gemini",
            AsyncMock(side_effect=("Ты серьёзно?", "новая конкретная мысль")),
        ):
            await bot.handle_message(first)
        first.reply.assert_awaited_once_with("новая конкретная мысль")

        second = FakeTelegramMessage(message_id=92, text="@bot ещё")
        with patch.object(
            bot,
            "call_gemini",
            AsyncMock(side_effect=("отвали уже отсюда", "другая короткая реплика")),
        ):
            await bot.handle_message(second)
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
        self.assertFalse(any(item["role"] == "assistant" for item in bot.history[100]))
```

- [ ] **Step 3: Добавить RED-тест общего пути для самостоятельного ответа**

Обновить существующую фикстуру `spontaneous_loop` так, чтобы первая генерация была длиннее 11 слов, вторая — `"новая короткая мысль для чата"`, и проверить:

```python
self.assertEqual(generate.await_count, 2)
deliver.assert_awaited_once_with("новая короткая мысль для чата", chat_id=100)
```

Добавить отдельный тест: если вторая самостоятельная генерация тоже недопустима, `deliver` не вызывается.

- [ ] **Step 4: Запустить новые интеграционные тесты и подтвердить RED**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_bot_memory -v
```

Expected: новые тесты FAIL, потому что `bot.py` доставляет первый вариант и не вызывает Gemini повторно.

- [ ] **Step 5: Подключить валидатор и общую подготовку ответа**

В `bot.py` импортировать:

```python
from reply_quality import RECENT_REPLY_LIMIT, validate_reply
```

Добавить рядом с `call_gemini`:

```python
RETRY_REPLY_INSTRUCTION = (
    "Предыдущий вариант отклонён: {reason}. "
    "Сформулируй совершенно другую конкретную реплику от 1 до 11 слов. "
    "Не начинай с реакции недоверия и не повторяй недавние ответы."
)


def recent_bot_replies(chat_history) -> tuple[str, ...]:
    return tuple(
        item["content"]
        for item in chat_history
        if item.get("role") == "assistant" and item.get("content")
    )[-RECENT_REPLY_LIMIT:]


def prepare_model_reply(text: str) -> str:
    text = normalize_bot_self_names(text)
    return strip_trailing_period(remove_em_dash(text))


async def generate_validated_reply(
    chat_history,
    system_instruction: str,
) -> str | None:
    recent_replies = recent_bot_replies(chat_history)
    rejection_reason = None
    for attempt in range(2):
        instruction = system_instruction
        if attempt == 1:
            instruction += "\n\n" + RETRY_REPLY_INSTRUCTION.format(
                reason=rejection_reason or "нарушение формата"
            )
        try:
            candidate = prepare_model_reply(
                await call_gemini(chat_history, instruction)
            )
        except Exception:
            if attempt == 0:
                raise
            logger.warning("Повторная генерация ответа не удалась")
            return None
        if not candidate or is_skip_response(candidate):
            return None
        validation = validate_reply(candidate, recent_replies)
        if validation.is_valid:
            return apply_random_caps(candidate)
        rejection_reason = validation.reason
    return None
```

- [ ] **Step 6: Перевести оба пути генерации на общий helper**

В `handle_message` заменить прямой `call_gemini` и последующую нормализацию на:

```python
try:
    reply_text = await generate_validated_reply(
        chat_history,
        request_system_prompt,
    )
except Exception:
    logger.exception("Gemini API call failed")
    chat_history.pop()
    await message.reply("Что-то сломалось, попробуй ещё раз.")
    return

if not reply_text:
    chat_history.pop()
    return
chat_history.append({"role": "assistant", "content": reply_text})
```

В `spontaneous_loop` вызвать тот же helper с `SYSTEM_PROMPT + "\n\n" + SPONTANEOUS_PROMPT`, добавлять в историю и доставлять только непустой результат.

- [ ] **Step 7: Уточнить промпты и лимит агрессии**

В `persona.json` заменить `speech_style.sentence_length` на:

```json
"sentence_length": "обычно 3–11 слов и одна конкретная мысль; без пустых вступлений и повторения недавних формулировок"
```

В `build_system_prompt` добавить жёсткое текстовое правило: обычная реплика — 3–11 слов; не начинать с «ты серьёзно», «ты реально», «и это всё», «ну давай», «опять ты» и сходных пустых реакций.

В `aggression_profile.json` установить:

```json
"target_word_range": [1, 11]
```

В `aggression.py` изменить `_DEFAULT_WORD_RANGE = (1, 11)`.

- [ ] **Step 8: Обновить тесты промптов и агрессии**

В `tests/test_aggression.py` заменить ожидание `"1-10 слов"` на `"1-11 слов"`, включая fallback-тест.

В `tests/test_persona.py` добавить проверку, что `build_system_prompt` содержит `3–11 слов`, запрещённые семейства дежурных реакций и не содержит прежний предел `1-10 слов`.

- [ ] **Step 9: Запустить профильные тесты и подтвердить GREEN**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_reply_quality tests.test_aggression tests.test_persona tests.test_bot_memory -v
```

Expected: все тесты `OK`, включая проверку одной повторной генерации, молчания после двух нарушений, Telegram-доставки и SQLite.

- [ ] **Step 10: Контрольный пункт Task 2**

Run:

```powershell
.\venv\Scripts\python.exe -m py_compile reply_quality.py bot.py aggression.py
```

Expected: exit code `0`. Зафиксировать изменённые файлы: `reply_quality.py`, `bot.py`, `persona.json`, `aggression.py`, `aggression_profile.json`, `tests/test_reply_quality.py`, `tests/test_bot_memory.py`, `tests/test_aggression.py`, `tests/test_persona.py`.

---

### Task 3: Полная проверка и безопасный перезапуск

**Files:**
- Verify: `reply_quality.py`
- Verify: `bot.py`
- Verify: `persona.json`
- Verify: `aggression.py`
- Verify: `aggression_profile.json`
- Verify: `memory.db`
- Runtime logs: `bot-<timestamp>.stdout.log`, `bot-<timestamp>.stderr.log`

**Interfaces:**
- Consumes: вся реализация Task 1–2
- Produces: один работающий экземпляр Telegram polling на новой версии

- [ ] **Step 1: Запустить полный набор тестов**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: все тесты `OK`; сетевые Gemini-вызовы отсутствуют.

- [ ] **Step 2: Запустить компиляцию всех рабочих модулей**

Run:

```powershell
.\venv\Scripts\python.exe -m py_compile reply_quality.py bot.py aggression.py identity.py memory.py memory_summaries.py import_history.py startup_check.py
```

Expected: exit code `0`.

- [ ] **Step 3: Выполнить независимое read-only ревью**

Проверить соответствие утверждённой спецификации, особое внимание уделить:

- максимуму 11 слов после нормализации и до TTS;
- отсутствию более одной повторной генерации;
- недоставке и несохранению обоих недопустимых вариантов;
- сравнению только с десятью ответами текущего чата;
- неизменности стикерного пути, памяти, privacy-изоляции и safety-границ.

Critical и Important замечания исправить через отдельный RED→GREEN цикл до перезапуска.

- [ ] **Step 4: Выполнить локальную и сетевую проверку запуска**

Run:

```powershell
.\venv\Scripts\python.exe startup_check.py
.\venv\Scripts\python.exe startup_check.py --online
```

Expected: `allowed_chat_id_valid=1`, `memory_db_ready=1`, `telegram_ok=1`, `gemini_ok=1`, `errors=none`.

- [ ] **Step 5: Остановить только проверенную цепочку старого бота**

Найти процессы, чья командная строка заканчивается на `bot.py`, подтвердить один корневой экземпляр и остановить его родительский и дочерний Python-процессы. Не использовать широкое завершение всех Python-процессов.

- [ ] **Step 6: Запустить новую версию скрыто с отдельными журналами**

Run:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Start-Process -FilePath '.\venv\Scripts\python.exe' `
  -ArgumentList 'bot.py' `
  -WorkingDirectory 'E:\pampers-bot' `
  -WindowStyle Hidden `
  -RedirectStandardOutput ".\bot-$stamp.stdout.log" `
  -RedirectStandardError ".\bot-$stamp.stderr.log"
```

- [ ] **Step 7: Проверить polling и отсутствие ошибок**

Через 10 секунд подтвердить:

- одна корневая цепочка `bot.py`;
- в журнале есть `Start polling` и `Run polling for bot`;
- нет `ERROR`, `Traceback`, `Conflict`, `Unauthorized`, `RESOURCE_EXHAUSTED` или `UNAVAILABLE`;
- `PRAGMA integrity_check` возвращает `ok`.

- [ ] **Step 8: Финальный контрольный пункт**

Сообщить пользователю фактическое число пройденных тестов, результаты онлайн-проверки, PID корневого процесса, путь к журналу и подтверждение: максимум 11 слов, одна повторная генерация, запрещённые дежурные реакции и защита от десяти последних повторов активны.
