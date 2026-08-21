# Answer Questions Before Humor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заставить Памперса сначала кратко отвечать на адресованный вопрос по существу, разрешая юмор только после понятного ответа.

**Architecture:** Семантический смысл вопроса остаётся внутри единственного Gemini-запроса, как и существующее различение цитат и прямой агрессии. `build_system_prompt()` добавляет правило приоритета ответа; `persona.json` перестаёт описывать юмор как замену содержательного ответа. Узкая локальная `is_question_like()` распознаёт `?` и частые вопросительные начала только для пропуска стикера до Gemini; она не выбирает тон, не добавляет API-вызовов и не изменяет доставку текста.

**Tech Stack:** Python 3.11, JSON, unittest, Google Gen AI SDK.

## Global Constraints

- Вопрос получает короткий понятный ответ на конкретный смысл до юмора, сарказма и абсурда.
- Если надёжного ответа нет, бот говорит об этом прямо и не выдумывает факт.
- Юмор разрешён только после ответа и не должен менять либо скрывать его смысл.
- Обычные ответы остаются 3–11 слов, адресная агрессия — 1–11 слов.
- Распознанный вопрос пропускает стикерный путь и получает текстовую генерацию; обычные адресованные сообщения сохраняют прежнюю вероятность стикера.
- Условный режим прямой адресной агрессии, память, приватность, идентификация, голос, повторная генерация и максимум в 11 слов не меняются.
- `E:\pampers-bot` не является Git-репозиторием: контрольные точки фиксируются тестами и списком изменённых файлов без коммита.

---

### Task 1: Промпт приоритета ответа

**Files:**
- Modify: `persona.json:8-24`
- Modify: `bot.py:205-236`
- Modify: `bot.py:412-415`
- Modify: `bot.py:810-818`
- Modify: `tests/test_persona.py:53-66`
- Modify: `tests/test_bot_memory.py:228-250`

**Interfaces:**
- Consumes: `PERSONA`, `build_system_prompt(persona: dict) -> str`, `build_request_system_prompt(memory_instruction: str, aggression_instruction: str) -> str`.
- Produces: system prompt, который содержит правило прямого ответа на вопрос перед юмором; `is_question_like(text: str) -> bool` блокирует стикер только для распознанного вопроса.

- [ ] **Step 1: Добавить RED-тест правила в системный промпт**

В `tests/test_persona.py` добавить:

```python
    def test_prompt_answers_questions_before_any_optional_humor(self):
        prompt = build_system_prompt(self.persona).casefold()

        self.assertIn("сначала коротко и по существу", prompt)
        self.assertIn("не подменяй ответ шуткой", prompt)
        self.assertIn("если не знаешь", prompt)
        self.assertIn("не выдумывай факт", prompt)
```

В `tests/test_bot_memory.py` добавить:

```python
    def test_request_prompt_keeps_question_answer_priority_before_conditional_aggression(self):
        prompt = bot.build_request_system_prompt("", "АГРЕССИЯ")

        self.assertIn("сначала коротко и по существу", prompt.casefold())
        self.assertLess(
            prompt.index("сначала коротко и по существу"),
            prompt.index("АГРЕССИЯ"),
        )
```

Добавить отдельные проверки:

```python
    def test_question_like_text_recognizes_questions_without_changing_regular_messages(self):
        for text in ("Какая погода", "Ты можешь помочь", "Подскажи время", "Это правда?"):
            with self.subTest(text=text):
                self.assertTrue(bot.is_question_like(text))
        self.assertFalse(bot.is_question_like("рассказываю анекдот"))

    async def test_question_skips_sticker_and_gets_a_text_reply(self):
        bot.sticker_file_ids = ["sticker-id"]
        bot.STICKER_PROBABILITY = 1
        message = FakeTelegramMessage(text="@bot как дела")
        generate = AsyncMock(return_value="у меня всё нормально")

        with patch.object(bot, "call_gemini", generate):
            await bot.handle_message(message)

        generate.assert_awaited_once()
        message.reply.assert_awaited_once_with("у меня всё нормально")
        message.reply_sticker.assert_not_awaited()
```

- [ ] **Step 2: Запустить новые тесты и подтвердить RED**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest \
  tests.test_persona.PersonaTests.test_prompt_answers_questions_before_any_optional_humor \
  tests.test_bot_memory.BotMemoryTests.test_request_prompt_keeps_question_answer_priority_before_conditional_aggression -v
```

Expected: FAIL, because the current prompt has no exact answer-before-humor rule and the current sticker branch returns before Gemini for a question.

- [ ] **Step 3: Добавить минимальное правило в промпт и стиль**

В `persona.json` изменить `summary` так, чтобы юмор описывался как дополнение к ответу, а не замена смысла:

```json
"summary": "Дерзкий, прямой и саркастичный собеседник. На вопросы сначала отвечает коротко и по существу; абсурдный юмор использует только как дополнение, а не вместо ответа. Обычно пишет коротко. За публичной бравадой сохраняет потребность в близости, поддержке и лояльности."
```

В `build_system_prompt()` добавить после правила длины:

```python
        "Если тебе задали вопрос, сначала коротко и по существу ответь именно на него. "
        "Не подменяй ответ шуткой, абсурдом или сарказмом. Юмор допустим только после "
        "понятного ответа и не должен менять его смысл. Если не знаешь ответа, скажи это прямо "
        "и не выдумывай факт.",
```

Не менять `build_request_system_prompt`, `handle_message`, `generate_validated_reply`, `deliver`, `aggression.py`, память или голос.

Добавить рядом с `strip_mention`:

```python
QUESTION_START_PATTERN = re.compile(
    r"^\\s*(?:как|кто|что|где|когда|почему|зачем|сколько|какой|какая|какие|"
    r"чей|чья|чьё|чьи|можно|можешь|можете|могу|будешь|будете|знаешь|знаете|"
    r"помнишь|помните|подскажи|расскажи|объясни)\\b|"
    r"^\\s*(?:ты|вы)\\s+(?:можешь|можете|будешь|будете|знаешь|знаете|"
    r"считаешь|считаете)\\b",
    re.IGNORECASE,
)


def is_question_like(text: str) -> bool:
    return "?" in text or bool(QUESTION_START_PATTERN.search(text))
```

В `handle_message()` заменить условие стикера на:

```python
    if (
        sticker_file_ids
        and roll < STICKER_PROBABILITY
        and not is_question_like(user_text)
    ):
```

Не менять `build_request_system_prompt`, `generate_validated_reply`, `deliver`, `aggression.py`, память или голос.

- [ ] **Step 4: Запустить новые тесты и подтвердить GREEN**

Run the Step 2 command.

Expected: both tests pass.

- [ ] **Step 5: Запустить связанные проверки**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_persona tests.test_bot_memory tests.test_aggression tests.test_reply_quality -v
.\venv\Scripts\python.exe -m py_compile bot.py
```

Expected: all focused tests and compilation pass. Record changed files: `persona.json`, `bot.py`, `tests/test_persona.py`, `tests/test_bot_memory.py`.

---

### Task 2: Полная проверка и безопасное обновление процесса

**Files:**
- Verify: `persona.json`, `bot.py`, `tests/test_persona.py`, `tests/test_bot_memory.py`
- Runtime logs: `bot-<timestamp>.stdout.log`, `bot-<timestamp>.stderr.log`

**Interfaces:**
- Consumes: Task 1 prompt rule and existing bot startup procedure.
- Produces: one running bot process using the updated system prompt.

- [ ] **Step 1: Run the full test suite**

Run:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: all tests pass without network calls.

- [ ] **Step 2: Perform local and online startup checks**

Run:

```powershell
.\venv\Scripts\python.exe startup_check.py
.\venv\Scripts\python.exe startup_check.py --online
```

Expected: `allowed_chat_id_valid=1`, `memory_db_ready=1`, `telegram_ok=1`, `gemini_ok=1`, `errors=none`.

- [ ] **Step 3: Obtain a read-only review**

Review that the new rule: answers questions first; does not remove direct-aggression protections; retains privacy, word limit, skip/deflection handling and only one retry. Any Critical or Important finding must be fixed in a new RED→GREEN cycle before restart.

- [ ] **Step 4: Replace only the verified bot process chain**

Find Python processes whose command line matches `bot.py`; confirm one parent and its child. Stop only that pair, then launch `E:\pampers-bot\venv\Scripts\python.exe bot.py` hidden with new timestamped stdout/stderr logs.

Expected: no unrelated Python process is stopped.

- [ ] **Step 5: Verify the new process**

After polling begins, verify one bot parent/child chain, `Start polling`, `Run polling for bot`, no `ERROR`, `Traceback`, `Conflict`, `Unauthorized`, `RESOURCE_EXHAUSTED` or `UNAVAILABLE`, and `PRAGMA integrity_check = ok`.
