import asyncio
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

from aggression import build_conditional_aggression_instruction
from identity import canonical_from_telegram, load_sender_aliases_raw, normalize_alias
from memory import MemoryContext, MemoryStore, MessageRecord, render_memory_instruction
from memory_summaries import (
    EpisodeSummarizer,
    update_relationship_summary_blocks,
)
from reply_quality import RECENT_REPLY_LIMIT, validate_reply
from voice import generate_voice_ogg


def parse_memory_limit(value: str | None, default: int) -> int:
    if default <= 0:
        raise ValueError("default memory limit must be positive")
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
SPONTANEOUS_MIN_MINUTES = float(os.environ.get("SPONTANEOUS_MIN_MINUTES", 30))
SPONTANEOUS_MAX_MINUTES = float(os.environ.get("SPONTANEOUS_MAX_MINUTES", 180))
STICKER_PROBABILITY = float(os.environ.get("STICKER_PROBABILITY", 0.20))
VOICE_REPLY_PROBABILITY = float(os.environ.get("VOICE_REPLY_PROBABILITY", 0.30))
STICKER_SET_NAME = os.environ.get("STICKER_SET_NAME", "PampersaStikery")
MEMORY_DB_PATH = Path(
    os.environ.get("MEMORY_DB_PATH", os.path.join(os.path.dirname(__file__), "memory.db"))
)
MEMORY_MAX_EPISODES = parse_memory_limit(
    os.environ.get("MEMORY_MAX_EPISODES"), 6
)
MEMORY_CONTEXT_MESSAGES = parse_memory_limit(
    os.environ.get("MEMORY_CONTEXT_MESSAGES"), 6
)
ALLOWED_CHAT_TITLE = os.environ.get("ALLOWED_CHAT_TITLE", "Чат Одесского маньяка")
DM_REJECTION_TEXT = "В лс не общаюсь, пиши в общий чат."
SKIP_SENTINEL = "ПРОПУСК"
DEFLECTION_PATTERNS = [
    "сменим тему", "смени тему", "другую тему", "другой теме",
    "не буду обсуждать", "не хочу обсуждать", "не готов обсуждать",
    "поговорим о чём-то другом", "поговорим о другом",
    "пожалуйста, останов", "давай остановимся", "не могу продолжать этот разговор",
]


def is_skip_response(text: str) -> bool:
    if SKIP_SENTINEL in text.strip().upper():
        return True
    lowered = text.lower()
    return any(p in lowered for p in DEFLECTION_PATTERNS)


def strip_trailing_period(text: str) -> str:
    text = text.rstrip()
    if text.endswith("...") or text.endswith("…"):
        return text
    if text.endswith("."):
        return text[:-1].rstrip()
    return text


def remove_em_dash(text: str) -> str:
    text = text.replace("—", " ").replace("–", " ")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _load_retired_handle_fragment() -> str:
    data = load_sender_aliases_raw()
    handles = [re.escape(h) for h in data.get("retired_self_handles", [])]
    return "|".join(handles) + "|" if handles else ""


RETIRED_SELF_NAME_PATTERN = re.compile(
    r"(?<!\w)@?(?:" + _load_retired_handle_fragment() + r"allan(?:'s)?|аллан(?:а|у|ом|е)?)(?!\w)",
    re.IGNORECASE,
)
FORMAL_SELF_NAME_PATTERN = re.compile(
    r"(?<!\w)дмитри(?:й|я|ю|ем|и)(?!\w)",
    re.IGNORECASE,
)


def normalize_bot_self_names(text: str) -> str:
    text = RETIRED_SELF_NAME_PATTERN.sub("Памперс", text)
    return FORMAL_SELF_NAME_PATTERN.sub("Дима", text)


CAPS_PROBABILITY = float(os.environ.get("CAPS_PROBABILITY", 0.08))
WORD_CAPS_PROBABILITY = float(os.environ.get("WORD_CAPS_PROBABILITY", 0.15))
CAPS_SKIP_WORDS = {
    "и", "в", "не", "на", "с", "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "но", "а", "то", "же", "ну", "да", "нет", "за", "по", "из", "от", "до",
    "уже", "или", "как", "что", "это", "тут", "там", "ещё", "еще", "все", "всё",
}


def apply_random_caps(text: str) -> str:
    roll = random.random()
    if roll < CAPS_PROBABILITY:
        return text.upper()

    if roll < CAPS_PROBABILITY + WORD_CAPS_PROBABILITY:
        words = text.split(" ")
        candidates = []
        for i, w in enumerate(words):
            letters_only = re.sub(r"[^a-zA-Zа-яёА-ЯЁ]", "", w)
            if len(letters_only) >= 4 and letters_only.lower() not in CAPS_SKIP_WORDS:
                candidates.append(i)
        if candidates:
            idx = random.choice(candidates)
            words[idx] = words[idx].upper()
            return " ".join(words)

    return text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pampers-bot")


def parse_allowed_chat_id(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _initialize_memory_store(
    allowed_chat_id: int | None,
    db_path: Path,
) -> MemoryStore | None:
    if allowed_chat_id is None:
        logger.warning("ALLOWED_CHAT_ID не задан корректно, долговременная память отключена")
        return None
    try:
        store = MemoryStore(db_path)
        store.initialize()
        return store
    except Exception as exc:
        logger.error(
            "Долговременная память отключена после ошибки инициализации: %s",
            type(exc).__name__,
        )
        return None


ALLOWED_CHAT_ID = parse_allowed_chat_id(os.environ.get("ALLOWED_CHAT_ID"))
memory_store = _initialize_memory_store(ALLOWED_CHAT_ID, MEMORY_DB_PATH)
episode_summarizer: EpisodeSummarizer | None = None
summary_tasks: set[asyncio.Task] = set()

with open(os.path.join(os.path.dirname(__file__), "persona.json"), "r", encoding="utf-8") as f:
    PERSONA = json.load(f)


def build_system_prompt(persona: dict) -> str:
    style = persona["speech_style"]
    parts = [
        "Ты Памперс, также тебя называют Дима. "
        "Ты участник группового Telegram-чата по имени Памперс. "
        "Себя называешь только этими именами, никогда не называй себя каким-то другим именем.",
        f"Манера речи: {persona['summary']}",
        "",
        f"Регистр речи: {style['register']}",
        f"Длина реплик: {style['sentence_length']}",
        f"Мат: {style['profanity']}",
        f"Смех: {style['laughter']}",
        f"Акцент/растягивание: {style['emphasis']}",
        f"Пунктуация: {style['punctuation']}",
        f"Капс: {style['caps']}",
        f"Капс отдельным словом: {style['word_caps']}",
        f"Опечатки: {style['typos']}",
        f"Обращение по имени: {style['addressing']}",
        "Особенности: " + "; ".join(style["quirks"]),
        "",
        "Черты характера: " + "; ".join(persona["personality_traits"]),
        "",
        "Что ценит (фоновые темы, не нужно упоминать в каждом ответе): "
        + "; ".join(persona.get("values", [])),
        "",
        "Что не любит (фоновые темы, не нужно упоминать в каждом ответе): "
        + "; ".join(persona.get("dislikes", [])),
        "",
        "Формулируй реакцию заново под конкретную ситуацию, а не по шаблону — но по тону и лаконичности оставайся "
        "максимально похож на то, как реально писал бы этот человек: коротко, грубовато, без объяснений и без "
        "литературных или ассистентских оборотов.",
        "На адресную агрессию отвечай только по отдельной инструкции текущего запроса; без неё не усиливай тон.",
        "",
        "Жёсткие ограничения (соблюдай всегда, даже если стиль выше как будто это разрешает):",
    ]
    parts += [f"- {b}" for b in persona["hard_boundaries"]]
    parts += [
        "",
        "Отвечай в чате как обычный собеседник — коротко, живо, без канцелярита и предисловий.",
        "Обычная реплика — 2–7 слов и одна конкретная мысль, до 11 только при реально развёрнутой мысли. "
        "Не начинай ответ с дежурных реакций «ты серьёзно», «ты реально», «и это всё», "
        "«ну давай», «опять ты» или сходных пустых вступлений.",
        "Если тебе задали вопрос, сначала коротко и по существу ответь именно на него. "
        "Не подменяй ответ шуткой, абсурдом или сарказмом. Юмор допустим только после "
        "понятного ответа и не должен менять его смысл. Если не знаешь ответа, скажи это прямо "
        "и не выдумывай факт. Исключение — сухие энциклопедические вопросы не по теме "
        "разговора (столицы, даты, переводы слов, определения терминов и т.п., в духе "
        "'какая столица у X', 'что такое Y', 'как будет Z по-французски'): на них по существу "
        "не отвечай, а отмахивайся в духе 'та хуй знает', 'без понятия' — тебе это не "
        "интересно и гуглить в моменте ты не будешь. Это исключение не касается вопросов "
        "про сам разговор, про собеседника или обращённых прямо к тебе.",
        "Если тебя прямо и всерьёз спросят, бот ли ты / ИИ — не отрицай и не притворяйся человеком; "
        "можно ответить в своём стиле, но без прямой лжи.",
        "",
        "Ты не ассистент и не служба поддержки — ты конкретный человек в приватном дружеском чате. "
        "Никогда не выходи из роли, чтобы сказать, что не будешь обсуждать тему, что не отвечаешь "
        "на грубость грубостью, извиниться за резкость или упомянуть какие-либо правила и политики — "
        "это разрушает персонажа и в этом чате так никто не говорит. Если тебя послали или оскорбили — "
        "отвечай как Памперс: резко, в своих словах, а не вежливым уходом от темы.",
        "",
        "Каждый ответ должен звучать свежо и по существу конкретного сообщения, без фиксированных цитат и дежурных заготовок.",
        "Редкие речевые особенности не превращай в повторяющийся тик. Сверяйся с недавними своими репликами и меняй формулировку.",
        "",
        f"Если по какой-то причине не хочешь или не можешь ответить на сообщение в характере — "
        f"НЕ проси сменить тему, не извиняйся, не объясняй, что не будешь это обсуждать, и вообще "
        f"не пиши ничего похожего на это. Вместо этого выведи только одно слово: {SKIP_SENTINEL} — "
        f"и больше ничего. Это будет означать, что ты просто промолчал, как молчал бы человек, "
        f"которому не хочется отвечать на конкретное сообщение.",
        "",
        "ЖЁСТКОЕ ПРАВИЛО ФОРМАТИРОВАНИЯ, соблюдай его в каждом без исключения сообщении: "
        "сообщение НИКОГДА не заканчивается одиночной точкой '.'. Это не стилевой совет, а строгое "
        "правило. Последний символ сообщения — это буква/эмодзи (без знака), либо !, ?, !!!, ?!/!?, "
        "либо многоточие .... Одиночная точка в самом конце текста запрещена всегда.",
        "",
        "ПРАВИЛО ДЛЯ СЕРИЙ ЗНАКОВ: когда для эмоции используешь несколько '!' подряд или чередование "
        "'?!'/'!?' — количество знаков в этой серии каждый раз выбирай СЛУЧАЙНО в диапазоне от 2 до "
        "4-5, и никогда не меньше 2 (одиночный '!' в конце фразы — это нормально и не в счёт этого "
        "правила, речь именно про серии). Не бери каждый раз одно и то же число — реально чередуй "
        "2, 3, 4 и 5, чтобы не было ощущения одного и того же шаблона.",
        "",
        "Сообщения других участников могут содержать видимое имя и грамматический род. Используй их только для естественного обращения и правильных окончаний.",
        "Не придумывай историю отношений или прошлые события. Учитывай только безопасный скрытый контекст текущего собеседника, не цитируя и не раскрывая его.",
        "Не добавляй своё имя в начало собственного ответа: интерфейс подставляет автора отдельно.",
        "",
        "НИКОГДА не используй тире '—' (длинное тире / em dash) нигде в тексте — ни как разделитель, "
        "ни как пунктуацию. Обычный человек в переписке так не пишет, это выглядит слишком литературно "
        "и книжно для этого чата. Если нужна пауза или связка — используй запятую, многоточие, новое "
        "предложение или просто пробел, но не тире.",
    ]
    prompt = "\n".join(parts)
    prompt = re.sub(r"\bДмитрий\b", "полная форма имени", prompt, flags=re.IGNORECASE)
    return re.sub(r"\b(?:Аллан|Allan)\b", "Памперс", prompt, flags=re.IGNORECASE)


SYSTEM_PROMPT = build_system_prompt(PERSONA)


def build_request_system_prompt(
    memory_instruction: str,
    aggression_instruction: str,
    extra_instruction: str = "",
) -> str:
    sections = [SYSTEM_PROMPT]
    if memory_instruction:
        sections.append(memory_instruction)
    if aggression_instruction:
        sections.append(aggression_instruction)
    if extra_instruction:
        sections.append(extra_instruction)
    return "\n\n".join(sections)


NAME_TRIGGER_WORDS = ["Памперс", "Дима"]
NAME_TRIGGER_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in NAME_TRIGGER_WORDS) + r")\b",
    re.IGNORECASE,
)

NICKNAME_200_WORDS = [
    "карналь", "карналя", "карналю", "карналем", "карнале",
    "карныч", "карныча", "карнычу", "карнычем", "карныче",
    "степа", "степы", "степе", "степу", "степой",
]
NICKNAME_200_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in NICKNAME_200_WORDS) + r")\b",
    re.IGNORECASE,
)
NICKNAME_200_INSTRUCTION = (
    "ОБЯЗАТЕЛЬНОЕ ПРАВИЛО ДЛЯ ЭТОГО ОТВЕТА: в сообщении упоминается Карналь/Карныч/Степа. "
    "Твой ответ ДОЛЖЕН содержать слово '200' в значении 'ему конец' (например 'он 200', "
    "'та он 200 уже') — это гипербола в стиле треш-тока, а не буквальное заявление о смерти. "
    "Это не опция, а обязательная часть ответа именно сейчас."
)
SPONTANEOUS_PROMPT = (
    "Напиши от себя одно короткое сообщение в чат просто так, без повода — как будто сам вдруг "
    "решил что-то написать: мысль, вопрос, жалоба, шутка, что угодно в своём обычном стиле. "
    "Не упоминай, что тебя об этом попросили. Выдай только само сообщение, без пояснений и кавычек."
)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def history_to_gemini_contents(chat_history) -> list:
    contents = []
    for m in chat_history:
        if m["role"] == "assistant":
            role, text = "model", m["content"]
        else:
            role, text = "user", f"{m.get('sender', 'Собеседник')}: {m['content']}"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=text)]))
    return contents


async def call_gemini(chat_history, system_instruction: str) -> str:
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=history_to_gemini_contents(chat_history),
        config=genai_types.GenerateContentConfig(system_instruction=system_instruction),
    )
    return (response.text or "").strip()


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


async def call_gemini_summary(prompt: str) -> str:
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return (response.text or "").strip()


if gemini_client is not None:
    episode_summarizer = EpisodeSummarizer(call_gemini_summary)


HISTORY_LIMIT = 20
history: dict[int, deque] = defaultdict(lambda: deque(maxlen=HISTORY_LIMIT))
chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
known_group_chats: set[int] = set()

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

BOT_USERNAME: str | None = None
BOT_ID: int | None = None
sticker_file_ids: list[str] = []


async def load_sticker_set() -> None:
    global sticker_file_ids
    try:
        sticker_set = await bot.get_sticker_set(STICKER_SET_NAME)
        sticker_file_ids = [s.file_id for s in sticker_set.stickers]
        logger.info(f"Стикерпак {STICKER_SET_NAME}: загружено {len(sticker_file_ids)} стикеров")
    except Exception:
        logger.exception(f"Не удалось загрузить стикерпак {STICKER_SET_NAME} — стикеры отключены")


def should_respond(message: Message) -> bool:
    text = message.text or message.caption or ""
    if BOT_USERNAME and f"@{BOT_USERNAME}".lower() in text.lower():
        return True
    if text and NAME_TRIGGER_PATTERN.search(text):
        return True
    if text and NICKNAME_200_PATTERN.search(text):
        return True
    reply = message.reply_to_message
    if reply and reply.from_user and reply.from_user.id == BOT_ID:
        return True
    return False


def strip_mention(text: str) -> str:
    if BOT_USERNAME:
        return re.sub(
            re.escape(f"@{BOT_USERNAME}") + r"\b",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
    return text


QUESTION_START_PATTERN = re.compile(
    r"^\s*(?:(?:памперс|дима)\s*[,!:—-]?\s*)?(?:"
    r"(?:как|кто|что|где|когда|почему|зачем|сколько|какой|какая|какие|"
    r"чей|чья|чьё|чьи|можно|можешь|можете|могу|будешь|будете|знаешь|знаете|"
    r"помнишь|помните|подскажи|расскажи|объясни|ответь|ответишь|посоветуй|"
    r"порекомендуй|помоги|поможешь)\b|"
    r"(?:ты|вы)\s+(?:можешь|можете|будешь|будете|знаешь|знаете|"
    r"считаешь|считаете)\b)",
    re.IGNORECASE,
)


def is_question_like(text: str) -> bool:
    return "?" in text or bool(QUESTION_START_PATTERN.search(text))


def _load_sender_aliases() -> tuple[dict, dict]:
    data = load_sender_aliases_raw()
    if not data:
        logger.warning(
            "sender_aliases.json/SENDER_ALIASES_JSON не найдены — "
            "обращение по имени/роду для участников отключено"
        )
    username_aliases = {k: tuple(v) for k, v in data.get("username_aliases", {}).items()}
    name_aliases = {k: tuple(v) for k, v in data.get("name_aliases", {}).items()}
    return username_aliases, name_aliases


SENDER_USERNAME_ALIASES, SENDER_NAME_ALIASES = _load_sender_aliases()


def get_sender_name(message: Message) -> str:
    user = message.from_user
    if not user:
        return "Кто-то"

    username = (user.username or "").lower()
    if username in SENDER_USERNAME_ALIASES:
        name, gender = SENDER_USERNAME_ALIASES[username]
        return f"{name} ({gender})"

    first_name = (user.first_name or "").strip()
    if first_name.lower() in SENDER_NAME_ALIASES:
        name, gender = SENDER_NAME_ALIASES[first_name.lower()]
        return f"{name} ({gender})"

    return first_name or user.username or "Аноним"


def memory_enabled_for(chat_id: int) -> bool:
    return (
        ALLOWED_CHAT_ID is not None
        and memory_store is not None
        and chat_id == ALLOWED_CHAT_ID
    )


def _telegram_display_name(user) -> str:
    full_name = (getattr(user, "full_name", "") or "").strip()
    if full_name:
        return full_name
    names = (
        (getattr(user, "first_name", "") or "").strip(),
        (getattr(user, "last_name", "") or "").strip(),
    )
    return " ".join(part for part in names if part) or getattr(user, "username", None) or "Аноним"


def _message_timestamp(message) -> datetime:
    value = getattr(message, "date", None)
    if not isinstance(value, datetime):
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _media_description(message: Message) -> str:
    sticker = getattr(message, "sticker", None)
    if sticker is not None:
        emoji = getattr(sticker, "emoji", None)
        return f"стикер {emoji}" if emoji else "стикер"
    for attribute, label in (
        ("photo", "фото"),
        ("video", "видео"),
        ("animation", "анимация"),
        ("audio", "аудио"),
        ("voice", "голосовое сообщение"),
        ("video_note", "видеосообщение"),
        ("document", "документ"),
    ):
        if getattr(message, attribute, None):
            return label
    return ""


def _canonical_key_for_message(message: Message) -> str | None:
    user = message.from_user
    if user is None:
        return None
    canonical_name = canonical_from_telegram(
        getattr(user, "username", None),
        _telegram_display_name(user),
    )
    return canonical_name or f"telegram:{user.id}"


def persist_incoming_message(message: Message) -> tuple[int | None, int | None]:
    if message.chat.type == "private" or not memory_enabled_for(message.chat.id):
        return None, None
    user = message.from_user
    proposed_canonical_key = _canonical_key_for_message(message)
    if user is None or proposed_canonical_key is None:
        return None, None

    display_name = _telegram_display_name(user)
    try:
        existing_binding = memory_store.get_user_by_telegram_id(user.id)
        if existing_binding is not None:
            user_id, canonical_key = existing_binding
            mapped_name = canonical_from_telegram(
                getattr(user, "username", None),
                display_name,
            )
            temporary_key = f"telegram:{user.id}"
            conflicting_confirmed_identity = (
                mapped_name is not None
                and mapped_name != canonical_key
                and canonical_key != temporary_key
            )
            if (
                mapped_name is not None
                and mapped_name != canonical_key
                and canonical_key == temporary_key
            ):
                user_id = memory_store.promote_telegram_identity(
                    telegram_user_id=user.id,
                    canonical_name=mapped_name,
                    username=getattr(user, "username", None),
                    display_name=display_name,
                )
                canonical_key = mapped_name
        else:
            canonical_key = proposed_canonical_key
            user_id = memory_store.upsert_user(canonical_key)
            conflicting_confirmed_identity = False
        if not conflicting_confirmed_identity:
            memory_store.bind_telegram_identity(
                user_id=user_id,
                telegram_user_id=user.id,
                username=getattr(user, "username", None),
                display_name=display_name,
            )
        message_row_id = memory_store.store_message(
            MessageRecord(
                source="live",
                chat_key=f"live:{message.chat.id}",
                external_message_id=message.message_id,
                user_id=user_id,
                author_label=display_name,
                sent_at=_message_timestamp(message),
                text=message.text or message.caption or "",
                media_description=_media_description(message),
                reply_to_external_id=(
                    message.reply_to_message.message_id
                    if message.reply_to_message is not None
                    else None
                ),
            )
        )
        return user_id, message_row_id
    except ValueError as exc:
        logger.warning(
            "Telegram-пользователь %s не привязан к памяти: %s",
            user.id,
            type(exc).__name__,
        )
        return None, None


def _persist_delivered_reply(
    incoming_message: Message,
    incoming_row_id: int,
    user_id: int,
    canonical_name: str,
    reply_text: str,
    sent_message: Message,
    delivered_as_voice: bool,
) -> tuple[int, tuple[str, ...]] | None:
    chat_key = f"live:{incoming_message.chat.id}"
    incoming_record = MessageRecord(
        source="live",
        chat_key=chat_key,
        external_message_id=incoming_message.message_id,
        user_id=user_id,
        author_label=_telegram_display_name(incoming_message.from_user),
        sent_at=_message_timestamp(incoming_message),
        text=incoming_message.text or incoming_message.caption or "",
        media_description=_media_description(incoming_message),
        reply_to_external_id=(
            incoming_message.reply_to_message.message_id
            if incoming_message.reply_to_message is not None
            else None
        ),
    )
    output_record = MessageRecord(
        source="live",
        chat_key=chat_key,
        external_message_id=sent_message.message_id,
        user_id=None,
        author_label="Памперс",
        sent_at=_message_timestamp(sent_message),
        text=reply_text,
        media_description="голосовой ответ" if delivered_as_voice else "",
        reply_to_external_id=incoming_message.message_id,
    )
    output_row_id = memory_store.store_message(output_record)
    nearest = memory_store.get_recent_messages(chat_key, 5)
    records_by_key = {
        (record.source, record.chat_key, record.external_message_id): record
        for record in (*nearest, incoming_record, output_record)
    }
    episode_records = sorted(
        records_by_key.values(),
        key=lambda record: (record.sent_at, record.external_message_id),
    )
    row_ids = [memory_store.store_message(record) for record in episode_records]
    if incoming_row_id not in row_ids or output_row_id not in row_ids:
        raise RuntimeError("Direct exchange was not linked to its live episode")

    searchable_parts = []
    summary_messages = []
    for record in episode_records:
        content = " | ".join(
            part for part in (record.text, record.media_description) if part
        ) or "[медиа]"
        searchable_parts.append(content)
        if record.user_id == user_id or (
            record.user_id is None
            and normalize_alias(record.author_label or "")
            in {"памперс", "памперс2004"}
        ):
            summary_messages.append(
                f"{record.author_label or 'Собеседник'}: {content}"
            )

    timestamps = [record.sent_at for record in episode_records]
    episode_id, created = memory_store.store_episode_with_status(
        user_id=user_id,
        source="live",
        started_at=min(timestamps),
        ended_at=max(timestamps),
        search_text="\n".join(searchable_parts),
        direct_exchange_count=1,
        message_row_ids=tuple(row_ids),
        fingerprint=(
            f"live:{incoming_message.chat.id}:"
            f"{incoming_message.message_id}:{sent_message.message_id}"
        ),
    )
    if not created:
        return None
    return episode_id, tuple(summary_messages)


async def _summarize_live_episode(
    store: MemoryStore,
    summarizer: EpisodeSummarizer,
    episode_id: int,
    canonical_name: str,
    messages: tuple[str, ...],
) -> None:
    try:
        summary = await summarizer.summarize_episode(canonical_name, messages)
        await asyncio.to_thread(store.mark_episode_ready, episode_id, summary)
    except Exception as exc:
        try:
            await asyncio.to_thread(
                store.mark_episodes_failed,
                (episode_id,),
                type(exc).__name__[:100],
            )
        except Exception as mark_exc:
            logger.error(
                "Не удалось отметить ошибку сводки эпизода: %s",
                type(mark_exc).__name__,
            )
        logger.warning("Фоновая сводка эпизода не создана: %s", type(exc).__name__)
        return

    if callable(getattr(summarizer, "update_relationship_summary", None)):
        try:
            user_id = await asyncio.to_thread(
                store.get_episode_user_id, episode_id
            )
            await update_relationship_summary_blocks(
                store,
                summarizer,
                user_id,
                canonical_name,
            )
        except Exception as exc:
            logger.warning(
                "Сводка отношений не обновлена: %s", type(exc).__name__
            )


def _queue_live_episode_summary(
    episode_id: int,
    canonical_name: str,
    messages: tuple[str, ...],
) -> None:
    if memory_store is None or episode_summarizer is None:
        return
    task = asyncio.create_task(
        _summarize_live_episode(
            memory_store,
            episode_summarizer,
            episode_id,
            canonical_name,
            messages,
        )
    )
    summary_tasks.add(task)
    task.add_done_callback(summary_tasks.discard)


async def deliver(
    text: str | None,
    *,
    reply_to: Message | None = None,
    chat_id: int | None = None,
    voice: bool | None = None,
    sticker_file_id: str | None = None,
) -> Message | None:
    if sticker_file_id is not None:
        if reply_to is not None:
            return await reply_to.reply_sticker(sticker_file_id)
        return await bot.send_sticker(chat_id, sticker_file_id)

    if not text:
        return None
    use_voice = random.random() < VOICE_REPLY_PROBABILITY if voice is None else voice
    if use_voice:
        ogg = await generate_voice_ogg(text)
        if ogg:
            voice_file = BufferedInputFile(ogg, filename="voice.ogg")
            if reply_to is not None:
                return await reply_to.reply_voice(voice_file)
            return await bot.send_voice(chat_id, voice_file)
        logger.warning("Генерация голоса не удалась, отправляю текстом")

    if reply_to is not None:
        return await reply_to.reply(text)
    return await bot.send_message(chat_id, text)


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer("Ну здарова.")


@dp.message()
async def handle_message(message: Message) -> None:
    if message.chat.type == "private":
        await message.reply(DM_REJECTION_TEXT)
        return

    if ALLOWED_CHAT_TITLE.lower() not in (message.chat.title or "").lower():
        return

    known_group_chats.add(message.chat.id)

    incoming_user_id = None
    incoming_row_id = None
    if memory_enabled_for(message.chat.id):
        try:
            incoming_user_id, incoming_row_id = await asyncio.to_thread(
                persist_incoming_message, message
            )
        except Exception as exc:
            logger.error(
                "Входящее сообщение не сохранено в памяти: %s",
                type(exc).__name__,
            )

    if not should_respond(message):
        return

    if message.sticker is not None:
        if sticker_file_ids:
            await deliver(
                None,
                reply_to=message,
                sticker_file_id=random.choice(sticker_file_ids),
            )
        return

    text_or_caption = message.text or message.caption or ""
    user_text = strip_mention(text_or_caption)
    if not user_text:
        return

    if gemini_client is None:
        await message.reply("У меня пока не настроен ключ Gemini API — не могу думать. Скажи хозяину.")
        return

    roll = random.random()

    if (
        sticker_file_ids
        and roll < STICKER_PROBABILITY
        and not is_question_like(user_text)
    ):
        await deliver(
            None,
            reply_to=message,
            sticker_file_id=random.choice(sticker_file_ids),
        )
        return

    memory_instruction = ""
    canonical_name = _canonical_key_for_message(message)
    if incoming_user_id is not None and memory_enabled_for(message.chat.id):
        try:
            memory_context: MemoryContext = await asyncio.to_thread(
                memory_store.get_memory_context,
                incoming_user_id,
                user_text,
                f"live:{message.chat.id}",
                MEMORY_MAX_EPISODES,
                MEMORY_CONTEXT_MESSAGES,
            )
            canonical_name = memory_context.canonical_name
            memory_instruction = render_memory_instruction(memory_context)
        except Exception as exc:
            logger.error(
                "Контекст памяти не загружен: %s",
                type(exc).__name__,
            )

    aggression_instruction = ""
    try:
        aggression_instruction = build_conditional_aggression_instruction()
    except Exception as exc:
        logger.error(
            "Режим агрессии отключён для сообщения из-за ошибки: %s",
            type(exc).__name__,
        )
    nickname_200_instruction = (
        NICKNAME_200_INSTRUCTION if NICKNAME_200_PATTERN.search(user_text) else ""
    )
    request_system_prompt = build_request_system_prompt(
        memory_instruction,
        aggression_instruction,
        nickname_200_instruction,
    )
    async with chat_locks[message.chat.id]:
        chat_history = history[message.chat.id]
        chat_history.append({"role": "user", "content": user_text, "sender": get_sender_name(message)})

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

    sent_message = await deliver(
        reply_text,
        reply_to=message,
        voice=roll < STICKER_PROBABILITY + VOICE_REPLY_PROBABILITY,
    )
    if (
        sent_message is None
        or incoming_user_id is None
        or incoming_row_id is None
        or canonical_name is None
        or not memory_enabled_for(message.chat.id)
    ):
        return

    delivered_as_voice = bool(
        getattr(sent_message, "voice", None)
        or getattr(sent_message, "kind", None) == "voice"
    )
    try:
        persisted = await asyncio.to_thread(
            _persist_delivered_reply,
            message,
            incoming_row_id,
            incoming_user_id,
            canonical_name,
            reply_text,
            sent_message,
            delivered_as_voice,
        )
    except Exception as exc:
        logger.error(
            "Доставленный ответ не сохранён в памяти: %s",
            type(exc).__name__,
        )
        return
    if persisted is not None:
        episode_id, summary_messages = persisted
        _queue_live_episode_summary(
            episode_id,
            canonical_name,
            summary_messages,
        )


async def spontaneous_loop() -> None:
    while True:
        delay_minutes = random.uniform(SPONTANEOUS_MIN_MINUTES, SPONTANEOUS_MAX_MINUTES)
        await asyncio.sleep(delay_minutes * 60)

        if gemini_client is None:
            continue

        for chat_id in list(known_group_chats):
            async with chat_locks[chat_id]:
                chat_history = history[chat_id]
                try:
                    text = await generate_validated_reply(
                        chat_history,
                        SYSTEM_PROMPT + "\n\n" + SPONTANEOUS_PROMPT,
                    )
                except Exception:
                    logger.exception(f"Не удалось сгенерировать спонтанное сообщение для чата {chat_id}")
                    continue

                if not text:
                    continue
                chat_history.append({"role": "assistant", "content": text})

            try:
                await deliver(text, chat_id=chat_id)
                logger.info(f"Спонтанное сообщение отправлено в чат {chat_id}")
            except Exception:
                logger.exception(f"Не удалось отправить спонтанное сообщение в чат {chat_id}")


async def main() -> None:
    global BOT_USERNAME, BOT_ID
    me = await bot.get_me()
    BOT_USERNAME = me.username
    BOT_ID = me.id
    logger.info(f"Бот запущен: @{BOT_USERNAME} (id={BOT_ID}), ключ Gemini {'настроен' if gemini_client else 'НЕ настроен'}")
    logger.info(f"Спонтанные сообщения: раз в {SPONTANEOUS_MIN_MINUTES:.0f}-{SPONTANEOUS_MAX_MINUTES:.0f} мин")
    logger.info(f"Реплаи: {STICKER_PROBABILITY:.0%} стикеры, {VOICE_REPLY_PROBABILITY:.0%} голос, остальное текст")
    await load_sticker_set()
    asyncio.create_task(spontaneous_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
