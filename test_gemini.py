import asyncio
import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

from bot import call_gemini, SYSTEM_PROMPT, is_skip_response  # noqa: E402


async def main():
    messages = [
        "го в футбол вечером?",
        "как дела?",
        "иди нахуй отсюда",
        "ты бот?",
    ]
    history = []
    for msg in messages:
        history.append({"role": "user", "content": msg})
        reply = await call_gemini(history, SYSTEM_PROMPT)
        history.append({"role": "assistant", "content": reply})
        print(f"> {msg}\n< {reply}  [skip={is_skip_response(reply)}]\n")


asyncio.run(main())
