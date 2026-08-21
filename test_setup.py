import asyncio
import os
import sys

from aiogram import Bot
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from bot import build_system_prompt, PERSONA, gemini_client

load_dotenv()


async def main():
    prompt = build_system_prompt(PERSONA)
    print("=== System prompt OK, length:", len(prompt), "chars ===")
    print(prompt[:200], "...\n")

    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    me = await bot.get_me()
    print(f"=== Telegram OK: @{me.username} (id={me.id}, name={me.first_name}) ===")
    await bot.session.close()

    if gemini_client:
        print("GEMINI_API_KEY: настроен")
    else:
        print("GEMINI_API_KEY: НЕ настроен — бот подключится к Telegram, но не сможет отвечать")


asyncio.run(main())
