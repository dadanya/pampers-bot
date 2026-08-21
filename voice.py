import asyncio
import json
import logging
import os
import shutil
import tempfile

import aiohttp

logger = logging.getLogger("pampers-bot.voice")

VOICE_ID = "f40ad9f4-947e-42a4-869d-96e2ef7cb23d"  # "ДимаПамперс" voice clone in Higgsfield
VOICE_MODEL = "minimax"
HIGGSFIELD_CMD = shutil.which("higgsfield") or "higgsfield"
FFMPEG_CMD = shutil.which("ffmpeg") or "ffmpeg"


async def generate_voice_ogg(text: str) -> bytes | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            HIGGSFIELD_CMD, "generate", "create", "text2speech_v2",
            "--prompt", text,
            "--voice_id", VOICE_ID,
            "--voice_type", "element",
            "--model", VOICE_MODEL,
            "--language_boost", "ru",
            "--wait", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except Exception:
        logger.exception("Не удалось запустить higgsfield CLI")
        return None

    if proc.returncode != 0:
        logger.error(f"higgsfield CLI завершился с кодом {proc.returncode}: {stderr.decode(errors='replace')}")
        return None

    try:
        jobs = json.loads(stdout.decode("utf-8"))
        result_url = jobs[0]["result_url"]
    except Exception:
        logger.exception(f"Не удалось разобрать вывод higgsfield: {stdout[:500]!r}")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(result_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                mp3_bytes = await resp.read()
    except Exception:
        logger.exception("Не удалось скачать сгенерированное аудио")
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        mp3_path = os.path.join(tmp_dir, "in.mp3")
        ogg_path = os.path.join(tmp_dir, "out.ogg")
        with open(mp3_path, "wb") as f:
            f.write(mp3_bytes)

        try:
            ffmpeg_proc = await asyncio.create_subprocess_exec(
                FFMPEG_CMD, "-y", "-i", mp3_path,
                "-c:a", "libopus", "-b:a", "32k", "-ar", "48000", "-ac", "1",
                ogg_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(ffmpeg_proc.wait(), timeout=30)
        except Exception:
            logger.exception("Не удалось запустить ffmpeg")
            return None

        if ffmpeg_proc.returncode != 0 or not os.path.exists(ogg_path):
            logger.error("Конвертация ffmpeg в ogg/opus не удалась")
            return None

        with open(ogg_path, "rb") as f:
            return f.read()
