from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import dotenv_values

from import_history import read_import_status


TelegramProbe = Callable[[str], Awaitable[bool]]
GeminiProbe = Callable[[str, str], Awaitable[bool]]
_ONLINE_PROBE_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class StartupStatus:
    allowed_chat_id_valid: bool
    memory_db_ready: bool
    pending_episodes: int
    ready_episodes: int
    failed_episodes: int
    estimated_import_calls: int
    online_checked: bool = False
    telegram_ok: bool = False
    gemini_ok: bool = False
    errors: tuple[str, ...] = ()

    @property
    def local_ready(self) -> bool:
        return self.allowed_chat_id_valid and self.memory_db_ready

    @property
    def fully_ready(self) -> bool:
        return self.local_ready and not self.errors and (
            not self.online_checked or (self.telegram_ok and self.gemini_ok)
        )

    def public_report(self) -> str:
        error_names = ",".join(self.errors) if self.errors else "none"
        return "\n".join(
            (
                f"allowed_chat_id_valid={int(self.allowed_chat_id_valid)}",
                f"memory_db_ready={int(self.memory_db_ready)}",
                f"pending_episodes={self.pending_episodes}",
                f"ready_episodes={self.ready_episodes}",
                f"failed_episodes={self.failed_episodes}",
                f"estimated_import_calls={self.estimated_import_calls}",
                f"online_checked={int(self.online_checked)}",
                f"telegram_ok={int(self.telegram_ok)}",
                f"gemini_ok={int(self.gemini_ok)}",
                f"errors={error_names}",
            )
        )


def _valid_chat_id(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        int(value.strip())
    except ValueError:
        return False
    return True


def check_local_configuration(
    environment: Mapping[str, object],
    db_path: Path,
) -> StartupStatus:
    allowed_chat_id_valid = _valid_chat_id(environment.get("ALLOWED_CHAT_ID"))
    errors: list[str] = []
    counts = {
        "pending": 0,
        "ready": 0,
        "failed": 0,
        "estimated_requests": 0,
    }

    try:
        if not Path(db_path).is_file():
            raise FileNotFoundError
        counts.update(read_import_status(Path(db_path)))
    except Exception as exc:
        errors.append(type(exc).__name__)

    pending = int(counts["pending"])
    failed = int(counts["failed"])
    memory_db_ready = (
        allowed_chat_id_valid
        and not errors
        and pending == 0
        and failed == 0
    )

    return StartupStatus(
        allowed_chat_id_valid=allowed_chat_id_valid,
        memory_db_ready=memory_db_ready,
        pending_episodes=pending,
        ready_episodes=int(counts["ready"]),
        failed_episodes=failed,
        estimated_import_calls=int(counts["estimated_requests"]),
        errors=tuple(errors),
    )


async def _telegram_health_probe(token: str) -> bool:
    from aiogram import Bot

    bot = Bot(token=token)
    try:
        account = await bot.get_me()
        return bool(account.id)
    finally:
        await bot.session.close()


async def _gemini_health_probe(api_key: str, model: str) -> bool:
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=model,
        contents="Reply with OK only.",
        config=genai_types.GenerateContentConfig(max_output_tokens=8),
    )
    return bool((response.text or "").strip())


async def run_online_checks(
    status: StartupStatus,
    environment: Mapping[str, object],
    *,
    telegram_probe: TelegramProbe = _telegram_health_probe,
    gemini_probe: GeminiProbe = _gemini_health_probe,
) -> StartupStatus:
    if not status.local_ready:
        return replace(
            status,
            errors=status.errors + ("LocalPrerequisitesError",),
        )

    token = environment.get("TELEGRAM_BOT_TOKEN")
    api_key = environment.get("GEMINI_API_KEY")
    model = environment.get("GEMINI_MODEL") or "gemini-3.6-flash"
    missing: list[str] = []
    if not isinstance(token, str) or not token.strip():
        missing.append("MissingTelegramToken")
    if not isinstance(api_key, str) or not api_key.strip():
        missing.append("MissingGeminiKey")
    if missing:
        return replace(status, errors=status.errors + tuple(missing))

    errors = list(status.errors)
    telegram_ok = False
    gemini_ok = False
    try:
        telegram_ok = bool(
            await asyncio.wait_for(
                telegram_probe(token.strip()),
                timeout=_ONLINE_PROBE_TIMEOUT_SECONDS,
            )
        )
        if not telegram_ok:
            errors.append("TelegramHealthCheckError")
    except Exception as exc:
        errors.append(type(exc).__name__)

    try:
        gemini_ok = bool(
            await asyncio.wait_for(
                gemini_probe(api_key.strip(), str(model)),
                timeout=_ONLINE_PROBE_TIMEOUT_SECONDS,
            )
        )
        if not gemini_ok:
            errors.append("GeminiHealthCheckError")
    except Exception as exc:
        errors.append(type(exc).__name__)

    return replace(
        status,
        online_checked=True,
        telegram_ok=telegram_ok,
        gemini_ok=gemini_ok,
        errors=tuple(errors),
    )


def _load_environment(env_path: Path) -> dict[str, str]:
    loaded = {
        key: value
        for key, value in dotenv_values(env_path).items()
        if isinstance(value, str)
    }
    loaded.update(os.environ)
    return loaded


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate bot startup without starting Telegram polling."
    )
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--db", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    environment = _load_environment(args.env)
    configured_db = environment.get("MEMORY_DB_PATH")
    db_path = args.db or (
        Path(configured_db) if configured_db else Path(__file__).with_name("memory.db")
    )
    status = check_local_configuration(environment, db_path)
    if args.online:
        status = asyncio.run(run_online_checks(status, environment))
    print(status.public_report())
    return 0 if status.fully_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
