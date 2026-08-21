from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, Sequence


_RESPONSE_MARKER = "- Ответ Памперса2004: «"
_RESPONSE_END = "» ("
_PROFANITY_SIGNALS = (
    "бля",
    "нахуй",
    "хуй",
    "пизд",
    "еб",
    "сука",
    "додик",
    "уеб",
    "шлюх",
)
_JUSTIFICATION_SIGNALS = (
    "потому",
    "почему",
    "я же",
    "я просто",
    "на самом деле",
    "если я",
    "у меня",
    "я хочу",
    "я не",
)


def extract_pampers_responses(report_text: str) -> tuple[str, ...]:
    responses = []
    for line in report_text.splitlines():
        if _RESPONSE_MARKER not in line:
            continue
        remainder = line.split(_RESPONSE_MARKER, 1)[1]
        if _RESPONSE_END not in remainder:
            continue
        response = remainder.rsplit(_RESPONSE_END, 1)[0].strip()
        if response:
            responses.append(response)
    return tuple(responses)


def _rate(values: Iterable[bool]) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0


def _keyword_rate(responses: Sequence[str], signals: Sequence[str]) -> float:
    return _rate(
        any(signal in response.casefold() for signal in signals)
        for response in responses
    )


def build_aggression_profile(responses: Sequence[str]) -> dict[str, object]:
    if not responses:
        raise ValueError("at least one response is required")
    word_counts = [len(response.split()) for response in responses]
    return {
        "version": 1,
        "source_response_count": len(responses),
        "median_words": statistics.median(word_counts),
        "up_to_5_words_rate": _rate(count <= 5 for count in word_counts),
        "up_to_10_words_rate": _rate(count <= 10 for count in word_counts),
        "profanity_rate": _keyword_rate(responses, _PROFANITY_SIGNALS),
        "justification_rate": _keyword_rate(
            responses, _JUSTIFICATION_SIGNALS
        ),
        "strategy_priority": [
            "situational_mirror",
            "contradiction",
            "dismissal",
            "absurd_bravado",
        ],
        "target_word_range": [1, 10],
    }


def write_aggression_profile(
    source_path: Path,
    output_path: Path,
) -> dict[str, object]:
    responses = extract_pampers_responses(
        Path(source_path).read_text(encoding="utf-8")
    )
    profile = build_aggression_profile(responses)
    Path(output_path).write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a raw-text-free aggression style profile."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error("source report does not exist")
    profile = write_aggression_profile(args.source, args.output)
    for key in (
        "source_response_count",
        "median_words",
        "up_to_5_words_rate",
        "up_to_10_words_rate",
        "profanity_rate",
        "justification_rate",
    ):
        print(f"{key}={profile[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
