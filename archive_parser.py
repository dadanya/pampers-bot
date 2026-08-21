from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path


_SPACE_RE = re.compile(r"\s+")
_MESSAGE_ID_RE = re.compile(r"^message(-?\d+)$")
_REPLY_ID_RE = re.compile(r"GoToMessage\((-?\d+)\)")
_REPLY_HREF_RE = re.compile(r"#go_to_message(-?\d+)$")
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


@dataclass(frozen=True)
class ArchiveMessage:
    id: int
    author: str | None
    sent_at: datetime
    text: str
    media_description: str
    reply_to: int | None
    page: str


def _clean(value: str) -> str:
    return _SPACE_RE.sub(" ", html.unescape(value)).strip()


def _parse_sent_at(value: str) -> datetime:
    return datetime.strptime(value, "%d.%m.%Y %H:%M:%S UTC%z")


class _TelegramExportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str]]] = []
        self.current: dict[str, object] | None = None
        self.current_depth: int | None = None
        self.messages: list[ArchiveMessage] = []
        self.last_author: str | None = None
        self.page = ""

    def handle_starttag(
        self,
        tag: str,
        attrs_list: list[tuple[str, str | None]],
    ) -> None:
        attrs = dict(attrs_list)
        classes = set((attrs.get("class") or "").split())
        if tag not in _VOID_TAGS:
            self.stack.append((tag, classes))

        if tag == "div" and {"message", "default"}.issubset(classes):
            match = _MESSAGE_ID_RE.match(attrs.get("id") or "")
            if match is None:
                return
            self.current = {
                "id": int(match.group(1)),
                "author": None,
                "date": None,
                "text_parts": [],
                "media_parts": [],
                "reply_to": None,
                "joined": "joined" in classes,
                "page": self.page,
            }
            self.current_depth = len(self.stack)
            return

        if self.current is None:
            return

        if tag == "br" and self._inside("text") and not self._inside("reply_to"):
            self._parts("text_parts").append("\n")

        if (
            tag == "div"
            and "from_name" in classes
            and self.current_depth is not None
            and len(self.stack) == self.current_depth + 2
            and self.current.get("author_capture_depth") is None
            and not self.current.get("author_parts")
        ):
            self.current["author_capture_depth"] = len(self.stack)

        if tag == "div" and "date" in classes and attrs.get("title"):
            if self.current["date"] is None:
                self.current["date"] = attrs["title"]

        if tag == "a":
            href = attrs.get("href")
            if self._inside("reply_to"):
                onclick_match = _REPLY_ID_RE.search(attrs.get("onclick") or "")
                href_match = _REPLY_HREF_RE.search(href or "")
                if onclick_match:
                    self.current["reply_to"] = int(onclick_match.group(1))
                elif href_match:
                    self.current["reply_to"] = int(href_match.group(1))

            if href and self._inside("media_wrap"):
                self._parts("media_parts").append(href)

    def handle_startendtag(
        self,
        tag: str,
        attrs_list: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs_list)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if (
            tag == "div"
            and self.current is not None
            and self.current.get("author_capture_depth") == len(self.stack)
        ):
            self.current["author_capture_depth"] = None

        if (
            tag == "div"
            and self.current is not None
            and self.current_depth == len(self.stack)
        ):
            self._finish_message()

        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.current is None:
            return

        if self._inside("text") and not self._inside("reply_to"):
            self._parts("text_parts").append(data)
            return
        if not data.strip():
            return
        if self.current.get("author_capture_depth") is not None:
            self._parts("author_parts").append(data)
        elif self._inside("media_wrap"):
            self._parts("media_parts").append(data)

    def _inside(self, class_name: str) -> bool:
        return any(class_name in classes for _, classes in self.stack)

    def _parts(self, key: str) -> list[str]:
        if self.current is None:
            raise RuntimeError("message capture is not active")
        parts = self.current.setdefault(key, [])
        if not isinstance(parts, list):
            raise TypeError(f"{key} must be a list")
        return parts

    def _finish_message(self) -> None:
        if self.current is None:
            return

        author = _clean("".join(self._parts("author_parts")))
        if author:
            self.last_author = author
        elif bool(self.current["joined"]):
            author = self.last_author or ""

        date = self.current["date"]
        if not isinstance(date, str):
            raise ValueError(f"message {self.current['id']} has no timestamp")

        self.messages.append(
            ArchiveMessage(
                id=int(self.current["id"]),
                author=author or None,
                sent_at=_parse_sent_at(date),
                text=_clean("".join(self._parts("text_parts"))),
                media_description=_clean(" ".join(self._parts("media_parts"))),
                reply_to=(
                    int(self.current["reply_to"])
                    if self.current["reply_to"] is not None
                    else None
                ),
                page=str(self.current["page"]),
            )
        )
        self.current = None
        self.current_depth = None


def _page_number(path: Path) -> int:
    match = re.fullmatch(r"messages(\d*)\.html", path.name)
    if match is None:
        return 10**9
    return int(match.group(1) or "1")


def parse_export(export_dir: Path) -> list[ArchiveMessage]:
    parser = _TelegramExportParser()
    for path in sorted(export_dir.glob("messages*.html"), key=_page_number):
        parser.page = path.name
        parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser.messages
