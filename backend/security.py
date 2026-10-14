import html
import ipaddress
from html.parser import HTMLParser
from typing import Dict, FrozenSet, List, Optional, Tuple
from urllib.parse import urlsplit

DOWNLOAD_HOSTS: Dict[str, FrozenSet[str]] = {
    "civitai": frozenset({"civitai.com"}),
    "huggingface": frozenset({"huggingface.co", "*.huggingface.co"}),
    "openmodeldb": frozenset({
        "huggingface.co",
        "github.com",
        "objects.githubusercontent.com",
        "raw.githubusercontent.com",
        "*.oraclecloud.com",
        "*.backblazeb2.com",
    }),
    "civarchive": frozenset({"civitai.com", "huggingface.co"}),
}


def _host_allowed(host: str, patterns: FrozenSet[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]) and len(host) > len(pattern) - 1:
                return True
        elif host == pattern:
            return True
    return False


def _is_local_or_numeric(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if host == "localhost" or host.endswith(".localhost"):
        return True
    return host.rsplit(".", 1)[-1].isdigit()


def is_allowed_download_url(source_id: str, url: str) -> bool:
    patterns = DOWNLOAD_HOSTS.get(source_id)
    if not patterns or not isinstance(url, str) or not url.isascii():
        return False
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F or char == "\\" for char in url):
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    if parts.scheme != "https" or port not in (None, 443):
        return False
    if "@" in parts.netloc:
        return False
    host = parts.hostname or ""
    if not host or host.endswith(".") or _is_local_or_numeric(host):
        return False
    return _host_allowed(host, patterns)


_KEPT = frozenset({
    "p", "br", "b", "strong", "i", "em", "u", "s", "ul", "ol", "li",
    "blockquote", "code", "pre", "hr", "a", "h3", "h4",
})
_RENAMED = {"h1": "h3", "h2": "h3", "h3": "h4", "h4": "h4", "h5": "h4", "h6": "h4", "strike": "s", "del": "s"}
_VOID_KEPT = frozenset({"br", "hr"})
_DROPPED_WITH_CONTENT = frozenset({
    "script", "style", "iframe", "object", "video", "audio", "svg", "math", "template",
    "noscript", "noembed", "noframes", "textarea", "select", "title", "head", "canvas",
    "frameset", "applet", "picture", "xmp", "plaintext", "button", "map",
})
_BLOCKS = frozenset({
    "div", "section", "article", "header", "footer", "main", "aside", "nav", "table",
    "thead", "tbody", "tfoot", "tr", "figure", "figcaption", "dl", "dt", "dd", "details",
    "summary", "center", "form", "fieldset", "address", "caption", "legend",
})
_CELLS = frozenset({"td", "th"})


def _safe_href(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if not value or any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return value


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: List[str] = []
        self._open: List[Tuple[str, str]] = []
        self._skipping: List[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _DROPPED_WITH_CONTENT:
            self._skipping.append(tag)
            return
        if self._skipping:
            return
        name = _RENAMED.get(tag, tag)
        if name in _VOID_KEPT:
            self._out.append(f"<{name}>")
            return
        if name not in _KEPT:
            return
        if name == "a":
            href = _safe_href(dict(attrs).get("href"))
            if href is None:
                self._open.append((tag, ""))
                return
            self._out.append(f'<a href="{html.escape(href, quote=True)}" rel="noopener noreferrer" target="_blank">')
        else:
            self._out.append(f"<{name}>")
        self._open.append((tag, name))

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._skipping:
            if tag in self._skipping:
                while self._skipping and self._skipping.pop() != tag:
                    pass
            return
        if tag in _BLOCKS:
            self._line_break()
            return
        if tag in _CELLS:
            self._out.append(" ")
            return
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index][0] == tag:
                while len(self._open) > index:
                    _, name = self._open.pop()
                    if name:
                        self._out.append(f"</{name}>")
                return

    def handle_data(self, data):
        if not self._skipping:
            self._out.append(html.escape(data, quote=False))

    def _line_break(self) -> None:
        if self._out and self._out[-1] != "<br>":
            self._out.append("<br>")

    def result(self) -> str:
        self.close()
        while self._open:
            _, name = self._open.pop()
            if name:
                self._out.append(f"</{name}>")
        return "".join(self._out).strip()


def sanitize_html(value: Optional[str]) -> str:
    if not value:
        return ""
    sanitizer = _Sanitizer()
    sanitizer.feed(str(value))
    return sanitizer.result()


class _TextExtractor(HTMLParser):
    _BREAKS = frozenset({"p", "br", "li", "tr", "td", "th", "div", "blockquote", "pre", "hr", "h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skipping: List[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _DROPPED_WITH_CONTENT:
            self._skipping.append(tag)
        elif tag in self._BREAKS:
            self._parts.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._skipping:
            while self._skipping and self._skipping.pop() != tag:
                pass
        elif tag in self._BREAKS:
            self._parts.append(" ")

    def handle_data(self, data):
        if not self._skipping:
            self._parts.append(data)

    def text(self) -> str:
        self.close()
        return " ".join("".join(self._parts).split())


def html_to_text(value: Optional[str], limit: int = 300) -> str:
    if not value or limit <= 0:
        return ""
    extractor = _TextExtractor()
    extractor.feed(str(value))
    text = extractor.text()
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space >= (limit - 1) // 2:
        cut = cut[:space]
    return cut.rstrip() + "…"
