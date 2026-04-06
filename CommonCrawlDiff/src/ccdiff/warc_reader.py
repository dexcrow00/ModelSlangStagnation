import html
import re

from warcio.archiveiterator import ArchiveIterator


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(raw_html: str) -> str:
    """Crude HTML→text: drop tags, decode entities, normalise whitespace."""
    text = _TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def read_warc(path: str) -> dict[str, str]:
    """Read a WARC file and return {url: extracted_text} for HTTP responses."""
    pages: dict[str, str] = {}
    with open(path, "rb") as fh:
        for record in ArchiveIterator(fh):
            if record.rec_type != "response":
                continue
            url = record.rec_headers.get_header("WARC-Target-URI")
            content_type = record.http_headers.get_header("Content-Type") or ""
            if "text/html" not in content_type:
                continue
            raw = record.content_stream().read().decode("utf-8", errors="replace")
            pages[url] = _strip_html(raw)
    return pages
