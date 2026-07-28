"""Making untrusted corpus text safe to put in a report.

Every string in a dropoutt report came from data we did not write. Three
specific hazards, all handled here.

Control characters are made **visible** rather than stripped. A tool that
reports "your data contains control characters" and then renders them
invisibly has told the user nothing. They are mapped into the Unicode Control
Pictures block, so U+0000 shows as ␀.

Bidirectional overrides are neutralised. A single unbalanced RTL override in one
record would otherwise reverse the layout of every finding after it, which is
both a rendering bug and a spoofing vector.

JSON destined for a ``<script>`` element has its angle brackets escaped as
unicode sequences. A record containing the literal text ``</script>`` would
otherwise close the element and inject whatever follows.
"""

from __future__ import annotations

import html as _html
import json as _json
import re

# U+202A..U+202E and U+2066..U+2069 change text direction for everything after
# them until popped. In a report full of concatenated snippets, one stray
# override corrupts the rest of the page.
_BIDI = re.compile(r"[‪-‮⁦-⁩‎‏]")

_CONTROL_PICTURE_BASE = 0x2400


def visible_controls(text: str) -> str:
    """Map C0 controls into the Control Pictures block so they can be seen."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp < 0x20 and ch not in "\t\n":
            out.append(chr(_CONTROL_PICTURE_BASE + cp))
        elif cp == 0x7F:
            out.append("␡")
        elif 0x80 <= cp <= 0x9F:
            out.append("�")
        else:
            out.append(ch)
    return "".join(out)


def safe_snippet(text: str, limit: int = 400) -> str:
    """The only way corpus text is allowed into a report."""
    if not text:
        return ""
    cleaned = _BIDI.sub("␣", text)
    cleaned = visible_controls(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return cleaned


def escape_html(text: str) -> str:
    return _html.escape(text, quote=True)


_SCRIPT_UNSAFE = (
    ("<", "\\u003c"),
    (">", "\\u003e"),
    ("&", "\\u0026"),
    (" ", "\\u2028"),
    (" ", "\\u2029"),
)


def json_script_payload(obj: object) -> str:
    """Serialise for embedding inside a script element.

    The replacements keep the output valid JSON while making it impossible for
    the payload to terminate the element it lives in.
    """
    raw = _json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    for bad, good in _SCRIPT_UNSAFE:
        raw = raw.replace(bad, good)
    return raw
