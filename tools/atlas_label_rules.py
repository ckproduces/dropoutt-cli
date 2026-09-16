"""Rules every hand-written atlas name is checked against before it is stamped.

Written after the 13 Sep 2026 audit of atlas-v3, which found three ways a name
can mislead without being a typo:

* **Soup names.** "Linux hardening logs, Brazilian court appeals and
  sentence-rewriting prompts" lists the subjects of the few records nearest a
  cell's centre. When a cell holds no shared subject, those records are close to
  random, and the list reads like a subject that is not there.
* **Letter names.** "Names and entries starting with T" describes an encoder
  artifact faithfully and tells a reader nothing about content.
* **Narrow names.** "La Liga footballer biography stubs" for a cell of players
  from every league. A blind reader found the name narrower than the cell for
  41% of a 160-cell sample.

So every name now carries a **kind**, and the kind constrains the wording:

``subject``
    Most members share one subject. Name it at the level the members support.
``form``
    Members share a format, genre or template but not a subject. Name the form,
    and say the subjects vary ("Pipe-delimited tables on assorted subjects").
``mixed``
    Members share neither. The name starts with "Mixed" and says what little
    they do share ("Mixed short web fragments with no shared subject").
"""

from __future__ import annotations

import re

KINDS = ("subject", "form", "mixed")

#: Spellings of a letter-initial name. A cell held together by first letters is
#: mixed, and its name says so.
_LETTER_NAME = re.compile(
    r"\b[A-Z]-initial\b|\bstarting with (?:the letter )?[A-Z]\b|"
    r"\bbeginning with (?:the letter )?[A-Z]\b|\b[A-Z]-names?\b",
)

#: Language names. A name may contain one only when the language is the subject.
LANGUAGE_WORDS = {
    "afrikaans", "albanian", "arabic", "armenian", "azerbaijani", "basque", "belarusian",
    "bengali", "bosnian", "bulgarian", "catalan", "chinese", "croatian", "czech",
    "danish", "dutch", "english", "estonian", "finnish", "french", "galician",
    "georgian", "german", "greek", "gujarati", "hebrew", "hindi", "hungarian",
    "icelandic", "indonesian", "irish", "italian", "japanese", "kannada", "kazakh",
    "korean", "latvian", "lithuanian", "macedonian", "malay", "malayalam", "marathi",
    "norwegian", "persian", "polish", "portuguese", "punjabi", "romanian", "russian",
    "serbian", "sinhala", "slovak", "slovenian", "spanish", "swahili", "swedish",
    "tagalog", "tamil", "telugu", "thai", "turkish", "ukrainian", "urdu", "uzbek",
    "vietnamese", "welsh",
}
#: Words that make a language the subject rather than the medium.
_LANGUAGE_AS_SUBJECT = re.compile(
    r"\b(language|languages|linguistic|linguistics|grammar|dialects?|vocabulary|"
    r"literature|literary|poetry|translation|translations|script|alphabet|"
    r"learning|lessons|etymolog\w*)\b",
    re.IGNORECASE,
)


def list_items(name: str) -> list[str]:
    """The comma- and "and"-separated parts of a name."""
    return [part for part in re.split(r",\s*|\s+and\s+|\s*&\s*", name) if part.strip()]


def check_name(name: str, kind: str | None) -> tuple[list[str], list[str]]:
    """Errors (refuse to stamp) and warnings (print, stamp anyway) for one name."""
    errors: list[str] = []
    warnings: list[str] = []
    text = name.strip()
    if not text:
        return ["blank name"], warnings
    if kind not in KINDS:
        errors.append(f"kind {kind!r} is not one of {', '.join(KINDS)}")
    if _LETTER_NAME.search(text):
        errors.append("names a letter, not what the records are about; a letter-held cell is mixed")
    if kind == "mixed" and not text.lower().startswith("mixed"):
        errors.append('a mixed cell\'s name starts with "Mixed"')
    if kind in ("form", "mixed") and len(list_items(text)) >= 3:
        errors.append("lists three or more items for a cell with no shared subject")
    if kind == "subject" and text.lower().startswith("mixed"):
        errors.append('a subject cell is not named "Mixed"')
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", text)}
    languages = sorted(words & LANGUAGE_WORDS)
    if languages and not _LANGUAGE_AS_SUBJECT.search(text):
        warnings.append(
            f"names a language ({', '.join(languages)}) without making it the subject; "
            "name the place or the subject instead"
        )
    if kind == "subject" and len(list_items(text)) >= 4:
        warnings.append("four or more listed items; check they are facets of one subject")
    return errors, warnings
