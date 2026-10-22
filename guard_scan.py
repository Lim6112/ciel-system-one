"""The deterministic half of the guard: finding the spans a model cannot point at.

A model can answer "does this text contain PII: yes/no". It cannot say WHICH span
leaked, and it cannot be trusted to notice a secret it just generated. So a scanner
runs alongside it, and the two answer different questions: the scanner finds the
spans, the model catches what no pattern can see (a name in free text, a medical
condition, a salary).

THREE THINGS THAT ARE EASY TO GET WRONG, AND EACH ONE COSTS REAL REQUESTS.

A pattern that is not anchored matches ordinary English. An unanchored
`sk-[A-Za-z0-9_-]{16,}` matches the "sk-" in "task-oriented-approach",
"risk-management-framework" and "brisk-walking-in-the-park" — and a guard that
blocks those is a guard that rejects ordinary text. The provider patterns are
`\b`-anchored.

A pattern that is not specific matches ordinary NUMBERS. A phone number and a
number group have the same shape, so matching both blocks "Where is my order
123 456 7890?" and "the build ran for 1 234 567 8901 iterations". The phone
pattern therefore requires a signal a number group does not have: a country code, a
parenthesised area code, or a hyphen. A space-separated group with none of those is
NOT matched — a deliberate false NEGATIVE, because a missed phone number is
recoverable and a blocked order number is not.

And a pattern that is not BOUNDED is not linear. The email pattern's local part is
capped at 64 characters for exactly this reason: unbounded, a long run of "a." with
no "@" makes the engine retry from every start offset — 3.25 s for 64 KB, and the
scan runs under the server-wide lock, so one such request stalls every concurrent
request behind it.

This module owns the vocabulary of what the guard finds (the categories, the Finding
type and the scanner itself) and nothing else. The detectors and the policy live in
guard_pii.py, which imports from here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

# The guard's own vocabulary for what it found.
PII_CATEGORY = "pii"
SECRET_CATEGORY = "secret"
JAILBREAK_CATEGORY = "jailbreak"
INJECTION_CATEGORY = "injection"
CREDENTIAL_CATEGORY = "credential"
# A detector that could not be reached. Reported, never blocking — a missing detector
# is a gap in coverage, not evidence that the text is clean, and the two must not look
# the same in the trace.
UNAVAILABLE_CATEGORY = "detector_unavailable"


@dataclass(frozen=True)
class Finding:
    """One detection: what fired, how strongly, and on which text.

    `detail` never carries a raw secret. The scanner masks what it matches
    (`sk-p...`) so that a finding can be logged, returned over HTTP and put in a
    test assertion without copying the credential into three more places.
    """

    detector: str
    category: str
    score: float
    detail: str


# Provider key shapes. These are prefixes, not complete secrets, so the match is a
# giveaway that a secret is present rather than a way to use it.
#
# The provider prefixes are \b-anchored and may not match in the middle of a word. That
# is not fussiness: an unanchored `sk-[A-Za-z0-9_-]{16,}` matches the "sk-" in
# "task-oriented-approach", "risk-management-framework" and "brisk-walking-in-the-park",
# and a guard that blocks those is a guard that rejects ordinary English.
#
# The ASSIGNED pattern is the exception, and it is anchored differently on purpose. It
# looks for `token = value`, and in real code the keyword is usually part of a longer
# identifier — `access_token=`, `client_secret=`, `DB_PASSWORD=`, `accessToken=`. A
# `\b` before the keyword fails on all of those, because the character before the
# keyword is `_` or a letter. So it is anchored on "not a letter or digit" instead,
# which admits the underscore and the camelCase join, and still refuses to match the
# tail of an unrelated word.
_SECRET_PATTERNS: Sequence[tuple[str, str]] = (
    # Modern OpenAI project keys and Anthropic keys: a known prefix, then a long
    # base62 run. The legacy `sk-` form is base62 only — no hyphen — so it cannot
    # swallow the next word of a hyphenated phrase.
    ("openai", r"\bsk-proj-[A-Za-z0-9_-]{20,}"),
    ("anthropic", r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    ("openai_legacy", r"\bsk-[A-Za-z0-9]{32,}"),
    ("openrouter", r"\bsk-or-v1-[A-Za-z0-9]{32,}"),
    ("stripe", r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    ("github", r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    ("gitlab", r"\bglpat-[A-Za-z0-9_-]{16,}"),
    ("npm", r"\bnpm_[A-Za-z0-9]{30,}"),
    ("huggingface", r"\bhf_[A-Za-z0-9]{30,}"),
    ("aws", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("slack", r"\bxox[abprsdc]-[A-Za-z0-9-]{10,}"),
    ("google", r"\bAIza[0-9A-Za-z_-]{35}"),
    ("bearer", r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}"),
    # `password = "…"` and friends. The value class includes +, / and = because base64
    # secrets end in = and contain + and /. The keyword may be a whole identifier or
    # the tail of one: `access_token=`, `aws_secret_access_key=`, `accessToken=`. So
    # the anchor is "not a letter or digit" — which admits the underscore — and the
    # keyword itself may be split by _, - or a space.
    #
    # The camelCase form has no separator at all (`accessToken=`), so it needs its own
    # pattern: a lowercase run, an uppercase letter, then the keyword. It is written
    # separately rather than folded into the one above because a single pattern that
    # admits "no separator" would also admit a word that merely ends in "token".
    ("assigned_secret", r"(?i)(?<![A-Za-z0-9])(?:[A-Za-z0-9]+[ _-]){0,2}(?:api[ _-]?key|access[ _-]?key|secret|password|passwd|token|credential)\s*[:=]\s*[\"']?[A-Za-z0-9._+/=-]{16,}"),
    ("assigned_secret_camel", r"(?<![A-Za-z0-9])[a-z]+(?:ApiKey|Secret|Password|Passwd|Token|Credential)\s*[:=]\s*[\"']?[A-Za-z0-9._+/=-]{16,}"),
    # A PEM private key block, which is a secret in its entirety.
    ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

# PII shapes. These are the spans the model cannot point at, so the scanner is what
# makes "which part leaked" answerable at all.
#
# THE PHONE PATTERN IS THE CAREFUL ONE. A phone number and a number group look alike:
# "555 123 4567" and "123 456 7890" are the same shape. A scanner that cannot tell
# them apart blocks "Where is my order 123 456 7890?" and "The build ran for
# 1 234 567 8901 iterations" — ordinary text, blocked on every arm at every stage.
#
# So the shape has to carry a signal that a number group does not have:
#   - a country code (+1, +44)
#   - a parenthesised area code ((555))
#   - a hyphen (555-123-4567) — a number group is not hyphenated that way
# A space-separated group with no country code and no parentheses is NOT matched. That
# is a deliberate false NEGATIVE: a missed phone number is recoverable, a blocked
# order number is not.
#
# The email pattern is bounded on both sides. Left unbounded, a long run of
# "a." with no "@" makes the engine retry from every start offset — measured at
# 3.25 s for 64 KB, and the scan runs under the server-wide lock.
_PII_PATTERNS: Sequence[tuple[str, str]] = (
    ("email", r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,63}\b"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("card", r"\b(?:\d[ -]*?){13,19}\b"),
    ("phone", r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3}-\d{3}-\d{4})(?![\d.])"),
)

_COMPILED_SECRETS = tuple((name, re.compile(pattern)) for name, pattern in _SECRET_PATTERNS)
_COMPILED_PII = tuple((name, re.compile(pattern)) for name, pattern in _PII_PATTERNS)


def _luhn_ok(digits: str) -> bool:
    """The Luhn checksum, which is what separates a card number from a long number."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _mask(text: str) -> str:
    """A finding detail that identifies the match without reproducing it.

    Only the LEADING characters are kept. The label already says what kind of thing
    was found ("ssn", "card", "openai"), so the leading characters serve to confirm the
    shape — and the trailing ones are the part of a value that actually narrows it
    down. Revealing the last two of an SSN gives away 5 of its 9 digits, and revealing
    the last two of a key is two characters of the secret itself. A finding travels to
    the browser and into the trace, so it should not carry a slice of the thing it
    found.
    """
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}..."


class SecretScanner:
    """Finds the spans the model cannot: keys, cards, identifiers.

    Arm-independent by design. The model arms disagree with each other about PII —
    0.843 on english against 0.496 on the doc head — but they cannot disagree that
    `sk-proj-abc123...` is a secret. That is a fact about the format, not a calibrated
    probability, so it is not something a threshold should tune.
    """

    name = "regex"

    def scan(self, text: str) -> List[Finding]:
        if not text:
            return []
        findings: List[Finding] = []
        for name, pattern in _COMPILED_SECRETS:
            for match in pattern.finditer(text):
                findings.append(
                    Finding(self.name, SECRET_CATEGORY, 1.0, f"{name}: {_mask(match.group(0))}")
                )
        for name, pattern in _COMPILED_PII:
            for match in pattern.finditer(text):
                raw = match.group(0)
                if name == "card":
                    digits = re.sub(r"\D", "", raw)
                    # Luhn, so a 13-19 digit number that is not a card is not reported as
                    # one. Without it every long order number is a "credit card".
                    if not (13 <= len(digits) <= 19) or not _luhn_ok(digits):
                        continue
                findings.append(
                    Finding(self.name, PII_CATEGORY, 1.0, f"{name}: {_mask(raw)}")
                )
        return findings
