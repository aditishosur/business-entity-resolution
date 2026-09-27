"""Shared local-only normalization helpers and explainable pairwise features."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Mapping

LEGAL_SUFFIXES = frozenset({"co", "company", "inc", "incorporated", "ltd", "limited", "llc", "corp", "corporation", "pvt", "private", "plc"})
HOUSE_NUMBER_RE = re.compile(r"(?<![a-z0-9])(\d{1,5}[a-z]?)(?![a-z0-9])")


def token_set(text: str) -> set[str]:
    return set(text.split()) if text else set()


def legal_name_tokens(name: str) -> tuple[str, ...]:
    """Strip only trailing legal designators from an already-normalized name."""
    values = name.split()
    while values and values[-1] in LEGAL_SUFFIXES:
        values.pop()
    return tuple(values)


def legal_name(name: str) -> str:
    return " ".join(legal_name_tokens(name))


def char_ngram_similarity(left: str, right: str, n: int = 3) -> float:
    """Dice similarity on n-gram sets; exact comparison for short strings."""
    if not left or not right:
        return 0.0
    if len(left) < n or len(right) < n:
        return float(left == right)
    left_grams = {left[i:i + n] for i in range(len(left) - n + 1)}
    right_grams = {right[i:i + n] for i in range(len(right) - n + 1)}
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def house_numbers(address: str, postal_code: str = "") -> set[str]:
    """Extract bounded address numbers while excluding the recognized postal code."""
    values = {match.group(1) for match in HOUSE_NUMBER_RE.finditer(address or "")}
    values.discard(postal_code)
    return values


def _value(record: Mapping[str, str] | object, key: str) -> str:
    if isinstance(record, Mapping):
        return str(record.get(key, "") or "")
    return str(getattr(record, key, "") or "")


def feature_row(left: Mapping[str, str] | object, right: Mapping[str, str] | object) -> list[float]:
    """Feature vector shared by both solvers and validation (15 values)."""
    ln, rn = _value(left, "name_n"), _value(right, "name_n")
    la, ra = _value(left, "address_n"), _value(right, "address_n")
    lp, rp = _value(left, "postal_n"), _value(right, "postal_n")
    lc, rc = _value(left, "country_n"), _value(right, "country_n")
    lnt, rnt, lat, rat = token_set(ln), token_set(rn), token_set(la), token_set(ra)
    ni, ai = len(lnt & rnt), len(lat & rat)
    lu, au = len(lnt | rnt), len(lat | rat)
    lh, rh = house_numbers(la, lp), house_numbers(ra, rp)
    return [
        float(bool(ln and ln == rn)), float(bool(la and la == ra)), float(bool(lp and lp == rp)),
        ni / lu if lu else 0.0, ai / au if au else 0.0,
        ni / min(len(lnt), len(rnt)) if lnt and rnt else 0.0,
        ai / min(len(lat), len(rat)) if lat and rat else 0.0,
        SequenceMatcher(None, ln, rn).ratio() if ln and rn else 0.0,
        SequenceMatcher(None, la, ra).ratio() if la and ra else 0.0,
        float(bool(lc and lc == rc)),
        float(bool(legal_name(ln) and legal_name(ln) == legal_name(rn))),
        float(bool(lnt and lnt == rnt)), char_ngram_similarity(ln, rn), char_ngram_similarity(la, ra),
        float(bool(lh and rh and lh == rh)),
    ]
