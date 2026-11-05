"""Numeric workout notation, independent of intent and exercise dictionaries.

The model chooses tasks/entities. This module only checks that typed values
occur in the source with compatible units or a complete multiplication tuple.
"""
from __future__ import annotations
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NumberToken:
    value: float
    unit: str
    start: int
    end: int


_MULTIPLICATION_RE = re.compile(r"\s*(?:x|X|×|\*|＊)\s*")
_DATE_TOKEN_RE = re.compile(
    r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}月\d{1,2}日"
    r"|[零〇一二三四五六七八九十]{1,3}[年月][零〇一二三四五六七八九十]{1,3}日?")


def _chain_gap(gap: str) -> bool:
    """True only for a multiplication operator, including reasonable spacing."""
    return _MULTIPLICATION_RE.fullmatch(gap) is not None


def numeric_token_spans(text: str) -> list[tuple[int, int]]:
    """Positions of numeric tokens used by the finite workout syntax.

    Dates are excluded because they are metadata, not weight/reps/sets
    quantities. The caller uses these spans only to prove that a multi-action
    source was covered by independently committed action refs.
    """
    from app.services.agent.workout_extractor import NUMBER_TOKEN

    date_spans = [match.span() for match in _DATE_TOKEN_RE.finditer(text)]
    spans: list[tuple[int, int]] = []
    for match in re.finditer(NUMBER_TOKEN, text):
        if any(start <= match.start() and match.end() <= end for start, end in date_spans):
            continue
        if text[match.end():match.end() + 1] == "项":
            # "两项动作" counts targets, not a weight/reps/sets quantity.
            continue
        spans.append((match.start(), match.end()))
    return spans


def source_values(text):
    from app.services.agent.workout_extractor import (
        NUMBER_TOKEN, UNIT_PATTERN, REPS_UNIT_PATTERN, SETS_UNIT_PATTERN,
        BODYWEIGHT_TOKENS, chinese_number_to_value, _as_positive_int,
    )
    pattern = re.compile(rf"(?P<sign>[-−负]?)(?P<number>{NUMBER_TOKEN})(?P<unit>{UNIT_PATTERN}|{REPS_UNIT_PATTERN}|{SETS_UNIT_PATTERN})?", re.IGNORECASE)
    tokens = []
    # 自重/徒手 occupies the weight position, so "自重*10*4" is a complete
    # weight x reps x sets chain rather than two bare counts (sets x reps).
    for match in re.finditer("|".join(map(re.escape, BODYWEIGHT_TOKENS)), text):
        tokens.append(NumberToken(0.0, "kg", match.start(), match.end()))
    for match in pattern.finditer(text):
        value = chinese_number_to_value(match.group("number"))
        if value is None:
            continue
        value = float(value) * (-1 if match.group("sign") else 1)
        unit = (match.group("unit") or "").lower()
        if unit in {"公斤", "千克", "kg"}: unit = "kg"
        elif unit in {"lb", "磅"}: unit = "lb"
        elif unit == "斤": unit = "jin"
        elif unit in {"组", "套"}: unit = "sets"
        elif unit in {"次", "个", "下", "回", "rep", "reps"}: unit = "reps"
        else: unit = "bare"
        tokens.append(NumberToken(value, unit, match.start(), match.end()))
    tokens.sort(key=lambda token: (token.start, token.end))
    values = {"weight_kg": [], "sets": [], "reps": []}
    for token in tokens:
        if token.unit in {"kg", "jin", "lb"}:
            values["weight_kg"].append(token.value * {"kg": 1, "jin": .5, "lb": .45359237}[token.unit])
        elif token.unit in {"sets", "reps"}:
            number = _as_positive_int(token.value)
            if number is not None:
                values[token.unit].append(number)
        elif text[:token.start].endswith("每组"):
            number = _as_positive_int(token.value)
            if number is not None:
                values["reps"].append(number)

    bodyweight_positions = [
        match.start() for match in
        re.finditer("|".join(map(re.escape, BODYWEIGHT_TOKENS)), text)
    ]
    separators = "，,；;。\n"

    def clause_start(position: int) -> int:
        return max((text.rfind(separator, 0, position) for separator in separators),
                   default=-1) + 1

    chains = []
    for token in tokens:
        if chains and _chain_gap(text[chains[-1][-1].end:token.start]):
            chains[-1].append(token)
        else:
            chains.append([token])
    for chain in chains:
        pair = chain
        if len(chain) == 3 and chain[0].unit in {"bare", "kg", "jin", "lb"}:
            # Complete weight × reps × sets tuple; explicit labels override
            # the order of the last two factors. Bare weight uses app kg units.
            if chain[0].unit == "bare":
                values["weight_kg"].append(chain[0].value)
            pair = chain[1:]
            default_order = ("reps", "sets")
        elif len(chain) == 2:
            if chain[0].unit in {"kg", "jin", "lb"} and chain[1].unit in {"bare", "reps"}:
                # Weight × reps, with the set count specified elsewhere.
                number = _as_positive_int(chain[1].value)
                if number is not None:
                    values["reps"].append(number)
                continue
            # A bodyweight marker earlier in the same clause occupies the
            # weight position even when the action name sits between it and
            # the numbers ("徒手深蹲×12×3", "自重引体向上×12×3"). Without the
            # marker, two bare numbers keep the existing sets×reps contract.
            start = clause_start(chain[0].start)
            ends = [text.find(separator, chain[-1].end) for separator in separators]
            ends = [end for end in ends if end >= 0]
            end = min(ends) if ends else len(text)
            bodyweight_clause = any(start <= position < end
                                    for position in bodyweight_positions)
            default_order = ("reps", "sets") if bodyweight_clause else ("sets", "reps")
        else:
            continue
        if any(token.unit not in {"bare", "sets", "reps"} for token in pair):
            continue
        first, second = pair
        if first.unit in {"sets", "reps"}:
            order = (first.unit, "reps" if first.unit == "sets" else "sets")
        elif second.unit in {"sets", "reps"}:
            order = ("reps" if second.unit == "sets" else "sets", second.unit)
        else:
            order = default_order
        # Two labels naming the same role cannot supply the missing role.
        if any(token.unit != "bare" and token.unit != role for token, role in zip(pair, order)):
            continue
        for token, role in zip(pair, order):
            number = _as_positive_int(token.value)
            if number is not None:
                values[role].append(number)
    return {role: list(dict.fromkeys(items)) for role, items in values.items()}
