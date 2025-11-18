from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

from nlp.common.regex import EMAIL_PATTERN, TIME_PATTRN, URL_PATTERN
from nlp.common.utils.regex_utils import is_match_pattern

if TYPE_CHECKING:
    import re


def is_blank(text: str) -> bool:
    return text.strip() == ''


def is_only_numeric(text: float | str) -> bool:
    if isinstance(text, (int, float)):
        return True
    normalized = text.replace('.', '')
    return normalized.isdecimal() and normalized.isascii()


def is_out_of_length_range(text: str, min_str_len: int = 0, max_str_len: int | None = None) -> bool:
    text_len = len(text)

    is_shorter_than_min = text_len < min_str_len
    is_longer_than_max = max_str_len is not None and text_len > max_str_len

    return is_shorter_than_min or is_longer_than_max


def is_include_url(text: str) -> bool:
    return is_match_pattern(text, URL_PATTERN)


def is_include_email(text: str) -> bool:
    return is_match_pattern(text, EMAIL_PATTERN)


class TextRule(Protocol):
    def should_remove(self, text: str) -> bool: ...


class OnlyNumericRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    def should_remove(self, text: str) -> bool:
        return is_only_numeric(text)


class TimeScheduleRule(BaseModel):
    threshold: int = 3
    model_config = ConfigDict(frozen=True)

    def should_remove(self, text: str) -> bool:
        return judge_include_time_schedule(text, threshold=self.threshold)


class UrlRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    def should_remove(self, text: str) -> bool:
        return is_include_url(text)


class EmailRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    def should_remove(self, text: str) -> bool:
        return is_include_email(text)


def judge_include_time_schedule(
    text: str,
    pattern: re.Pattern[str] = TIME_PATTRN,
    threshold: int = 3,
) -> bool:
    return len(pattern.findall(text)) >= threshold
