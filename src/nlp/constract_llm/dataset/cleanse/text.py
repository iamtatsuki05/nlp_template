import logging
import re
from collections.abc import Sequence
from functools import partial
from multiprocessing import cpu_count
from typing import Any, Protocol

from datasketch import MinHash, MinHashLSH
from pydantic import BaseModel, ConfigDict
from tqdm.contrib.concurrent import process_map

from nlp.common.regex import (
    EMAIL_PATTERN,
    TIME_PATTRN,
    URL_PATTERN,
)
from nlp.common.utils.regex_utils import is_match_pattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class TextCleaner:
    def __init__(self, rules: Sequence[TextRule]) -> None:
        self._rules: tuple[TextRule, ...] = tuple(rules)

    def clean(self, text: str | None) -> str | None:
        if text is None:
            return None

        stripped = text.strip()
        if not stripped:
            return None

        for rule in self._rules:
            if rule.should_remove(stripped):
                return None

        return stripped


def create_text_cleaner(
    *,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
) -> TextCleaner:
    rules: list[TextRule] = []
    if do_rm_only_numeric:
        rules.append(OnlyNumericRule())
    if do_rm_time_schedule:
        rules.append(TimeScheduleRule(threshold=rm_time_schedule_threshold))
    if do_rm_include_url_text:
        rules.append(UrlRule())
    if do_rm_include_email_text:
        rules.append(EmailRule())
    return TextCleaner(rules)


def judge_include_time_schedule(
    text: str,
    pattern: re.Pattern[str] = TIME_PATTRN,
    threshold: int = 3,
) -> bool:
    return len(pattern.findall(text)) >= threshold


def build_minhash_index(
    texts: list[str | None],
    num_perm: int = 128,
    threshold: float = 0.95,
    num_workers: int | None = None,
) -> MinHashLSH:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    valid_texts = [t for t in texts if t is not None]

    builder = partial(build_minhash, num_perm=num_perm)
    minhashes = process_map(
        builder,
        valid_texts,
        max_workers=num_workers or cpu_count(),
        desc='Build MinHash',
    )

    for text, m in zip(valid_texts, minhashes, strict=True):
        lsh.insert(text, m)

    return lsh


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for d in text:
        m.update(d.encode('utf8'))
    return m


def find_similar_strings(
    strings: str,
    num_perm: int = 128,
    lsh: MinHashLSH | None = None,
) -> str | None:
    if lsh is None:
        msg = 'lsh must be provided'
        raise ValueError(msg)
    m = build_minhash(strings, num_perm=num_perm)
    result = lsh.query(m)
    if len(result) > 0:
        return result[0]
    return None


def _cleanse_candidate(text: str | None, lsh: MinHashLSH, num_perm: int = 128) -> str | None:
    if text is None:
        return None

    similar = find_similar_strings(text, lsh=lsh, num_perm=num_perm)
    if similar is not None and similar != text:
        return None
    return text


def cleansed_duplicated_texts_by_minhash(
    texts: list[str | None],
    threshold: float = 0.95,
    num_perm: int = 128,
    num_workers: int | None = None,
) -> list[str | None]:
    lsh = build_minhash_index(texts, threshold=threshold, num_perm=num_perm, num_workers=num_workers)

    worker = partial(_cleanse_candidate, lsh=lsh, num_perm=num_perm)

    chunksize = max(1, len(texts) // (cpu_count() * 4))

    intermediate = process_map(
        worker,
        texts,
        max_workers=num_workers or cpu_count(),
        chunksize=chunksize,
        desc='Cleanse duplicated samples with MinHash',
    )

    cleansed: list[str | None] = []
    for cand in intermediate:
        if cand is None or cand in cleansed:
            cleansed.append(None)
        else:
            cleansed.append(cand)

    return cleansed


def cleansed_duplicated_texts(texts: list[str | None]) -> list[str | None]:
    cleansed: list[str | None] = []
    mother_set = set()
    for text in texts:
        if text not in mother_set:
            cleansed.append(text)
            mother_set.add(text)
        else:
            cleansed.append(None)
    return cleansed


def cleanse_column_duplicates(  # noqa: PLR0913
    dataset: list[dict[str, Any]],
    col: str,
    do_rm_duplicated_by_minhash: bool = True,
    threshold: float = 0.95,
    num_perm: int = 128,
    num_workers: int | None = None,
) -> tuple[list[str | None], int]:
    total_deduplicated_texts_count = 0
    texts = [sample[col] for sample in dataset]
    cleansed_texts = cleansed_duplicated_texts(texts)
    num_none_texts = texts.count(None)
    num_cleanse_texts = cleansed_texts.count(None)
    total_deduplicated_texts_count += num_cleanse_texts - num_none_texts
    logger.info(f'Total deduplicated texts count by duplicated_texts for {col}: {total_deduplicated_texts_count}')
    if len(dataset) != len(cleansed_texts):
        logger.error(f'Length mismatch: dataset({len(dataset)}) vs cleansed_texts({len(cleansed_texts)})')
        raise ValueError('Length mismatch between dataset and cleansed_texts')

    if do_rm_duplicated_by_minhash:
        cleansed_texts = cleansed_duplicated_texts_by_minhash(
            cleansed_texts, threshold=threshold, num_perm=num_perm, num_workers=num_workers
        )
        deduplicated_texts_count_by_minhash = cleansed_texts.count(None) - num_cleanse_texts
        total_deduplicated_texts_count += deduplicated_texts_count_by_minhash
        logger.info(f'Total deduplicated texts count by MinHash for {col}: {deduplicated_texts_count_by_minhash}')

    logger.info(f'Total deduplicated texts count for {col}: {total_deduplicated_texts_count}')
    return cleansed_texts, total_deduplicated_texts_count


def cleanse_text(  # noqa: PLR0913
    text: str | None,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
    text_cleaner: TextCleaner | None = None,
) -> str | None:
    """Cleanse a single text string by applying text cleaning rules.

    Args:
        text: Text to cleanse.
        do_rm_time_schedule: Whether to remove texts containing time schedules.
        rm_time_schedule_threshold: Minimum time patterns to trigger removal.
        do_rm_only_numeric: Whether to remove numeric-only texts.
        do_rm_include_url_text: Whether to remove texts with URLs.
        do_rm_include_email_text: Whether to remove texts with email addresses.
        text_cleaner: Optional pre-configured TextCleaner instance. If None, creates one from parameters.

    Returns:
        Cleansed text or None if the text should be removed.

    """
    cleaner = text_cleaner or create_text_cleaner(
        do_rm_time_schedule=do_rm_time_schedule,
        rm_time_schedule_threshold=rm_time_schedule_threshold,
        do_rm_only_numeric=do_rm_only_numeric,
        do_rm_include_url_text=do_rm_include_url_text,
        do_rm_include_email_text=do_rm_include_email_text,
    )
    return cleaner.clean(text)
