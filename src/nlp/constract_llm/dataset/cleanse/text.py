import logging
import re
from functools import partial
from multiprocessing import cpu_count
from typing import Any

from datasketch import MinHash, MinHashLSH
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


def is_only_numeric(text: int | float | str) -> bool:
    return (isinstance(text, int) or isinstance(text, float)) or (
        text.replace('.', '').isdecimal() and text.replace('.', '').isascii()
    )


def is_out_of_length_range(text: str, min_str_len: int = 0, max_str_len: int | None = None) -> bool:
    text_len = len(text)

    is_shorter_than_min = text_len < min_str_len
    is_longer_than_max = max_str_len is not None and text_len > max_str_len

    return is_shorter_than_min or is_longer_than_max


def is_include_url(text: str) -> bool:
    return is_match_pattern(text, URL_PATTERN)


def is_include_email(text: str) -> bool:
    return is_match_pattern(text, EMAIL_PATTERN)


def judge_include_time_schedule(
    text: str,
    pattern: re.Pattern | str = TIME_PATTRN,
    threshold: int = 3,
) -> bool:
    return len(pattern.findall(text)) >= threshold


def build_minhash_index(
    texts: list[str],
    num_perm: int = 128,
    threshold: float = 0.95,
    num_workers: int | None = None,
):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    valid_texts = [t for t in texts if t is not None]

    builder = partial(build_minhash, num_perm=num_perm)
    minhashes = process_map(
        builder,
        valid_texts,
        max_workers=num_workers or cpu_count(),
        desc='Build MinHash',
    )

    for text, m in zip(valid_texts, minhashes):
        lsh.insert(text, m)

    return lsh


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    assert text is not None, 'text must be not None'
    for d in text:
        m.update(d.encode('utf8'))
    return m


def find_similar_strings(
    strings: str,
    num_perm: int = 128,
    lsh: MinHashLSH | None = None,
) -> str | None:
    assert lsh is not None, 'lsh must be not None'
    assert strings is not None, 'strings must be not None'
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
    cleansed = []
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
    cleansed = []
    mother_set = set()
    for text in texts:
        if text not in mother_set:
            cleansed.append(text)
            mother_set.add(text)
        else:
            cleansed.append(None)
    return cleansed


def cleanse_column_duplicates(
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
    assert len(dataset) == len(cleansed_texts)

    if do_rm_duplicated_by_minhash:
        cleansed_texts = cleansed_duplicated_texts_by_minhash(
            cleansed_texts, threshold=threshold, num_perm=num_perm, num_workers=num_workers
        )
        deduplicated_texts_count_by_minhash = cleansed_texts.count(None) - num_cleanse_texts
        total_deduplicated_texts_count += deduplicated_texts_count_by_minhash
        logger.info(f'Total deduplicated texts count by MinHash for {col}: {deduplicated_texts_count_by_minhash}')

    logger.info(f'Total deduplicated texts count for {col}: {total_deduplicated_texts_count}')
    return cleansed_texts, total_deduplicated_texts_count


def cleanse_text(
    text: str,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
) -> str | None:
    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    if (
        (do_rm_only_numeric and is_only_numeric(text))
        or (do_rm_time_schedule and judge_include_time_schedule(text, threshold=rm_time_schedule_threshold))
        or (do_rm_include_url_text and is_include_url(text))
        or (do_rm_include_email_text and is_include_email(text))
    ):
        return None

    return text
