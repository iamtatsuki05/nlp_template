from __future__ import annotations

import logging
from functools import partial
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from datasketch import MinHash, MinHashLSH
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)


def build_minhash_index(
    texts: Sequence[str | None],
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
