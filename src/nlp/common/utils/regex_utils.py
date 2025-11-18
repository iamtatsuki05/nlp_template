import re
from collections.abc import Sequence


def unmatched_group(_regex: str) -> str:
    return r'(?:' + _regex + r')'


def concat(regexes: Sequence[str], without_grouping: bool = False) -> str:
    _regex = r'|'.join(regexes)
    if not without_grouping:
        _regex = unmatched_group(_regex)
    return _regex


def is_match_pattern(text: str, pattern: re.Pattern[str]) -> bool:
    return re.search(pattern, text) is not None
