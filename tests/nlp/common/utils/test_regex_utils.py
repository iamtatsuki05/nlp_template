import re

import pytest

from nlp.common.utils.regex_utils import concat, is_match_pattern, unmatched_group


@pytest.mark.parametrize(
    'input_regex,expected',
    [
        ('abc', r'(?:abc)'),
        ('a|b', r'(?:a|b)'),
        (r'\d+', r'(?:\d+)'),
        ('', r'(?:)'),
        ('a{2,3}', r'(?:a{2,3})'),
    ],
)
def test_unmatched_group(input_regex, expected):
    result = unmatched_group(input_regex)
    assert result == expected


@pytest.mark.parametrize(
    'regexes,without_grouping,expected',
    [
        (['a', 'b'], False, r'(?:a|b)'),
        (['a', 'b'], True, r'a|b'),
        ([r'\d+', r'\w+'], False, r'(?:\d+|\w+)'),
        ([], False, r'(?:)'),
        (['abc'], False, r'(?:abc)'),
        (['abc'], True, r'abc'),
    ],
)
def test_concat(regexes, without_grouping, expected):
    result = concat(regexes, without_grouping)
    assert result == expected


@pytest.mark.parametrize(
    'text,pattern,expected',
    [
        ('abc123', re.compile(r'\d+'), True),
        ('abcdef', re.compile(r'\d+'), False),
        ('', re.compile(r'.*'), True),
        ('test@example.com', re.compile(r'@'), True),
        ('Hello World', re.compile(r'[a-z]+'), True),
    ],
)
def test_is_match_pattern(text, pattern, expected):
    result = is_match_pattern(text, pattern)
    assert result == expected
