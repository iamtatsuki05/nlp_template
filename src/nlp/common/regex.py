import re
from typing import Final

from nlp.common.utils.regex_utils import concat, unmatched_group

# time regex
TIME_PATTRN: Final[re.Pattern[str]] = re.compile(r'\d+:\d+')


# email regex
LOCAL_PART_CHARS = r'[\w\-._]'
DOMAIN_CHARS = r'[\w\-._]'
TLD_CHARS = r'[A-Za-z]'

local_part = concat([LOCAL_PART_CHARS], without_grouping=True) + r'+'
domain = concat([DOMAIN_CHARS], without_grouping=True) + r'+'
tld = concat([TLD_CHARS], without_grouping=True) + r'+'

EMAIL_REGEX = local_part + r'@' + domain + r'\.' + tld
EMAIL_PATTERN = re.compile(EMAIL_REGEX)


# url regex
SCHEME = r'https?'
CHARS = r'[\w!?/+\-_~;.,*&@#$%()\[\]]'

url_chars = concat([CHARS], without_grouping=True) + r'+'

HTTP_URL_REGEX = SCHEME + r'://' + url_chars

DATA_SCHEME = r'data:'
MEDIATYPE = r'[\w/+.-]+'
BASE64 = r'base64'
DATA = r'[\w+/=]+'

mediatype_part = unmatched_group(MEDIATYPE) + r'?'
base64_part = unmatched_group(BASE64) + r'?'
data_part = unmatched_group(DATA)

DATA_URL_REGEX = DATA_SCHEME + mediatype_part + r'(?:;' + base64_part + r')?,' + data_part

URL_REGEX = concat([HTTP_URL_REGEX, DATA_URL_REGEX])

HTTP_URL_PATTERN = re.compile(HTTP_URL_REGEX)
DATA_URL_PATTERN = re.compile(DATA_URL_REGEX)
URL_PATTERN = re.compile(URL_REGEX)
