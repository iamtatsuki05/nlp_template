import requests

STOPWORDS_URL = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-ja/master/stopwords-ja.txt'


def load_stopwords(url: str, timeout: float = 30.0) -> list[str]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return [line for line in resp.text.splitlines() if line]
