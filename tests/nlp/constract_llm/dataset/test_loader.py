import json
from pathlib import Path

import pytest

from nlp.constract_llm.dataset import loader as loader_module
from nlp.constract_llm.dataset.loader import (
    iter_dataset_records,
    load_dataset_resource,
)


def test_load_dataset_resource_prefers_local_file(tmp_path: Path) -> None:
    data_path = tmp_path / 'data.json'
    data_path.write_text(json.dumps([{'text': 'hello'}]), encoding='utf-8')

    result = load_dataset_resource(data_path, local_split_name='')
    split_name, records = result.pick_split()

    assert result.is_local is True
    assert split_name == ''
    assert records == [{'text': 'hello'}]


def test_remote_dataset_loader_when_path_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy: dict[str, list[dict[str, str]]] = {'train': [{'text': 'a'}], 'validation': [{'text': 'b'}]}

    def fake_load_dataset(name: str, config: str | None = None) -> dict[str, list[dict[str, str]]]:  # noqa: ARG001
        return dummy

    monkeypatch.setattr(loader_module, 'load_dataset', fake_load_dataset)

    result = load_dataset_resource('dummy-dataset')

    assert result.is_local is False
    assert set(result.split_names) == {'train', 'validation'}
    _, records = result.pick_split('train')
    assert records[0]['text'] == 'a'


def test_loader_falls_back_to_remote_on_local_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken_path = tmp_path / 'broken.json'
    broken_path.write_text('{}', encoding='utf-8')

    dummy: dict[str, list[dict[str, str]]] = {'train': [{'text': 'fallback'}]}

    def fake_load_dataset(name: str, config: str | None = None) -> dict[str, list[dict[str, str]]]:  # noqa: ARG001
        return dummy

    monkeypatch.setattr(loader_module, 'load_dataset', fake_load_dataset)

    result = load_dataset_resource(broken_path, allow_remote_fallback=True)

    assert result.is_local is False
    _, records = result.pick_split('train')
    assert records == [{'text': 'fallback'}]


def test_iter_dataset_records_local(tmp_path: Path) -> None:
    data_path = tmp_path / 'records.json'
    data_path.write_text(json.dumps([{'text': 'x'}]), encoding='utf-8')

    records = list(iter_dataset_records(data_path))

    assert records == [{'text': 'x'}]


def test_iter_dataset_records_remote_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{'text': 'a'}, {'text': 'b'}]

    def fake_load_dataset(
        _name: str,
        _config: str | None = None,
        *,
        split: str,
        streaming: bool,
    ) -> list[dict[str, str]]:
        assert streaming is True
        assert split == 'train'
        return payload

    monkeypatch.setattr(loader_module, 'load_dataset', fake_load_dataset)

    records = list(iter_dataset_records('dummy', split='train', streaming=True))

    assert records == payload
