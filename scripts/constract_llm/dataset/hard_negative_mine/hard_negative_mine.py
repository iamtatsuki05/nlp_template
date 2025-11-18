# ruff: noqa: E501
# ref: https://huggingface.co/blog/Albertmade/nvretriever-into-financial-text
# NOTE: This package is not yet completely ready for production use.
"""Hard negative mining for retrieval model training.

Usage Examples

1. Mining hard negatives (default):

Prepare a local JSON file (list of dicts) that already contains the fields below:
```json
[
  {
    "doc_id": 0,
    "doc_text": "企業のサービス概要...",
    "query": "サービスの特徴は?"
  },
  {
    "doc_id": 1,
    "doc_text": "導入事例のまとめ...",
    "query": "導入先の業界は?"
  }
]
```

Create a `config.json` that references the local file:
```json
{
  "dataset_name_or_path": "example_retrieval.json",
  "output_dir": "./outputs",
  "split": "train",
  "text_field": "doc_text",
  "query_field": "query",
  "positive_field": "doc_id",
  "num_samples": 100,
  "num_negatives": 5,
  "tokenizer": {
    "type": "sudachi",
    "sudachi_mode": "C",
    "sudachi_dict": "core",
    "stopwords": ["の","に","は","を","た","が","で","て","と","し","れ","さ"],
    "pos_filter": ["名詞","動詞","形容詞"]
  },
  "embedder": {
    "type": "bm25s",
    "embedder_path": null
  }
}
```
Run mining:
```bash
python hard_negative_mine.py mine --config_path config.json
```

2. Direct CLI args (mining):
```bash
python hard_negative_mine.py \
    mine \
  --dataset_name_or_path example_retrieval.json \
  --output_dir ./outputs \
  --split train \
  --text_field doc_text \
  --query_field query \
  --positive_field doc_id \
  --num_samples 100 \
  --num_negatives 5 \
  --tokenizer "{'type':'sudachi','sudachi_mode':'C','sudachi_dict':'core','stopwords':['の','に','は','を'],'pos_filter':['名詞','動詞']}" \
  --embedder "{'type':'bm25s','embedder_path': None}"
```

3. Training only:
```bash
python hard_negative_mine.py train --config_path config.json
```
"""

import logging
from pathlib import Path
from typing import Any, Literal, TypeGuard

import fire
from datasets import load_dataset
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from nlp.common.utils.cli_utils import load_cli_config
from nlp.common.utils.file.json import load_json, save_as_indented_json
from nlp.constract_llm.model.embedder.model.base import BaseEmbedder
from nlp.constract_llm.model.embedder.model.bm25 import GensimBM25Model
from nlp.constract_llm.model.embedder.model.bm25_s import BM25SModel
from nlp.constract_llm.model.embedder.model.tfidf import GensimTfidfModel
from nlp.constract_llm.model.hard_negative_miner import HardNegativeMiner
from nlp.constract_llm.model.tokenizer.base import BaseTokenizer
from nlp.constract_llm.model.tokenizer.mecab import MeCabTokenizer
from nlp.constract_llm.model.tokenizer.stopword import STOPWORDS
from nlp.constract_llm.model.tokenizer.sudachi import SudachiTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizerConfig(BaseModel):
    type: Literal['sudachi', 'mecab'] = Field(..., description="'sudachi' or 'mecab'")
    sudachi_mode: Literal['A', 'B', 'C'] = Field('C', description='Sudachi split mode')
    sudachi_dict: Literal['small', 'core', 'full'] = Field('core', description='Sudachi dictionary')
    stopwords: list[str] = Field(default_factory=lambda: STOPWORDS, description='Stopwords list')
    pos_filter: list[str] = Field(
        default_factory=lambda: ['名詞', '動詞', '形容詞'],
        description='POS filter list',
    )


class EmbedderConfig(BaseModel):
    type: Literal['bm25s', 'tfidf', 'gensim_bm25'] = Field(..., description='Embedder type')
    embedder_path: Path | None = Field(
        None,
        description='Path to pretrained model (None for new model)',
    )


class CLIConfig(BaseModel):
    dataset_name_or_path: Path | str = Field(
        ...,
        description='Local JSON or HF dataset name',
    )
    dataset_config_name: str | None = Field(None, description='Optional dataset config name')
    output_dir: Path | str = Field(
        ...,
        description='Output directory',
    )
    split: Literal['train', 'validation', 'test'] = Field('train', description='Dataset split')
    text_field: str | None = Field(
        None,
        description='Document field (None for train-only)',
    )
    query_field: str = Field('query', description='Query field name')
    positive_field: str = Field('doc_id', description='Positive doc field')
    num_samples: int = Field(100, description='Max examples per split')
    num_negatives: int = Field(5, description='Hard negatives per query')
    tokenizer: TokenizerConfig = Field(..., description='Tokenizer settings')
    embedder: EmbedderConfig = Field(..., description='Embedder settings')


JsonRecord = dict[str, Any]


def _is_record_list(data: object) -> TypeGuard[list[JsonRecord]]:
    return isinstance(data, list) and all(isinstance(item, dict) for item in data)


def _ensure_records(data: object, context: str) -> list[JsonRecord]:
    if not _is_record_list(data):
        raise TypeError(f'{context} must be a list of dicts, got {type(data).__name__}')
    return [dict(item) for item in data]


def _extract_field[T](records: list[JsonRecord], field: str, expected_type: type[T]) -> list[T]:
    values: list[T] = []
    for record in records:
        value = record.get(field)
        if not isinstance(value, expected_type):
            raise TypeError(f'Field "{field}" must be {expected_type.__name__}, got {type(value).__name__}')
        values.append(value)
    return values


def initialize_tokenizer(cfg: TokenizerConfig) -> BaseTokenizer:
    if cfg.type == 'sudachi':
        return SudachiTokenizer(
            sudachi_mode=cfg.sudachi_mode,
            sudachi_dict=cfg.sudachi_dict,
            stopwords=cfg.stopwords,
            pos_filter=cfg.pos_filter,
        )
    return MeCabTokenizer(stopwords=cfg.stopwords, pos_filter=cfg.pos_filter)


def initialize_embedder(cfg: EmbedderConfig) -> tuple[BaseEmbedder, bool]:
    if cfg.embedder_path:
        path = str(cfg.embedder_path)
        logger.info(f"Loading pretrained '{cfg.type}' from {path}")
        if cfg.type == 'bm25s':
            return BM25SModel.load(path, load_corpus=True), False
        if cfg.type == 'gensim_bm25':
            return GensimBM25Model.load(path), False
        return GensimTfidfModel.load(path), False
    # new model
    if cfg.type == 'bm25s':
        return BM25SModel(corpus=None), True
    if cfg.type == 'gensim_bm25':
        return GensimBM25Model(), True
    return GensimTfidfModel(), True


def process_split(
    name: str,
    examples: list[dict[str, Any]],
    cfg: CLIConfig,
    tokenizer: BaseTokenizer,
    embedder_factory: tuple[BaseEmbedder, bool],
) -> None:
    if cfg.text_field is None:
        msg = 'text_field must be specified to process splits.'
        raise ValueError(msg)
    outdir = Path(cfg.output_dir)
    samples = examples[: cfg.num_samples]
    corpus = _extract_field(samples, cfg.text_field, str)
    queries = _extract_field(samples, cfg.query_field, str)
    positives = _extract_field(samples, cfg.positive_field, int)

    model, need_fit = embedder_factory
    tokens = tokenizer.tokenize(corpus, return_ids=model.requires_token_ids)
    if need_fit:
        model.fit(tokens)
        mpath = outdir / f'{cfg.embedder.type}_model'
        logger.info(f'Saving model to {mpath}')
        model.save(str(mpath))

    miner = HardNegativeMiner(model, tokenizer, num_negatives=cfg.num_negatives)
    negatives = miner.mine(queries, positives, corpus)

    rpath = outdir / f'hard_negatives_{name}.json'
    save_as_indented_json(negatives, rpath)
    logger.info(f"Saved hard negatives for split '{name}' to {rpath}")


def train(config_path: Path | str | None = None, **kwargs: object) -> None:
    """Train or load+save embedder only."""
    cfg = CLIConfig(**load_cli_config(config_path, **kwargs))

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = initialize_tokenizer(cfg.tokenizer)
    embedder, need_fit = initialize_embedder(cfg.embedder)

    path = Path(cfg.dataset_name_or_path)
    if cfg.text_field is None:
        msg = 'text_field must be specified when training the embedder.'
        raise ValueError(msg)

    if path.exists():
        raw_data = load_json(path)
        data = _ensure_records(raw_data, 'Local dataset')
    else:
        data = [
            dict(ex)
            for ex in tqdm(
                (
                    load_dataset(str(cfg.dataset_name_or_path), cfg.dataset_config_name)
                    if cfg.dataset_config_name
                    else load_dataset(str(cfg.dataset_name_or_path))
                )[cfg.split],
                desc='Loading data',
            )
        ]

    corpus = _extract_field(data, cfg.text_field, str)
    tokens = tokenizer.tokenize(corpus, return_ids=embedder.requires_token_ids)
    if need_fit:
        embedder.fit(tokens)
        mpath = outdir / f'{cfg.embedder.type}_model'
        logger.info(f'Training complete. Saving embedder to {mpath}')
        embedder.save(str(mpath))


def mine(config_path: Path | str | None = None, **kwargs: object) -> None:
    """Perform hard negative mining for each split."""
    cfg = CLIConfig(**load_cli_config(config_path, **kwargs))

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = initialize_tokenizer(cfg.tokenizer)

    path = Path(cfg.dataset_name_or_path)
    if path.exists():
        raw_data = load_json(path)
        data = _ensure_records(raw_data, 'Local dataset')
        proc = initialize_embedder(cfg.embedder)
        process_split('custom', data, cfg, tokenizer, proc)
    else:
        ds = (
            load_dataset(str(cfg.dataset_name_or_path), cfg.dataset_config_name)
            if cfg.dataset_config_name
            else load_dataset(str(cfg.dataset_name_or_path))
        )
        for split, subset in ds.items():
            examples = [dict(ex) for ex in tqdm(subset.select(range(cfg.num_samples)), desc=f'Loading {split}')]
            proc = initialize_embedder(cfg.embedder)
            process_split(split, examples, cfg, tokenizer, proc)

    logger.info('Completed all splits.')


if __name__ == '__main__':
    fire.Fire({'train': train, 'mine': mine})
