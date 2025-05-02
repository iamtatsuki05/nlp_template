# トークナイザー関連スクリプト

[English](README.md) / 日本語

このディレクトリには、トークナイザーの訓練、拡張、マージに関連するスクリプトが含まれています。

## スクリプト一覧

### add_tokens.py

既存のトークナイザーに新しいトークンを追加するスクリプトです。

#### 機能
- 通常トークンと特殊トークンの追加をサポート
- JSONファイルからトークン設定を読み込み
- 拡張されたトークナイザーをローカルに保存またはHugging Face Hubにプッシュ可能

#### 使用方法
```bash
python scripts/constract_llm/tokenizer/add_tokens.py config/constract_llm/tokenizer/add_tokens/config/config.json
```

#### 設定例
```json
{
    "tokenizer_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "normal_tokens_config_path": "./config/constract_llm/tokenizer/add_tokens/config/normal_tokens.json",
    "special_tokens_config_path": "./config/constract_llm/tokenizer/add_tokens/config/special_tokens.json",
    "push_to_hub": true,
    "private": true,
    "output_dir": "./data/misc/json"
}
```

#### パラメータ説明
| パラメータ | 説明 |
|------------|------|
| `tokenizer_name_or_path` | トークナイザーモデルのパスまたはHugging Faceモデル名 |
| `normal_tokens_config_path` | 通常トークンを含むJSONファイルのパス |
| `special_tokens_config_path` | 特殊トークンを含むJSONファイルのパス |
| `push_to_hub` | トークナイザーをHugging Face Hubにプッシュするかどうか |
| `private` | Hugging Face Hub上でモデルをプライベートにするかどうか |
| `output_dir` | 拡張されたトークナイザーを保存するディレクトリ |

### merge_spm.py

複数のSentencePieceモデルをマージするスクリプトです。

#### 機能
- ベーストークナイザーと追加トークナイザーを指定
- マージされたモデルをローカルに保存またはHugging Face Hubにプッシュ可能

#### 使用方法
```bash
python scripts/constract_llm/tokenizer/merge_spm.py config/constract_llm/tokenizer/merge_spm/config.json
```

#### 設定例
```json
{
    "base_tokenizer_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b",
    "additional_tokenizer_name_or_path": "./data/misc/json",
    "output_dir": "./data/misc/json",
    "push_to_hub": true,
    "private": true
}
```

#### パラメータ説明
| パラメータ | 説明 |
|------------|------|
| `base_tokenizer_name_or_path` | ベースHFトークナイザーのディレクトリまたは名前 |
| `additional_tokenizer_name_or_path` | 追加HFトークナイザーのディレクトリまたは名前 |
| `output_dir` | マージされたSPMモデルとHFトークナイザーを保存するディレクトリ |
| `push_to_hub` | Hugging Face Hubにプッシュするかどうか |
| `private` | デフォルトでHubリポジトリをプライベートにするかどうか |

### train_tokenizer.py

新しいトークナイザーを訓練するスクリプトです。

#### 機能
- Unigram、BPE、WordPieceモデルタイプをサポート
- 特殊トークンの設定
- 大規模コーパスでの訓練オプション
- バイトフォールバック、数字分割などの高度なオプション

#### 使用方法
```bash
python scripts/constract_llm/tokenizer/train_tokenizer.py config/constract_llm/tokenizer/train_tokenizer/config.json
```

#### 設定例
```json
{
    "dataset_name_or_path": "wikimedia/wikipedia",
    "dataset_config": "20231101.ja",
    "split": "train",
    "text_column": "text",
    "vocab_size": 30000,
    "min_frequency": 2,
    "model_type": "unigram",
    "push_to_hub": true,
    "private": true,
    "output_dir": "./data/misc/json",
    "max_train_samples": 10000
}
```

#### パラメータ説明
| パラメータ | 説明 |
|------------|------|
| `dataset_name_or_path` | HFデータセット識別子（例：'wikipedia'） |
| `dataset_config` | オプションのデータセット設定名 |
| `split` | 使用する分割（例：'train'） |
| `text_column` | テキストの列名 |
| `vocab_size` | 語彙サイズ |
| `model_type` | SentencePieceモデルタイプ（'unigram'、'bpe'、'wordpiece'） |
| `special_tokens_config` | 特殊トークンのJSONへのパス |
| `default_special_tokens` | デフォルトの特殊トークン |
| `max_train_samples` | 使用する例の最大数（デフォルトは全て） |
| `train_extremely_large_corpus` | 非常に大きなコーパスでのトレーニングを有効にする |
| `character_coverage` | モデルでカバーされる文字の量（0.0〜1.0） |
| `num_threads` | トレーニング用のスレッド数 |
| `byte_fallback` | バイトフォールバックを有効にする |
| `split_digits` | 数字を別々のトークンに分割する |
| `allow_whitespace_only_pieces` | 空白のみを含むピースを許可する |
| `remove_extra_whitespaces` | 入力の余分な空白を削除する |
| `input_sentence_size` | トレーニングに使用する文の最大数 |
| `push_to_hub` | HF Hubにプッシュするかどうか |
| `private` | Hubリポジトリをプライベートにするかどうか |
| `output_dir` | トークナイザーを保存するディレクトリ |
