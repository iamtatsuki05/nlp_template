# トークナイザー関連スクリプト

[English](README.md) / 日本語

このディレクトリには、トークナイザーの訓練、拡張、マージに関連するスクリプトが含まれています。

## `add_tokens.py`

既存トークナイザーに通常トークン・特殊トークンを追加します。

**主な機能**
- 通常トークンと特殊トークンの一覧を JSON から読み込み。
- 拡張後のトークナイザーをローカル保存し、必要に応じて Hugging Face Hub にプッシュ。

**使用方法**

```bash
python scripts/constract_llm/tokenizer/add_tokens.py config/constract_llm/tokenizer/add_tokens/config/config.json
```

トークンリストのテンプレートは `config/constract_llm/tokenizer/add_tokens/config/` にあります。

## `merge_spm.py`

ベースの SentencePiece モデルに追加モデルをマージし、SentencePiece/Hugging Face 両形式でエクスポートします。

**主な機能**
- ベース／追加トークナイザーをローカルパスまたは Hub ID で指定可能。
- マージ結果をローカル保存し、必要に応じて Hub へ公開。

**使用方法**

```bash
python scripts/constract_llm/tokenizer/merge_spm.py config/constract_llm/tokenizer/merge_spm/config.json
```

## `train_tokenizer.py`

Hugging Face データセットまたはローカルコーパスから SentencePiece 系トークナイザーを学習します。

**主な機能**
- SentencePiece Trainer を用いた `unigram` / `bpe` / `wordpiece` モデルタイプに対応。
- ストリーミングや `train_extremely_large_corpus` により巨大コーパスにも対応。
- バイトフォールバック、数字分割、空白処理など高度なスイッチを提供。
- JSON からの特殊トークン読み込みと CLI 設定内のデフォルトリストをサポート。

**使用方法**

```bash
python scripts/constract_llm/tokenizer/train_tokenizer.py config/constract_llm/tokenizer/train_tokenizer/config.json
```

**主なパラメータ**

| パラメータ                                                                                     | 説明                                                            |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `dataset_name_or_path`                                                                         | HF データセット識別子またはローカルパス。                       |
| `dataset_config`                                                                               | オプションのデータセット設定名。                                |
| `split`                                                                                        | 読み込むデータセット分割（デフォルトは `train`）。              |
| `text_column`                                                                                  | 入力テキストを含むフィールド名。                                |
| `vocab_size`                                                                                   | 目標語彙サイズ。                                                |
| `model_type`                                                                                   | SentencePiece モデルタイプ（`unigram` / `bpe` / `word` / `char`）。 |
| `special_tokens_config`                                                                        | 追加する特殊トークンを列挙した JSON ファイル。                  |
| `default_special_tokens`                                                                       | JSON が無い場合に利用するデフォルトの特殊トークン。             |
| `max_train_samples`                                                                            | 読み込む最大サンプル数（省略時は全件）。                        |
| `train_extremely_large_corpus`                                                                 | SentencePiece の大規模コーパスモードを有効化。                  |
| `character_coverage`                                                                           | モデルでカバーする文字の割合（0〜1）。                          |
| `num_threads`                                                                                  | SentencePiece Trainer が使用するスレッド数。                    |
| `byte_fallback` / `split_digits` / `allow_whitespace_only_pieces` / `remove_extra_whitespaces` | 前処理のタイプ。                                                |
| `input_sentence_size`                                                                          | 学習に使用する文の最大数。                                      |
| `push_to_hub`, `private`, `output_dir`                                                         | 出力と Hub 公開まわりの設定。                                   |

設定テンプレートは `config/constract_llm/tokenizer/` 配下にまとめられています。
