# モデル初期化スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデルの初期化を行うスクリプトが含まれています。

## 概要

`init_model.py`スクリプトは、事前学習済みモデルを初期化し、後続の微調整やその他の処理のために準備します。

## 機能

- 様々なモデルタイプをサポート
- Hugging Face Hubからのモデルロードまたはローカルパスからのロード
- モデルをローカルに保存またはHugging Face Hubにプッシュ可能
- 再現性のためのシード設定

## 使用方法

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json
```

コマンドライン引数を使用して設定ファイルの値を上書きすることも可能です：

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json --model_type=encoder --push_to_hub=False
```

## 設定ファイル

設定ファイルは`config/constract_llm/model/init_model/config.json`にあります。

### 設定例

```json
{
  "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "output_dir": "./data/misc/json",
  "model_type": "causal",
  "push_to_hub": true,
  "private": true,
  "seed": 42
}
```

### パラメータ説明

| パラメータ | 説明 |
|------------|------|
| `model_name_or_path` | モデルのパスまたはHugging Face Hubからのモデル名 |
| `model_type` | モデルのタイプ（`seq2seq`, `causal`, `masked`, `generic` ）|
| `output_dir` | モデルを保存するディレクトリ |
| `push_to_hub` | モデルをHugging Face Hubにプッシュするかどうか |
| `private` | Hugging Face Hub上でモデルをプライベートにするかどうか |
| `seed` | 初期化のためのランダムシード |
