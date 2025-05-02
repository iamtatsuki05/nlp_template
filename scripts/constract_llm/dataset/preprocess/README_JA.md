# データセット前処理スクリプト

[English](README.md) / 日本語

このディレクトリには、データセットの前処理を行うスクリプトが含まれています。

## 概要

`preprocess.py`スクリプトは、データセットに対して基本的な前処理を適用し、後続の処理ステップのために準備します。

## 機能

- 指定されたテキストフィールドに対して前処理を適用
- 入力はローカルファイルまたはHugging Faceデータセット名
- 処理済みデータを指定されたディレクトリに保存

## 使用方法

```bash
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json
```

コマンドライン引数を使用して設定ファイルの値を上書きすることも可能です：

```bash
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json --text_fields='["content"]'
```

## 設定ファイル

設定ファイルは`config/constract_llm/dataset/preprocess/config.json`にあります。

### 設定例

```json
{
    "input_name_or_path": "imdb",
    "output_dir": "./data/misc/json",
    "text_fields": ["text", "label"]
}
```

### パラメータ説明

| パラメータ | 説明 |
|------------|------|
| `input_name_or_path` | 入力データセットのパス。ローカルファイルまたはHugging Faceデータセット名 |
| `output_dir` | 処理済みデータを保存するディレクトリ |
| `text_fields` | 処理するテキストフィールドのリスト。指定されない場合は全てのフィールドが処理される |
