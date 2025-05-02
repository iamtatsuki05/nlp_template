# データセットクレンジングスクリプト

[English](README.md) / 日本語

このディレクトリには、データセットのクレンジングを行うスクリプトが含まれています。

## 概要

`cleanse.py`スクリプトは、生データセットから不要なデータを除去し、高品質なデータセットを作成するための様々なクレンジング操作を提供します。

## 機能

- 重複データの削除
- MinHashを使用した類似テキストの検出と削除
- 時間スケジュールを含むテキストの削除
- 数字のみのテキストの削除
- URLやメールアドレスを含むテキストの削除

## 使用方法

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
```

コマンドライン引数を使用して設定ファイルの値を上書きすることも可能です：

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json --do_deduplicate=False
```

## 設定ファイル

設定ファイルは`config/constract_llm/dataset/cleanse/config.json`にあります。

### 設定例

```json
{
    "input_name_or_path": "imdb",
    "output_dir": "./data/misc/json",
    "text_fields": ["text"],
    "do_deduplicate": true,
    "do_rm_duplicated_by_minhash": true,
    "minhash_threshold": 0.95,
    "do_rm_time_schedule": true,
    "rm_time_schedule_threshold": 3,
    "do_rm_only_numeric": true,
    "do_rm_include_url_text": true,
    "do_rm_include_email_text": true
}
```

### パラメータ説明

| パラメータ | 説明 |
|------------|------|
| `input_name_or_path` | 入力JSONファイルのパスまたはHugging Faceデータセット名 |
| `output_dir` | クレンジング済みデータを保存するディレクトリ |
| `text_fields` | テキストクレンジングを適用するフィールド名のリスト |
| `do_deduplicate` | レコードレベルで重複を削除するかどうか |
| `do_rm_duplicated_by_minhash` | MinHashを使用して類似テキストを削除するかどうか |
| `minhash_threshold` | MinHashによる類似テキスト検出の閾値 (0.0〜1.0) |
| `minhash_num_perm` | MinHashのパーミュテーション数 |
| `num_workers` | 並列処理のワーカー数 |
| `do_rm_time_schedule` | 時間スケジュールを含むテキストを削除するかどうか |
| `rm_time_schedule_threshold` | テキストを削除する時間パターンの最小出現回数 |
| `do_rm_only_numeric` | 数字のみのテキストを削除するかどうか |
| `do_rm_include_url_text` | URLを含むテキストを削除するかどうか |
| `do_rm_include_email_text` | メールアドレスを含むテキストを削除するかどうか |
| `max_use_samples` | データセットから使用するサンプルの最大数 |
| `max_save_samples` | 出力ファイルに保存するサンプルの最大数 |
