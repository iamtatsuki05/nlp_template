# データセット分割スクリプト

[English](README.md) / 日本語

このディレクトリには、データセットを訓練・検証・テストセットに分割するスクリプトが含まれています。

## 概要

`split.py`スクリプトは、データセットを訓練（train）、検証（validation）、テスト（test）の各セットに分割します。これは機械学習モデルの訓練と評価のために重要なステップです。

## 機能

- ランダム分割または連続分割をサポート
- テストサイズと検証サイズを比率または絶対数で指定可能
- 層化サンプリングのサポート（指定されたキーに基づく）
- 再現性のためのランダムシード設定

## 使用方法

```bash
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json
```

コマンドライン引数を使用して設定ファイルの値を上書きすることも可能です：

```bash
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json --test_size=0.2 --val_size=0.1
```

## 設定ファイル

設定ファイルは`config/constract_llm/dataset/split/config.json`にあります。

### 設定例

```json
{
    "dataset_name_or_path": "imdb",
    "output_dir": "./data/misc/json",
    "test_size": 0.1,
    "val_size": 0.1,
    "split_mode": "random",
    "random_seed": 42,
    "stratify_key": "label"
}
```

### パラメータ説明

| パラメータ             | 説明                                                                             |
| ---------------------- | -------------------------------------------------------------------------------- |
| `dataset_name_or_path` | 入力データセットのパス。ローカルファイルまたはHugging Faceデータセット名         |
| `output_dir`           | 分割されたデータを保存するディレクトリ                                           |
| `test_size`            | テスト分割に含めるデータセットの割合（浮動小数点数）または絶対サンプル数（整数） |
| `val_size`             | 検証分割に含めるデータセットの割合（浮動小数点数）または絶対サンプル数（整数）   |
| `split_mode`           | データセットの分割モード。「random」または「sequential」                         |
| `random_seed`          | 再現性のためのランダムシード                                                     |
| `stratify_key`         | 層化分割のためのキー。Noneの場合、層化は適用されない                             |
