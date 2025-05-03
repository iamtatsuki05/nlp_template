# データセット処理スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデル構築のためのデータセット処理に関連するスクリプトが含まれています。

## ディレクトリ構成

- `cleanse/`: データセットのクレンジングを行うスクリプト
- `preprocess/`: データセットの前処理を行うスクリプト
- `split/`: データセットを訓練・検証・テストセットに分割するスクリプト

## データ処理フロー

一般的なデータ処理フローは以下の通りです：

1. **クレンジング（cleanse）**: 生データから不要なデータを除去し、高品質なデータセットを作成します
2. **前処理（preprocess）**: クレンジング済みデータに対して基本的な前処理を適用します
3. **分割（split）**: 処理済みデータを訓練・検証・テストセットに分割します

## 使用例

以下は、IMDBデータセットを処理する一連のコマンド例です：

```bash
# 1. データセットのクレンジング
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json

# 2. クレンジング済みデータの前処理
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json

# 3. 処理済みデータの分割
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json
```

各スクリプトの詳細な使用方法と設定オプションについては、それぞれのディレクトリ内のREADMEを参照してください。
