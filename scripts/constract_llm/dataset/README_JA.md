# データセット処理スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデル構築のためのデータセット処理に関連するスクリプトが含まれています。

## ディレクトリ構成

- `cleanse/`: データセットのクレンジングを行うスクリプト
- `preprocess/`: データセットの前処理を行うスクリプト
- `split/`: データセットを訓練・検証・テストセットに分割するスクリプト
- `hard_negative_mine/` – 検索系学習データ向けハードネガティブマイニングスクリプト

## データ処理フロー

1. **クレンジング（`cleanse/`）** – 重複・スケジュール情報・URL等を除去し、クリーンなレコードを残します。
2. **前処理（`preprocess/`）** – 選択したテキストフィールドに軽量な正規化を適用します。
3. **分割（`split/`）** – 再現性のあるシードで訓練・検証・テストに分割します。
4. **ハードネガティブマイニング（`hard_negative_mine/`、任意）** – コントラスト学習や検索モデル向けにネガティブ例を生成します。

コマンド例：

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json
python scripts/constract_llm/dataset/hard_negative_mine/hard_negative_mine.py config/custom/hard_negative.json
```

詳細な設定やオプションは各サブディレクトリの README を参照してください。
