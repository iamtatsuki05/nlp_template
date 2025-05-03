# 言語モデル構築スクリプト

[English](README.md) / 日本語

このディレクトリには、言語モデル（LLM）の構築と訓練に関連する一連のスクリプトが含まれています。

## ディレクトリ構成

- `dataset/`: データセット処理関連のスクリプト
  - `cleanse/`: データセットのクレンジング（浄化）
  - `preprocess/`: データセットの前処理
  - `split/`: データセットの分割
- `model/`: モデル初期化関連のスクリプト
- `tokenizer/`: トークナイザー関連のスクリプト
  - トークン追加
  - SPMモデルのマージ
  - トークナイザーの訓練

## 言語モデル構築フロー

一般的な言語モデル構築フローは以下の通りです：

1. **データセット処理**:
   - クレンジング: 不要なデータの除去
   - 前処理: 基本的な前処理の適用
   - 分割: 訓練・検証・テストセットへの分割

2. **トークナイザー準備**:
   - 新規トークナイザーの訓練
   - または既存トークナイザーへのトークン追加
   - 必要に応じて複数のSPMモデルのマージ

3. **モデル初期化**:
   - 事前学習済みモデルのロード
   - または新規モデルの初期化

4. **モデル訓練**:
   - 処理済みデータセットを使用したモデルの訓練

## 使用例

以下は、言語モデル構築の一連のコマンド例です：

```bash
# 1. データセットのクレンジング
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json

# 2. クレンジング済みデータの前処理
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json

# 3. 処理済みデータの分割
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json

# 4. トークナイザーの訓練
python scripts/constract_llm/tokenizer/train_tokenizer.py config/constract_llm/tokenizer/train_tokenizer/config.json

# 5. トークナイザーへのトークン追加
python scripts/constract_llm/tokenizer/add_tokens.py config/constract_llm/tokenizer/add_tokens/config/config.json

# 6. モデルの初期化
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json
```

各スクリプトの詳細な使用方法と設定オプションについては、それぞれのディレクトリ内のREADMEを参照してください。
