# スクリプト一覧

[English](README.md) / 日本語

このディレクトリには、NLPテンプレートプロジェクトで使用される様々なスクリプトが含まれています。これらのスクリプトは主に言語モデル（LLM）の構築と訓練に関連するタスクを自動化するために設計されています。

## 基本スクリプト

- `main.py` - 基本的な "Hello, World!" を表示するサンプルスクリプト。

## constract_llm

言語モデル構築に関連する一連のスクリプトが含まれています。

### dataset

データセット処理に関連するスクリプト群です。

#### cleanse

- `cleanse.py` - データセットのクレンジングを行うスクリプト。
  - 重複データの削除
  - MinHashを使用した類似テキストの検出と削除
  - 時間スケジュールを含むテキストの削除
  - 数字のみのテキストの削除
  - URLやメールアドレスを含むテキストの削除
  - 設定ファイルを使用して様々なクレンジングオプションをカスタマイズ可能

#### preprocess

- `preprocess.py` - データセットの前処理を行うスクリプト。
  - 指定されたテキストフィールドに対して前処理を適用
  - 入力はローカルファイルまたはHugging Faceデータセット名
  - 処理済みデータを指定されたディレクトリに保存

#### split

- `split.py` - データセットを訓練・検証・テストセットに分割するスクリプト。
  - ランダム分割または連続分割をサポート
  - テストサイズと検証サイズを比率または絶対数で指定可能
  - 層化サンプリングのサポート（指定されたキーに基づく）
  - 再現性のためのランダムシード設定

### model

- `init_model.py` - モデルの初期化を行うスクリプト。
  - 様々なモデルタイプをサポート
  - Hugging Face Hubからのモデルロードまたはローカルパスからのロード
  - モデルをローカルに保存またはHugging Face Hubにプッシュ可能
  - 再現性のためのシード設定

### tokenizer

トークナイザー関連のスクリプト群です。

- `add_tokens.py` - 既存のトークナイザーに新しいトークンを追加するスクリプト。
  - 通常トークンと特殊トークンの追加をサポート
  - JSONファイルからトークン設定を読み込み
  - 拡張されたトークナイザーをローカルに保存またはHugging Face Hubにプッシュ可能

- `merge_spm.py` - 複数のSentencePieceモデルをマージするスクリプト。
  - ベーストークナイザーと追加トークナイザーを指定
  - マージされたモデルをローカルに保存またはHugging Face Hubにプッシュ可能

- `train_tokenizer.py` - 新しいトークナイザーを訓練するスクリプト。
  - Unigram、BPE、WordPieceモデルタイプをサポート
  - 特殊トークンの設定
  - 大規模コーパスでの訓練オプション
  - バイトフォールバック、数字分割などの高度なオプション
  - 訓練済みトークナイザーをローカルに保存またはHugging Face Hubにプッシュ可能

## 使用方法

各スクリプトは[Google Fire](https://github.com/google/python-fire)を使用してCLIインターフェースを提供しています。基本的な使用方法は以下の通りです：

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
```

各スクリプトは対応する設定ファイルを受け取り、その設定に基づいて処理を実行します。設定ファイルはJSON、YAML、TOMLなどの形式で提供できます。

また、コマンドライン引数を使用して設定ファイルの値を上書きすることも可能です：

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json --do_deduplicate=False
```

## 設定ファイル

各スクリプトに対応する設定ファイルは `config/` ディレクトリに格納されています。例えば：

- データセットクレンジング: `config/constract_llm/dataset/cleanse/config.json`
- データセット前処理: `config/constract_llm/dataset/preprocess/config.json`
- データセット分割: `config/constract_llm/dataset/split/config.json`
- モデル初期化: `config/constract_llm/model/init_model/config.json`
- トークナイザー訓練: `config/constract_llm/tokenizer/train_tokenizer/config.json`

各設定ファイルには、対応するスクリプトで使用される全てのパラメータが含まれています。
