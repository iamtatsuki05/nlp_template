# docker+poetry
## docker操作方法
### ディレクトリについて
1. `git clone`でディレクトをダウンロード
2. ターミナル上でcdコマンドを使用し、ディレクトリ内へ移動
### dockerのコンテナ生成
1. `docker-compose up -d --build`コンテナ生成
### dockerに接続、解除
1. `docker exec -it <コンテナの名前> bash`で接続
2. `exit`でdockerとの接続解除
### jupyter labの起動
1. コンテナ起動状態状態でhttp://localhost:8888/lab にアクセス
### コンテナの停止と起動
1. `docker stop <コンテナの名前>`でコンテナを停止
2. `docker start <コンテナの名前>`でコンテナを起動
### poetryの操作方法
1. dockerに接続後`poetry add ライブラリー名`でライブラリの追加
