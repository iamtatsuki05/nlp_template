from nlp.constract_llm.model.embedder import BaseEmbedder, BM25SModel
from nlp.constract_llm.model.tokenizer import BaseTokenizer


class HardNegativeMiner:
    """
    ハードネガティブマイニングを実行するクラス。
    各クエリに対し、正例以外の上位 k 件をネガティブサンプルとして抽出します。
    """

    def __init__(self, embedder: BaseEmbedder, tokenizer: BaseTokenizer, num_negatives: int = 5):
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives

    def mine(self, queries: list[str], positives: list[int], corpus: list[str]) -> dict[int, list[int]]:
        """
        queries: クエリ文字列のリスト
        positives: 各クエリの正例ドキュメントインデックスリスト
        corpus: ドキュメントコーパス

        戻り値: {クエリインデックス: [ハードネガティブドキュメントのインデックスリスト]} の辞書
        """
        hard_negatives: dict[int, list[int]] = {}
        for i, query in enumerate(queries):
            # トークン化 (ID/文字列は embedder に合わせて)
            if isinstance(self.embedder, BM25SModel):
                tokenized_query = self.tokenizer.tokenize(query, return_ids=True)
            else:
                tokenized_query = self.tokenizer.tokenize(query, return_ids=False)

            # コーパス全体を取得 (上位 len(corpus) 件)
            docs, _ = self.embedder.retrieve(tokenized_query, corpus, k=len(corpus))

            # 正例以外から上位 k 件選択
            negs: list[int] = []
            for doc in docs:
                idx = corpus.index(doc)
                if idx != positives[i]:
                    negs.append(idx)
                if len(negs) >= self.num_negatives:
                    break
            hard_negatives[i] = negs
        return hard_negatives
