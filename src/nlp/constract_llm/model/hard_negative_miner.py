from nlp.constract_llm.model.embedder.model.base import BaseEmbedder
from nlp.constract_llm.model.tokenizer.base import BaseTokenizer


class HardNegativeMiner:
    """Perform hard negative mining."""

    def __init__(self, embedder: BaseEmbedder, tokenizer: BaseTokenizer, num_negatives: int = 5) -> None:
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.num_negatives = num_negatives

    def mine(self, queries: list[str], positives: list[int], corpus: list[str]) -> dict[int, list[int]]:
        """Return hard negative indices per query."""
        hard_negatives: dict[int, list[int]] = {}
        for i, query in enumerate(queries):
            # トークン化 (ID/文字列は embedder の要求に従う)
            tokenized_query = self.tokenizer.tokenize(
                query,
                return_ids=self.embedder.requires_token_ids,
            )

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
