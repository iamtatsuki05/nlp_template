import pytest
from transformers import AutoTokenizer

from nlp.constract_llm.tokenizer.add_tokens import add_tokens_to_tokenizer


@pytest.mark.parametrize(
    'normal_tokens,special_tokens,expected_normal_tokens,expected_special_tokens',
    [
        (['token1', 'token2'], ['<special1>', '<special2>'], ['token1', 'token2'], ['<special1>', '<special2>']),
        (None, None, [], []),
        (['token3'], None, ['token3'], []),
        (None, ['<special3>'], [], ['<special3>']),
    ],
)
def test_add_tokens_to_tokenizer(normal_tokens, special_tokens, expected_normal_tokens, expected_special_tokens):
    # Create a mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    initial_vocab_size = len(tokenizer)

    # Call the function
    updated_tokenizer = add_tokens_to_tokenizer(tokenizer, normal_tokens, special_tokens)

    # Check the normal tokens
    if normal_tokens:
        assert len(updated_tokenizer) == initial_vocab_size + len(normal_tokens)
        for token in normal_tokens:
            assert token in updated_tokenizer.get_vocab()
    else:
        assert len(updated_tokenizer) == initial_vocab_size

    # Check the special tokens
    if special_tokens:
        assert updated_tokenizer.special_tokens_map['additional_special_tokens'] == special_tokens
    else:
        assert updated_tokenizer.special_tokens_map['additional_special_tokens'] == []
