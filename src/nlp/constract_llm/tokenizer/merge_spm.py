import logging
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_spm_models(
    base_tokenizer_name_or_path: str | Path,
    additional_tokenizer_name_or_path: str | Path,
    output_dir: str | Path,
    push_to_hub: bool = False,
    private: bool = True,
) -> None:
    base_tokenizer = AutoTokenizer.from_pretrained(str(base_tokenizer_name_or_path), use_fast=False)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())

    additional_tokenizer = AutoTokenizer.from_pretrained(str(additional_tokenizer_name_or_path), use_fast=False)
    add_proto = sentencepiece_model_pb2.ModelProto()
    add_proto.ParseFromString(additional_tokenizer.sp_model.serialized_model_proto())

    existing = {p.piece for p in proto.pieces}
    for p in tqdm(add_proto.pieces, desc='Merging pieces', unit='piece'):
        if p.piece not in existing:
            new_p = proto.pieces.add()
            new_p.piece = p.piece
            new_p.score = p.score
    logger.info(f'Merged pieces: before={len(existing)}, after={len(proto.pieces)}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_spm_path = output_dir / 'merged.model'
    merged_spm_path.write_bytes(proto.SerializeToString())
    logger.info(f'Saved merged SPM model to {merged_spm_path}')

    merged_tokenizer = LlamaTokenizer(vocab_file=str(merged_spm_path))
    merged_tokenizer.save_pretrained(str(output_dir))
    logger.info(f'Saved merged HF tokenizer to {output_dir}')

    if push_to_hub:
        merged_tokenizer.push_to_hub(output_dir.name, private=private)
        logger.info(f"Pushed merged tokenizer to hub '{output_dir.name}' (private={private})")
