"""Restore the CT2 vocabulary from the repository's local tokenizer, without downloads."""
import json
from pathlib import Path
from tokenizers import Tokenizer
from src.utils.utils import MODEL_NAME


def main():
    directory = Path(MODEL_NAME)
    vocab = Tokenizer.from_file(str(directory / "tokenizer.json")).get_vocab()
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    if [index for _, index in ordered] != list(range(len(ordered))):
        raise ValueError("Tokenizer IDs must be contiguous")
    target = directory / "vocabulary.json"
    target.write_text(json.dumps([token for token, _ in ordered], ensure_ascii=False, indent=2))
    print(f"Saved {len(ordered)} tokens to {target.name}")


if __name__ == "__main__":
    main()
