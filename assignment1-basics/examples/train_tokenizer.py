"""Train the course's word/character BPE implementation on a UTF-8 corpus."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cs336_basics.bpe import BPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("tiny_corpus.txt"))
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "outputs/tokenizer/bpe")
    parser.add_argument("--merges", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=100_000)
    args = parser.parse_args()
    if args.merges < 1 or args.max_words < 1:
        parser.error("--merges and --max-words must be positive")
    words = []
    with args.dataset.open(encoding="utf-8") as stream:
        for line in stream:
            words.extend(line.split()[:args.max_words - len(words)])
            if len(words) >= args.max_words:
                break
    if not words:
        parser.error("dataset has no words")
    tokenizer = BPE()
    tokenizer.train(words, args.merges)
    tokenizer.save(str(args.output_prefix))
    print(f"Saved {args.output_prefix}_vocab.pkl and {args.output_prefix}_merges.pkl")


if __name__ == "__main__":
    main()
