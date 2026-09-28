"""Exercise the actual tokenizer and Transformer on a tiny local corpus."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("tiny_corpus.txt"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/smoke")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    import torch
    from cs336_basics.bpe import BPE
    from cs336_basics.Transformer import TransformerLM

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(1)
    torch.manual_seed(2025)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = args.dataset.read_text(encoding="utf-8")
    tokenizer = BPE()
    tokenizer.train(text.split(), num_merges=20)
    prefix = str(args.output_dir / "bpe")
    tokenizer.save(prefix)
    restored = BPE()
    restored.load(prefix)
    ids = tokenizer.encode_to_ids(text)
    assert ids == restored.encode_to_ids(text), "BPE save/load changed token IDs"
    assert len(ids) == len(tokenizer.encode(text)), "Known corpus tokens were silently dropped"
    assert tokenizer.decode(tokenizer.encode(text)) == " ".join(text.split())
    assert len(ids) > 16, "Supply a corpus with more than 16 encoded tokens"
    explicit = BPE()
    explicit.load_files(prefix + "_vocab.pkl", prefix + "_merges.pkl")
    assert explicit.vocab == restored.vocab
    try:
        explicit.load(str(args.output_dir / "does-not-exist"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing tokenizer files must fail explicitly")

    config = dict(vocab_size=tokenizer.get_vocab_size(), context_length=16,
                  d_model=32, num_layers=2, num_heads=4, d_ff=64, rope_theta=10000.0)
    model = TransformerLM(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.tensor(ids, dtype=torch.long, device=args.device)
    x = tokens[:16].unsqueeze(0)
    y = tokens[1:17].unsqueeze(0)
    before = [p.detach().clone() for p in model.parameters()]
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert torch.isfinite(loss), "Non-finite training loss"
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    checkpoint = args.output_dir / "tiny.pt"
    torch.save(dict(model_config=config, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), step=args.steps - 1), checkpoint)
    state = torch.load(checkpoint, map_location=args.device, weights_only=True)
    reloaded = TransformerLM(**state["model_config"]).to(args.device)
    reloaded.load_state_dict(state["model_state_dict"])
    resumed_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(state["optimizer_state_dict"])
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], reloaded(x)[0], rtol=0, atol=0)
    # An identical next update verifies optimizer state, not just model weights.
    for net, opt in ((model, optimizer), (reloaded, resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        net(x, y)[1].backward()
        opt.step()
    for a, b in zip(model.parameters(), reloaded.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        generated = x.clone()
        for _ in range(4):
            logits, _ = reloaded(generated[:, -16:])
            generated = torch.cat((generated, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    report = dict(status="passed", device=args.device, seed=2025, steps=args.steps,
                  losses=losses, generated_ids=generated[0].tolist(), torch=torch.__version__)
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: BPE, training, checkpoint resume and generation; loss={losses[-1]:.4f}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
