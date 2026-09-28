"""Compare FlashAttention2 forward/backward with dense PyTorch attention."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/smoke")
    args = parser.parse_args()
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("This example requires Linux + NVIDIA CUDA GPU + Triton. GPU check NOT RUN.")
    from flashattention2 import FlashAttention2
    # Compare FP32 math; reduced-precision TF32 is a separate performance choice.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2025)
    results = []
    for length in (64, 65, 128):
        for causal in (False, True):
            tensors = [torch.randn(2, length, 32, device=args.device, requires_grad=True) for _ in range(3)]
            q, k, v = tensors
            grad = torch.randn_like(q)
            actual = FlashAttention2.apply(q, k, v, causal)
            actual_grads = torch.autograd.grad(actual, tensors, grad)
            rq, rk, rv = [t.detach().clone().requires_grad_(True) for t in tensors]
            scores = rq @ rk.transpose(-2, -1) / (q.shape[-1] ** 0.5)
            if causal:
                mask = torch.ones(length, length, device=args.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, -float("inf"))
            expected = scores.softmax(-1) @ rv
            expected_grads = torch.autograd.grad(expected, (rq, rk, rv), grad)
            errors = []
            for a, b in zip((actual, *actual_grads), (expected, *expected_grads)):
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
                errors.append((a - b).abs().max().item())
            results.append(dict(length=length, causal=causal, max_abs_errors=errors))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status="passed", device=torch.cuda.get_device_name(args.device),
                  torch=torch.__version__, seed=2025, cases=results)
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: 6 attention cases, each checking forward and dQ/dK/dV")


if __name__ == "__main__":
    main()
