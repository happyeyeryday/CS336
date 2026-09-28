"""Small distributed correctness checks, using CPU/Gloo or GPU/NCCL."""
import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ddp_worker(rank, world_size, config):
    import torch
    import torch.distributed as dist
    from dp.ddp import DDPModel, DDPBucketed
    from dp.optimizer import ShardedOptimizer

    torch.set_num_threads(1)
    device = f"cuda:{rank}" if config["device"] == "cuda" else "cpu"
    if config["device"] == "cuda":
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(config["port"])
    dist.init_process_group("nccl" if config["device"] == "cuda" else "gloo",
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=120))
    try:
        results = []
        for bucketed in (False, True):
            torch.manual_seed(2025)
            model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)).to(device)
            reference = copy.deepcopy(model)
            wrapper = DDPBucketed(model, bucket_size_mb=0.0001) if bucketed else DDPModel(model)
            optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-3)
            expected_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
            x = torch.randn(world_size * 4, 4, device=device)
            y = torch.randn(world_size * 4, 2, device=device)
            max_error = 0.0
            for step in range(3):
                optimizer.zero_grad(set_to_none=(step % 2 == 0))
                expected_opt.zero_grad()
                torch.nn.functional.mse_loss(reference(x), y).backward()
                local_x = x[rank * 4:(rank + 1) * 4]
                local_y = y[rank * 4:(rank + 1) * 4]
                if bucketed:
                    with wrapper.no_sync():
                        (torch.nn.functional.mse_loss(wrapper(local_x[:2]), local_y[:2]) / 2).backward()
                    (torch.nn.functional.mse_loss(wrapper(local_x[2:]), local_y[2:]) / 2).backward()
                else:
                    torch.nn.functional.mse_loss(wrapper(local_x), local_y).backward()
                wrapper.finish_gradient_synchronization()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
                optimizer.step()
                expected_opt.step()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    max_error = max(max_error, (actual - expected).abs().max().item())
            results.append(dict(wrapper="bucketed+no_sync" if bucketed else "per_parameter",
                                steps=3, max_parameter_error=max_error))
        if rank == 0:
            write_report(config, results)
    finally:
        dist.destroy_process_group()


def make_fixture(mode):
    import torch
    torch.manual_seed(2025)
    d = 32
    if mode == "pp":
        return {"x": torch.randn(8, d), **{f"linears.{i}.weight": torch.randn(d, d) / math.sqrt(d) for i in range(8)}}
    result = {"x": torch.randn(2, 8, d)}
    for layer in ("ln1", "ln2"):
        result[f"{layer}.weight"] = torch.ones(d)
        result[f"{layer}.bias"] = torch.zeros(d)
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        result[f"attn.{proj}.weight"] = torch.randn(d, d) / math.sqrt(d)
    result["ffn.fc1.weight"] = torch.randn(64, d) / math.sqrt(d)
    result["ffn.fc2.weight"] = torch.randn(d, 64) / math.sqrt(64)
    return result


def check_parallel_result(config, fixture):
    import numpy as np
    import torch
    from torch.nn import functional as F
    values = {k: v.detach().clone().requires_grad_(True) for k, v in fixture.items()}
    x = values["x"]
    if config["mode"] == "pp":
        out = x
        for i in range(8):
            out = F.gelu(F.linear(out, values[f"linears.{i}.weight"]))
        # Retain the original assignment's PP loss scale.
        (out.sum() * 1e3).backward()
    else:
        h = F.layer_norm(x, (32,), values["ln1.weight"], values["ln1.bias"])
        projections = [F.linear(h, values[f"attn.{p}_proj.weight"]).reshape(2, 8, 4, 8).transpose(1, 2) for p in ("q", "k", "v")]
        q, k, v = projections
        attention = ((q @ k.transpose(-2, -1)) / math.sqrt(8)).softmax(-1) @ v
        h = x + F.linear(attention.transpose(1, 2).reshape(2, 8, 32), values["attn.out_proj.weight"])
        normalized = F.layer_norm(h, (32,), values["ln2.weight"], values["ln2.bias"])
        out = h + F.linear(F.gelu(F.linear(normalized, values["ffn.fc1.weight"])), values["ffn.fc2.weight"])
        out.sum().backward()
    expected = {"out": out.detach()}
    for name, value in values.items():
        expected[name] = value.detach()
        expected[name + "_grad"] = value.grad
    errors = {}
    with np.load(Path(config["output_dir"]) / f"study_{config['mode']}.npz") as saved:
        assert set(saved.files) == set(expected), "Missing output or gradient in results"
        for name, value in expected.items():
            actual = torch.from_numpy(saved[name])
            assert torch.isfinite(actual).all(), name
            torch.testing.assert_close(actual, value, atol=2e-4, rtol=5e-4, msg=lambda msg: f"{name}: {msg}")
            errors[name] = (actual - value).abs().max().item()
    write_report(config, errors)


def write_report(config, checks):
    import torch
    report = dict(status="passed", mode=config["mode"], device=config["device"],
                  world_size=config["world_size"], seed=2025, torch=torch.__version__, checks=checks)
    path = Path(config["output_dir"]) / f"{config['mode']}_result.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {config['mode'].upper()} reference checks ({config['device']}); {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ddp", "tp", "pp"], default="ddp")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--port", type=int, default=0, help="0 chooses an available local port")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/smoke")
    args = parser.parse_args()
    if not 1 <= args.nproc_per_node <= 4:
        parser.error("the tiny fixture supports 1 to 4 processes")
    if args.mode == "pp" and args.nproc_per_node != 2:
        parser.error("PP requires exactly two processes")
    if args.mode == "tp" and 4 % args.nproc_per_node:
        parser.error("TP process count must divide the fixture's 4 attention heads")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    import torch
    import torch.multiprocessing as mp
    if args.device == "cuda" and torch.cuda.device_count() < args.nproc_per_node:
        parser.error(f"Need {args.nproc_per_node} visible CUDA GPUs; GPU check NOT RUN. CPU/Gloo is available.")
    torch.set_num_threads(1)
    if args.port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            args.port = sock.getsockname()[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(mode=args.mode, device=args.device, world_size=args.nproc_per_node,
                  port=args.port, output_dir=str(args.output_dir.resolve()), n_heads=4, d_ff=64)
    if args.mode == "ddp":
        mp.spawn(ddp_worker, args=(args.nproc_per_node, config), nprocs=args.nproc_per_node, join=True)
    else:
        from tpandpp.tppp import tensor_parallelism_main, pipeline_parallelism_main
        fixture = make_fixture(args.mode)
        target = tensor_parallelism_main if args.mode == "tp" else pipeline_parallelism_main
        mp.spawn(target, args=(args.nproc_per_node, fixture, config), nprocs=args.nproc_per_node, join=True)
        check_parallel_result(config, fixture)


if __name__ == "__main__":
    main()
