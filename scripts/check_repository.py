"""Audit tracked distribution files without installing training dependencies."""
import ast
from pathlib import Path
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = [ROOT / p for p in paths if p]
    if not files:
        raise SystemExit("No tracked files to audit; stage the distribution first")
    total = 0
    for path in files:
        parts = path.relative_to(ROOT).parts
        assert not set(parts) & {".venv", "__pycache__", "outputs", "checkpoints", "node_modules", "wandb", ".git"}, path
        assert path.is_file() and not path.is_symlink(), path
        size = path.stat().st_size
        assert size < 10 * 2**20, f"Unexpected large file: {path}"
        total += size
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        elif path.suffix in {".toml", ".lock"}:
            tomllib.loads(path.read_text(encoding="utf-8"))
    assert total < 25 * 2**20, f"Distribution is {total / 2**20:.2f} MiB"
    for path in [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *ROOT.glob("assignment*/README.md"), *ROOT.glob("docs/*.md")]:
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            assert (path.parent / target.split("#")[0]).exists(), f"Broken link: {path}: {target}"
    for assignment, expected in (("assignment1-basics", 21), ("assignment4-data", 11)):
        tree = ast.parse((ROOT / assignment / "tests/adapters.py").read_text())
        count = sum(isinstance(n, ast.Raise) and isinstance(n.exc, ast.Name) and n.exc.id == "NotImplementedError" for n in ast.walk(tree))
        assert count == expected, f"Update documented adapter status: {assignment}: {count}"
    for relative in ["assignment1-basics/tests/fixtures/ts_tests/model.pt", "assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl"]:
        assert ROOT / relative in files, f"Missing required test fixture: {relative}"
    print(f"PASS: {len(files)} tracked files, {total / 2**20:.2f} MiB; syntax, links, fixtures and adapter status")


if __name__ == "__main__":
    main()
