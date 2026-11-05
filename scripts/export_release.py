"""Export only the reviewed file list into a fresh candidate, without staging Git."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "release-files.txt"
APP_ASSET_MANIFESTS = ("backend/app/vendor/deepseek_v41/manifest.json",)
PRIVATE_PARTS = {".git", ".venv", "node_modules", "__pycache__", "_compare",
                 "evidence", "runtime-logs", "backups", "agent-handoffs", "data", ".test-tmp"}
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "provider_key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}


def approved_path(value):
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("invalid manifest path")
    if any(part.lower() in PRIVATE_PARTS for part in path.parts):
        raise ValueError("private directory in manifest")
    if (path.name.startswith(".env") and path.name != ".env.example") or path.suffix.lower() in (
            ".pem", ".key", ".p12", ".pfx", ".db", ".log", ".pyc"):
        raise ValueError("private file in manifest")
    return path


def source_file(root, value):
    path = approved_path(value)
    source = root / str(path)
    if (not source.is_file() or not source.resolve().is_relative_to(root.resolve())
            or any((root / Path(*path.parts[:i])).is_symlink()
                   for i in range(1, len(path.parts) + 1))):
        raise ValueError("missing or linked manifest file: " + value)
    return source


def inspect(root):
    entries = [line.strip() for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("manifest must contain unique files")
    # Application modules are one deployable unit; a successful allowlist scan
    # must not quietly produce an export missing a newly imported module.
    missing_modules = sorted(path.relative_to(root).as_posix()
                             for path in (root / "backend/app").rglob("*.py")
                             if "__pycache__" not in path.parts
                             and path.relative_to(root).as_posix() not in entries)
    if missing_modules:
        raise ValueError("application modules missing from manifest: " + ", ".join(missing_modules))
    asset_hashes = {}
    for manifest_path in APP_ASSET_MANIFESTS:
        asset_manifest = root / manifest_path
        if not asset_manifest.parent.exists():
            continue
        if manifest_path not in entries or not asset_manifest.is_file():
            raise ValueError("runtime asset manifest missing: " + manifest_path)
        metadata = json.loads(source_file(root, manifest_path).read_text(encoding="utf-8"))
        required = {"encoding.py", "tokenizer.json", "LICENSE", "UPSTREAM_README.md"}
        if not required.issubset(metadata.get("files", {})):
            raise ValueError("incomplete runtime asset manifest: " + manifest_path)
        for name, row in metadata["files"].items():
            relative = str(approved_path(name))
            value = (PurePosixPath(manifest_path).parent / relative).as_posix()
            if value not in entries:
                raise ValueError("runtime asset missing from manifest: " + value)
            asset_hashes[value] = row["sha256"]
    files, findings = [], []
    for value in sorted(entries):
        source = source_file(root, value)
        content = source.read_bytes()
        if value in asset_hashes and hashlib.sha256(content).hexdigest() != asset_hashes[value]:
            raise ValueError("runtime asset hash mismatch: " + value)
        if source.suffix.lower() not in (".woff2", ".ttf", ".png", ".jpg"):
            for category, pattern in SECRET_PATTERNS.items():
                for match in pattern.finditer(content):
                    findings.append({"path": value, "line": content[:match.start()].count(b"\n") + 1,
                                     "category": category})
        files.append({"path": value, "sha256": hashlib.sha256(content).hexdigest(),
                      "bytes": len(content)})
    return files, findings


def export(root, destination, files):
    if destination.exists() or not destination.resolve().is_relative_to(
            (root / "evidence/release-candidates").resolve()):
        raise ValueError("refusing existing or out-of-scope candidate directory")
    destination.mkdir(parents=True, exist_ok=False)
    source_copy = destination / "source"
    source_copy.mkdir()
    for row in files:
        source = root / row["path"]
        target = source_copy / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Detect edits between inspection and copy; never silently export a
        # file whose recorded hash or approved content was not inspected.
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("source changed during export: " + row["path"])
        with target.open("xb") as handle:
            handle.write(content)
    (destination / "manifest.json").write_text(json.dumps({"files": files,
            "secret_shape_findings": 0}, indent=2) + "\n", encoding="utf-8")
    return source_copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check candidate files without copying")
    parser.add_argument("--output", type=Path, help="new directory under evidence/release-candidates")
    args = parser.parse_args()
    try:
        files, findings = inspect(ROOT)
        print(json.dumps({"approved_files": len(files), "bytes": sum(r["bytes"] for r in files),
                          "secret_shape_findings": findings}, ensure_ascii=False))
        if findings:
            return 1
        if not args.check:
            destination = args.output or ROOT / "evidence/release-candidates" / ("candidate-" + uuid4().hex)
            print(export(ROOT, destination, files))
        return 0
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
