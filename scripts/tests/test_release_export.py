"""Export boundaries: no private files, symlink escapes, overwrites or changed sources."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("release_export", Path(__file__).parents[1] / "export_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize("path", ["../README.md", "/README.md", "C:/README.md", "backend\\app.py",
    "backend/.env", "backend/.env.test", "backend/data/records.json", "docs/agent-handoffs/state.json",
    "_compare/app.py", "frontend/node_modules/x.js", "certificate.pem", "runtime-logs/run.log",
    "backend/.test-tmp/result.json"])
def test_private_or_escaping_manifest_paths_are_rejected(path):
    with pytest.raises(ValueError):
        release.approved_path(path)


def fixture(root):
    (root / "app.py").write_text("print('synthetic')\n", encoding="utf-8")
    (root / "release-files.txt").write_text("app.py\nrelease-files.txt\n", encoding="utf-8")


def test_export_is_complete_and_cannot_overwrite_or_leave_workspace(tmp_path):
    fixture(tmp_path)
    files, findings = release.inspect(tmp_path)
    assert not findings
    destination = tmp_path / "evidence/release-candidates/synthetic"
    output = release.export(tmp_path, destination, files)
    assert {p.name for p in output.iterdir()} == {"app.py", "release-files.txt"}
    assert (output / "app.py").read_bytes() == (tmp_path / "app.py").read_bytes()
    assert len(json.loads((destination / "manifest.json").read_text())["files"]) == 2
    for other in (destination, tmp_path / "unapproved"):
        with pytest.raises(ValueError):
            release.export(tmp_path, other, files)


def test_secret_scan_reports_location_without_value(tmp_path):
    fixture(tmp_path)
    fake = b"ghp_" + b"A" * 36
    (tmp_path / "app.py").write_bytes(fake)
    _, findings = release.inspect(tmp_path)
    assert findings == [{"path": "app.py", "line": 1, "category": "github_token"}]
    assert fake.decode() not in json.dumps(findings)


def test_new_application_module_cannot_silently_be_omitted(tmp_path):
    fixture(tmp_path)
    module = tmp_path / "backend/app/services/memory/semantic_units.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="application modules missing"):
        release.inspect(tmp_path)
    with (tmp_path / "release-files.txt").open("a", encoding="utf-8") as manifest:
        manifest.write("backend/app/services/memory/semantic_units.py\n")
    files, findings = release.inspect(tmp_path)
    assert not findings and any(row["path"].endswith("semantic_units.py") for row in files)


def test_export_refuses_source_changed_after_inspection(tmp_path):
    fixture(tmp_path)
    files, _ = release.inspect(tmp_path)
    (tmp_path / "app.py").write_text("changed")
    with pytest.raises(ValueError, match="changed during export"):
        release.export(tmp_path, tmp_path / "evidence/release-candidates/changed", files)


@pytest.mark.parametrize('missing', ['manifest.json', 'tokenizer.json', 'LICENSE', 'UPSTREAM_README.md'])
def test_tokenizer_runtime_assets_and_licenses_cannot_silently_be_omitted(tmp_path, missing):
    import hashlib
    fixture(tmp_path)
    folder = tmp_path / 'backend/app/vendor/deepseek_v41'
    folder.mkdir(parents=True)
    metadata = {'files': {}}
    for name in ('encoding.py', 'tokenizer.json', 'LICENSE', 'UPSTREAM_README.md'):
        content = b'{}' if name.endswith('.json') else b'# synthetic\n'
        (folder / name).write_bytes(content)
        metadata['files'][name] = {'sha256': hashlib.sha256(content).hexdigest()}
    (folder / 'manifest.json').write_text(json.dumps(metadata))
    with (tmp_path / 'release-files.txt').open('a') as manifest:
        manifest.write(''.join(f'backend/app/vendor/deepseek_v41/{name}\n'
            for name in [*metadata['files'], 'manifest.json'] if name != missing))
    with pytest.raises(ValueError, match='missing'):
        release.inspect(tmp_path)
    with (tmp_path / 'release-files.txt').open('a') as manifest:
        manifest.write(f'backend/app/vendor/deepseek_v41/{missing}\n')
    assert not release.inspect(tmp_path)[1]
    (folder / 'tokenizer.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='runtime asset hash mismatch'):
        release.inspect(tmp_path)


def test_missing_duplicate_and_symlink_files_are_rejected(tmp_path):
    fixture(tmp_path)
    manifest = tmp_path / "release-files.txt"
    for body in ("missing.py\n", "app.py\napp.py\n"):
        manifest.write_text(body)
        with pytest.raises(ValueError):
            release.inspect(tmp_path)
    manifest.write_text("app.py\n")
    # File-system link creation requires extra privileges on some Windows
    # hosts; simulate its metadata, without weakening production resolution.
    from unittest.mock import patch
    with patch.object(Path, "is_symlink", return_value=True):
        with pytest.raises(ValueError, match="linked"):
            release.inspect(tmp_path)
