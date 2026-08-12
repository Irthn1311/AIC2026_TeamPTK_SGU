from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from triage_eg.retrieval.stage1b.adapters.openai_clip_official import (
    NetworkDownloadAttempted,
    OfficialOpenAIClipAdapter,
    _device_details,
    _numpy_output,
    controlled_import_clip,
    materialize_kaggle_expanded_tokenizer,
    preflight_official_openai_clip,
    resolve_official_asset_paths,
)
from triage_eg.retrieval.stage1b.asset_bundle import (
    AssetBundleConfig,
    build_openai_clip_asset_bundle,
)
from triage_eg.retrieval.stage1b.assets import preflight_candidate, sha256_file
from triage_eg.retrieval.stage1b.contracts import CandidateContract
from triage_eg.retrieval.stage1b.registry import (
    load_candidate_registry,
    write_official_runtime_config,
)


@pytest.fixture(autouse=True)
def clean_clip_modules():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "clip" or name.startswith("clip.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    original_torch = sys.modules.get("torch")
    if original_torch is None:
        sys.modules["torch"] = fake_torch_module()
    yield
    for name in list(sys.modules):
        if name == "clip" or name.startswith("clip."):
            sys.modules.pop(name, None)
    sys.modules.update(saved)
    if original_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original_torch


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype

    def __len__(self):
        return len(self.values)

    def detach(self):
        return self

    def float(self):
        return FakeTensor(self.values.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def to(self, device):
        return self


def fake_torch_module() -> ModuleType:
    module = ModuleType("torch")
    module.__version__ = "fake"
    module.cuda = SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda index: None,
        empty_cache=lambda: None,
    )
    module.stack = lambda tensors: FakeTensor(np.stack([tensor.values for tensor in tensors]))
    module.ones = lambda shape, dtype=None: FakeTensor(np.ones(shape))
    module.zeros = lambda shape, dtype=None: FakeTensor(np.zeros(shape))
    module.float32 = np.float32
    module.float16 = np.float16
    module.int64 = np.int64

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    module.no_grad = NoGrad
    return module


def make_asset(
    root: Path,
    *,
    package_source: str | None = None,
    checkpoint: bytes = b"checkpoint",
    declared_hash: bool = True,
    manifest: bool = True,
) -> tuple[Path, Path, Path]:
    asset = root / "asset"
    source = asset / "source/openai_clip"
    package = source / "clip"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        package_source
        or "def load(*args, **kwargs): return None\ndef tokenize(*args, **kwargs): return None\n",
        encoding="utf-8",
    )
    (package / "clip.py").write_text("_download = None\n", encoding="utf-8")
    (package / "model.py").write_text("", encoding="utf-8")
    (package / "simple_tokenizer.py").write_text("", encoding="utf-8")
    (package / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"tokenizer")
    (source / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    (source / "requirements.txt").write_text("torch\n", encoding="utf-8")
    (source / "LICENSE").write_text("license", encoding="utf-8")
    checkpoint_path = asset / "checkpoint/ViT-B-32.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(checkpoint)
    manifests = asset / "manifests"
    manifests.mkdir()
    digest = sha256_file(checkpoint_path) if checkpoint else ""
    if declared_hash:
        (manifests / "checkpoint.sha256").write_text(
            f"{digest}  checkpoint/ViT-B-32.pt\n", encoding="utf-8"
        )
    (manifests / "SOURCE_COMMIT.txt").write_text("source-commit\n", encoding="utf-8")
    if manifest:
        (manifests / "asset_manifest.json").write_text(
            json.dumps(
                {
                    "asset_bundle_version": "0.1.0",
                    "implementation": "openai_clip",
                    "source_repository": "OpenAI/CLIP",
                    "source_commit": "source-commit",
                    "architecture": "ViT-B/32",
                    "pretrained": "openai",
                    "checkpoint_relative_path": "checkpoint/ViT-B-32.pt",
                    "checkpoint_sha256": digest,
                    "internet_required_at_runtime": False,
                }
            ),
            encoding="utf-8",
        )
    return asset, source, checkpoint_path


def official_contract(source: Path, checkpoint: Path, **changes) -> CandidateContract:
    values = {
        "candidate_id": "openai_clip_vit_b32_openai_official",
        "enabled": True,
        "implementation": "openai_clip",
        "architecture": "ViT-B/32",
        "pretrained": "openai",
        "source_root": str(source),
        "checkpoint_path": str(checkpoint),
        "asset_manifest_path": str(checkpoint.parent.parent / "manifests/asset_manifest.json"),
        "tokenizer": "official clip.tokenize",
        "context_length": 77,
        "image_preprocessing": {
            "source": "official_clip_load_return_value",
            "manual_preprocess_override": False,
        },
        "text_preprocessing": {
            "strip": False,
            "lowercase": False,
            "unicode_normalization": None,
        },
        "device": "cpu",
        "batch_size": 2,
    }
    values.update(changes)
    return CandidateContract(**values)


def issue_codes(issues: list[dict]) -> set[str]:
    return {item["code"] for item in issues}


def test_resolve_asset_root_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AIC_OPENAI_CLIP_ASSET_ROOT",
        "AIC_OPENAI_CLIP_SOURCE_ROOT",
        "AIC_OPENAI_CLIP_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    paths = resolve_official_asset_paths()
    assert paths.source_root.as_posix().endswith("/aic2026-openai-clip-vit-b32/source/openai_clip")
    assert paths.checkpoint_path.name == "ViT-B-32.pt"


def test_explicit_source_and_checkpoint_override(tmp_path: Path) -> None:
    paths = resolve_official_asset_paths(
        tmp_path / "asset", tmp_path / "source", tmp_path / "weights.pt"
    )
    assert paths.source_root == (tmp_path / "source").resolve()
    assert paths.checkpoint_path == (tmp_path / "weights.pt").resolve()


def test_unresolved_environment_candidate_is_blocked(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        Path("configs/retrieval/stage1b_openai_clip_official_kaggle.yaml").read_text(),
        encoding="utf-8",
    )
    candidates, _, _, _ = load_candidate_registry(config)
    provenance, issues = preflight_candidate(candidates[0], tmp_path, tmp_path)
    assert not provenance["reproducible"]
    assert issue_codes(issues) == {"ENCODER_ENVIRONMENT_PATH_UNRESOLVED"}


def test_missing_source_root(tmp_path: Path) -> None:
    asset, _, checkpoint = make_asset(tmp_path)
    paths = resolve_official_asset_paths(asset, tmp_path / "missing", checkpoint)
    _, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert "OPENAI_CLIP_SOURCE_ROOT_MISSING" in issue_codes(issues)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [(None, "ENCODER_ASSET_NOT_FOUND"), (b"", "ENCODER_CHECKPOINT_INVALID")],
)
def test_missing_or_empty_checkpoint(tmp_path: Path, payload: bytes | None, expected: str) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    if payload is None:
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(payload)
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    _, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert expected in issue_codes(issues)


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    manifest_path = asset / "manifests/asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_relative_path"] = "../../outside.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    _, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert "ASSET_MANIFEST_PATH_TRAVERSAL" in issue_codes(issues)


def test_valid_controlled_source_import(tmp_path: Path) -> None:
    _, source, _ = make_asset(tmp_path)
    module, details, issues = controlled_import_clip(source)
    assert module is not None and not issues
    assert details["module_origin_valid"] and details["required_api_present"]
    assert Path(details["module_file"]).is_relative_to(source)


def test_controlled_import_reports_missing_runtime_dependency(tmp_path: Path) -> None:
    package_source = (
        "import triage_eg_intentionally_missing_dependency\n"
        "def load(*args, **kwargs): return None\n"
        "def tokenize(*args, **kwargs): return None\n"
    )
    _, source, _ = make_asset(tmp_path, package_source=package_source)
    module, details, issues = controlled_import_clip(source)
    assert module is None
    assert issues == ["ENCODER_DEPENDENCY_NOT_AVAILABLE"]
    assert details["missing_dependency"] == "triage_eg_intentionally_missing_dependency"


def test_kaggle_expanded_tokenizer_is_restored_in_working_copy(tmp_path: Path) -> None:
    _, source, _ = make_asset(tmp_path)
    compressed = source / "clip/bpe_simple_vocab_16e6.txt.gz"
    compressed.unlink()
    expanded = source / "clip/bpe_simple_vocab_16e6.txt"
    expanded.write_bytes(b"tokenizer contents")
    runtime, restored = materialize_kaggle_expanded_tokenizer(
        source, tmp_path / "working/openai_clip"
    )
    assert restored and runtime != source
    assert expanded.is_file() and not compressed.exists()
    with gzip.open(runtime / "clip/bpe_simple_vocab_16e6.txt.gz", "rb") as stream:
        assert stream.read() == b"tokenizer contents"


def test_preflight_imports_dependency_from_offline_wheel(tmp_path: Path) -> None:
    dependency = "triage_eg_offline_test_dependency"
    package_source = (
        f"import {dependency}\n"
        "def load(*args, **kwargs): return None\n"
        "def tokenize(*args, **kwargs): return None\n"
    )
    asset, source, checkpoint = make_asset(tmp_path, package_source=package_source)
    wheel = asset / "source/dependencies/dependency-1.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    with ZipFile(wheel, "w") as archive:
        archive.writestr(f"{dependency}.py", "VALUE = 'offline'\n")
    try:
        provenance, issues, module = preflight_official_openai_clip(
            resolve_official_asset_paths(asset, source, checkpoint), requested_device="cpu"
        )
        assert module is not None
        assert "ENCODER_DEPENDENCY_NOT_AVAILABLE" not in issue_codes(issues)
        assert provenance["offline_dependency_wheels"] == [str(wheel.resolve())]
    finally:
        sys.modules.pop(dependency, None)


def test_existing_wrong_clip_origin_is_rejected(tmp_path: Path) -> None:
    wrong = ModuleType("clip")
    wrong.__file__ = str(tmp_path / "wrong/clip/__init__.py")
    sys.modules["clip"] = wrong
    _, source, _ = make_asset(tmp_path)
    _, _, issues = controlled_import_clip(source)
    assert issues == ["OPENAI_CLIP_MODULE_ORIGIN_MISMATCH"]


@pytest.mark.parametrize(
    ("source_text", "expected"),
    [
        ("def tokenize(*args, **kwargs): return None\n", "OPENAI_CLIP_REQUIRED_API_MISSING"),
        ("def load(*args, **kwargs): return None\n", "OPENAI_CLIP_REQUIRED_API_MISSING"),
    ],
)
def test_required_official_api_is_enforced(tmp_path: Path, source_text: str, expected: str) -> None:
    _, source, _ = make_asset(tmp_path, package_source=source_text)
    _, _, issues = controlled_import_clip(source)
    assert expected in issues


def test_missing_tokenizer_asset_is_blocking(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    (source / "clip/bpe_simple_vocab_16e6.txt.gz").unlink()
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    _, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert "OPENAI_CLIP_TOKENIZER_ASSET_MISSING" in issue_codes(issues)


def test_streaming_hash_and_declared_hash_match(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    provenance, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert provenance["checkpoint_sha256"] == sha256_file(checkpoint)
    assert provenance["declared_hash_match"]
    assert "CHECKPOINT_HASH_MISMATCH" not in issue_codes(issues)


def test_declared_hash_mismatch_blocks(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    (asset / "manifests/checkpoint.sha256").write_text("0" * 64, encoding="utf-8")
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    _, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert "CHECKPOINT_HASH_MISMATCH" in issue_codes(issues)


def test_missing_declared_hash_warns_but_does_not_block(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path, declared_hash=False, manifest=False)
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    provenance, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert provenance["reproducible"]
    assert "CHECKPOINT_DECLARED_HASH_MISSING" in issue_codes(issues)
    assert not any(item["severity"] == "ERROR" for item in issues)


def test_source_commit_and_manifest_are_recorded(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    provenance, _, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    assert provenance["source_commit"] == "source-commit"
    assert provenance["source_repository"] == "OpenAI/CLIP"
    assert provenance["asset_manifest"]["implementation"] == "openai_clip"


def test_malformed_manifest_and_unknown_source_commit_are_reported(
    tmp_path: Path,
) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    (asset / "manifests/asset_manifest.json").write_text("{bad", encoding="utf-8")
    (asset / "manifests/SOURCE_COMMIT.txt").unlink()
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    _, issues, _ = preflight_official_openai_clip(paths, requested_device="cpu")
    codes = issue_codes(issues)
    assert "ASSET_MANIFEST_MALFORMED" in codes
    assert "SOURCE_COMMIT_UNKNOWN" in codes


def make_runtime_module(*, download: bool = False, long_text: bool = False):
    import torch

    calls = SimpleNamespace(load=[], tokenize=[], preprocess=0, image_batches=0)

    class Model:
        def eval(self):
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype="float32")])

        def encode_image(self, batch):
            calls.image_batches += 1
            output = np.zeros((len(batch), 512), dtype=np.float16)
            output[:, 0] = 2
            return FakeTensor(output)

        def encode_text(self, tokens):
            output = np.zeros((len(tokens), 512), dtype=np.float16)
            output[:, 0] = 3
            return FakeTensor(output)

    def preprocess(image):
        calls.preprocess += 1
        return torch.ones((3, 2, 2), dtype=torch.float32)

    module = ModuleType("clip")

    def download_helper(*args, **kwargs):
        return None

    module._download = download_helper

    def load(path, device, jit):
        calls.load.append((path, device, jit))
        if download:
            module._download(path)
        return Model(), preprocess

    def tokenize(texts, truncate=False):
        calls.tokenize.append((list(texts), truncate))
        if long_text and not truncate:
            raise RuntimeError("too long")
        return torch.ones((len(texts), 77), dtype=torch.int64)

    module.load = load
    module.tokenize = tokenize
    return module, calls


def loaded_adapter(
    tmp_path: Path,
    *,
    module: ModuleType | None = None,
    contract_changes: dict | None = None,
) -> tuple[OfficialOpenAIClipAdapter, SimpleNamespace, Path]:
    asset, source, checkpoint = make_asset(tmp_path)
    paths = resolve_official_asset_paths(asset, source, checkpoint)
    runtime_module, calls = (module, SimpleNamespace()) if module else make_runtime_module()
    provenance = {
        "selected_device": "cpu",
        "checkpoint_sha256": sha256_file(checkpoint),
        "module_file": str(source / "clip/__init__.py"),
    }
    contract = official_contract(source, checkpoint, **(contract_changes or {}))
    return (
        OfficialOpenAIClipAdapter(contract, paths, runtime_module, provenance),
        calls,
        asset,
    )


def test_local_absolute_checkpoint_is_passed_to_clip_load(tmp_path: Path) -> None:
    adapter, calls, _ = loaded_adapter(tmp_path)
    path, device, jit = calls.load[0]
    assert Path(path).is_absolute() and Path(path).name == "ViT-B-32.pt"
    assert (device, jit) == ("cpu", False)
    adapter.close()


def test_download_helper_invocation_is_blocked(tmp_path: Path) -> None:
    module, _ = make_runtime_module(download=True)
    with pytest.raises(NetworkDownloadAttempted, match="NETWORK_DOWNLOAD_ATTEMPTED"):
        loaded_adapter(tmp_path, module=module)


def test_official_preprocess_and_image_batching_are_used(tmp_path: Path) -> None:
    from PIL import Image

    adapter, calls, _ = loaded_adapter(tmp_path)
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2)).save(path)
        paths.append(path)
    result = adapter.encode_images(paths)
    assert result.shape == (3, 512) and result.dtype == np.float32
    assert calls.preprocess == 3 and calls.image_batches == 2
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    assert adapter.last_image_metrics["raw_norms"] == [2.0, 2.0, 2.0]
    assert adapter.runtime_manifest()["manual_preprocess_override"] is False
    adapter.close()


def test_file_and_in_memory_rgb_paths_have_exact_embedding_parity(tmp_path: Path) -> None:
    from PIL import Image

    adapter, calls, _ = loaded_adapter(tmp_path)
    array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    path = tmp_path / "frame.png"
    Image.fromarray(array, mode="RGB").save(path)
    file_embedding = adapter.encode_images([path])
    memory_embedding = adapter.encode_rgb_arrays([array])
    assert np.array_equal(file_embedding, memory_embedding)
    assert calls.preprocess == 2
    assert adapter.last_image_metrics["input_kind"] == "rgb_array"
    adapter.close()


def test_in_memory_rgb_rejects_non_rgb_or_non_uint8(tmp_path: Path) -> None:
    adapter, _, _ = loaded_adapter(tmp_path)
    with pytest.raises(ValueError, match="RGB_ARRAY_INVALID"):
        adapter.encode_rgb_arrays([np.ones((2, 2), dtype=np.uint8)])
    with pytest.raises(ValueError, match="RGB_ARRAY_INVALID"):
        adapter.encode_rgb_arrays([np.ones((2, 2, 3), dtype=np.float32)])
    adapter.close()


def test_official_tokenize_and_text_normalization(tmp_path: Path) -> None:
    adapter, calls, _ = loaded_adapter(tmp_path)
    result = adapter.encode_texts(["hello", "xin chào"])
    assert result.shape == (2, 512) and result.dtype == np.float32
    assert calls.tokenize[-1] == (["hello", "xin chào"], False)
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    assert adapter.last_text_metrics["raw_norms"] == [3.0, 3.0]
    adapter.close()


def test_over_context_text_rejected_by_default(tmp_path: Path) -> None:
    module, _ = make_runtime_module(long_text=True)
    adapter, _, _ = loaded_adapter(tmp_path, module=module)
    with pytest.raises(ValueError, match="TEXT_CONTEXT_LENGTH_EXCEEDED"):
        adapter.encode_text(["long"])


def test_explicit_text_truncation_is_recorded(tmp_path: Path) -> None:
    module, calls = make_runtime_module(long_text=True)
    adapter, _, _ = loaded_adapter(
        tmp_path, module=module, contract_changes={"text_truncate": True}
    )
    adapter.encode_text(["long"])
    assert adapter.last_text_metrics["text_was_truncated"] == [True]
    assert calls.tokenize[-1] == (["long"], True)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.ones((1, 511)), "DIMENSION"),
        (np.full((1, 512), np.nan), "NON_FINITE"),
        (np.zeros((1, 512)), "ZERO_NORM"),
    ],
)
def test_official_output_validation(values: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _numpy_output(values, 1)


def test_device_auto_cpu_and_cuda_policy() -> None:
    cpu = SimpleNamespace(
        __version__="test",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    assert _device_details(cpu, "auto")[0] == "cpu"
    cuda = SimpleNamespace(
        __version__="test",
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "Fake GPU"),
    )
    selected, details, issues = _device_details(cuda, "auto")
    assert selected == "cuda:0" and details["gpu_name"] == "Fake GPU" and not issues


def test_runtime_config_writes_resolved_absolute_paths(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    output = tmp_path / "runtime.yaml"
    write_official_runtime_config(
        "configs/retrieval/stage1b_openai_clip_official_kaggle.yaml",
        output,
        asset_root=asset,
        source_root=source,
        checkpoint_path=checkpoint,
        device="cpu",
        batch_size=8,
    )
    candidates, _, _, fingerprint = load_candidate_registry(output)
    candidate = candidates[0]
    assert Path(candidate.source_root).is_absolute()
    assert Path(candidate.checkpoint_path).is_absolute()
    assert candidate.device == "cpu" and candidate.batch_size == 8
    assert len(fingerprint) == 64


def test_asset_bundle_dry_run_creates_nothing(tmp_path: Path) -> None:
    _, source, checkpoint = make_asset(tmp_path)
    output = tmp_path / "portable/bundle"
    result = build_openai_clip_asset_bundle(
        AssetBundleConfig(source, checkpoint, output, dry_run=True)
    )
    assert result["dry_run"] and not output.exists()


def test_asset_bundle_contains_runtime_assets_and_excludes_git(tmp_path: Path) -> None:
    _, source, checkpoint = make_asset(tmp_path)
    (source / ".git").mkdir()
    (source / ".git/config").write_text("secret", encoding="utf-8")
    (source / "tests").mkdir()
    (source / "tests/test.py").write_text("ignored", encoding="utf-8")
    output = tmp_path / "portable/bundle"
    dependency_wheel = tmp_path / "wheels/ftfy-6.3.1-py3-none-any.whl"
    dependency_wheel.parent.mkdir()
    dependency_wheel.write_bytes(b"wheel")
    result = build_openai_clip_asset_bundle(
        AssetBundleConfig(
            source,
            checkpoint,
            output,
            source_commit="abc123",
            create_zip=True,
            dependency_wheels=(dependency_wheel,),
        )
    )
    assert (output / "source/openai_clip/clip/__init__.py").is_file()
    assert (output / "source/openai_clip/clip/bpe_simple_vocab_16e6.txt.gz").is_file()
    assert (output / "source/openai_clip/LICENSE").is_file()
    assert (output / "manifests/checkpoint.sha256").is_file()
    assert (output / "manifests/asset_manifest.json").is_file()
    assert (output / "manifests/source_provenance.json").is_file()
    assert (output / "manifests/file_inventory.jsonl").is_file()
    assert (output / "source/dependencies/ftfy-6.3.1-py3-none-any.whl").is_file()
    manifest = json.loads((output / "manifests/asset_manifest.json").read_text())
    assert manifest["source_repository"] == "https://github.com/openai/CLIP.git"
    assert manifest["checkpoint_filename"] == "ViT-B-32.pt"
    assert manifest["checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert manifest["runtime_model_load_policy"] == "absolute_local_checkpoint_path_only"
    assert manifest["offline_dependency_wheels"][0]["relative_path"] == (
        "source/dependencies/ftfy-6.3.1-py3-none-any.whl"
    )
    provenance = json.loads((output / "manifests/source_provenance.json").read_text())
    assert provenance["source_commit"] == "abc123"
    assert provenance["source_destination"] == "source/openai_clip"
    assert not provenance["nested_git_directory_included"]
    inventory = [
        json.loads(line)
        for line in (output / "manifests/file_inventory.jsonl").read_text().splitlines()
    ]
    assert inventory == sorted(inventory, key=lambda item: item["relative_path"])
    assert all(set(item) == {"relative_path", "size_bytes", "sha256"} for item in inventory)
    assert not (output / "source/openai_clip/.git").exists()
    assert not (output / "source/openai_clip/tests").exists()
    zip_path = Path(result["zip_path"])
    with ZipFile(zip_path) as archive:
        assert zip_path.name not in archive.namelist()
        assert not any(".git/" in name or "/tests/" in name for name in archive.namelist())


def test_asset_bundle_refuses_overwrite(tmp_path: Path) -> None:
    _, source, checkpoint = make_asset(tmp_path)
    output = tmp_path / "portable/bundle"
    build_openai_clip_asset_bundle(AssetBundleConfig(source, checkpoint, output))
    with pytest.raises(FileExistsError):
        build_openai_clip_asset_bundle(AssetBundleConfig(source, checkpoint, output))


def test_asset_bundle_preserves_checkpoint_when_input_is_inside_output(tmp_path: Path) -> None:
    asset, source, checkpoint = make_asset(tmp_path)
    checkpoint_identity = checkpoint.stat().st_ino
    build_openai_clip_asset_bundle(
        AssetBundleConfig(
            source,
            checkpoint,
            asset,
            source_commit="abc123",
            overwrite=True,
        )
    )
    assert checkpoint.is_file()
    assert checkpoint.stat().st_ino == checkpoint_identity
    assert checkpoint.read_bytes() == b"checkpoint"
    assert (asset / "source/openai_clip/clip/__init__.py").is_file()
    assert (asset / "manifests/source_provenance.json").is_file()


def test_official_stage1b_sources_have_no_download_clients() -> None:
    roots = [
        Path("src/triage_eg/retrieval/stage1b/adapters/openai_clip_official.py"),
        Path("src/triage_eg/retrieval/stage1b/asset_bundle.py"),
        Path("scripts/build_openai_clip_asset_bundle.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in roots)
    for forbidden in (
        "requests.get",
        "urllib.request",
        "torch.hub",
        "pip install",
        "git clone",
        "wget ",
        "curl ",
    ):
        assert forbidden not in source


def test_notebook_uses_runtime_yaml_and_contains_no_model_download_cell() -> None:
    notebook = json.loads(
        Path("notebooks/07_stage1b_encoder_compatibility.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "stage1b_openai_clip_runtime.yaml" in source
    assert "write_official_runtime_config" in source
    assert "materialize_kaggle_expanded_tokenizer" in source
    assert "!pip install" not in source
    assert "%pip install" not in source
    assert "OpenAI/CLIP.git" not in source
    assert "This run tests the official OpenAI CLIP ViT-B/32 hypothesis using only local " in source


def test_default_openclip_template_remains_disabled() -> None:
    candidates, _, _, _ = load_candidate_registry(
        "configs/retrieval/stage1b_encoder_candidates.yaml"
    )
    openclip = [item for item in candidates if item.implementation == "open_clip"]
    assert openclip and all(not item.enabled for item in openclip)
