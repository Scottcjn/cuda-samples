import json
import stat
from pathlib import Path

import run_tests


def make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_normalize_exe_name_strips_windows_extension():
    assert run_tests.normalize_exe_name("matrixMul.exe") == "matrixMul"
    assert run_tests.normalize_exe_name("nbody") == "nbody"


def test_load_args_config_returns_empty_for_missing_or_invalid_files(tmp_path, capsys):
    assert run_tests.load_args_config(None) == {}
    assert run_tests.load_args_config(tmp_path / "missing.json") == {}

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json")
    assert run_tests.load_args_config(invalid_json) == {}
    assert "Failed to parse" in capsys.readouterr().out

    wrong_shape = tmp_path / "list.json"
    wrong_shape.write_text(json.dumps(["not", "a", "dict"]))
    assert run_tests.load_args_config(wrong_shape) == {}
    assert "must contain a dictionary" in capsys.readouterr().out


def test_load_args_config_reads_valid_dictionary(tmp_path):
    config = tmp_path / "args.json"
    config.write_text(json.dumps({"sample": {"args": ["--iterations", "1"]}}))

    assert run_tests.load_args_config(config) == {
        "sample": {"args": ["--iterations", "1"]}
    }


def test_find_executables_skips_libraries_and_non_executable_files(tmp_path):
    executable = make_executable(tmp_path / "matrixMul")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_executable = make_executable(nested_dir / "nbody")
    make_executable(tmp_path / "plugin.so")
    make_executable(tmp_path / "runtime.dylib")
    (tmp_path / "README.md").write_text("not executable")

    found = set(run_tests.find_executables(tmp_path))

    assert executable in found
    assert nested_executable in found
    assert tmp_path / "plugin.so" not in found
    assert tmp_path / "runtime.dylib" not in found
    assert tmp_path / "README.md" not in found


def test_get_gpu_count_prefers_nvidia_smi_output(monkeypatch):
    class Result:
        returncode = 0
        stdout = (
            "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-a)\n"
            "GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-b)\n"
        )

    monkeypatch.setattr(run_tests.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert run_tests.get_gpu_count() == 2


def test_get_gpu_count_falls_back_to_cuda_visible_devices(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(run_tests.subprocess, "run", raise_missing)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,,4")
    assert run_tests.get_gpu_count() == 3

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "none")
    assert run_tests.get_gpu_count() == 0
