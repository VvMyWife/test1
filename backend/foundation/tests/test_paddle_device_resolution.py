from __future__ import annotations

import sys
import types

from platform_foundation.inference import paddle_table


class _FakePaddle(types.ModuleType):
    def __init__(self, *, compiled_with_cuda: bool, device_count: int) -> None:
        super().__init__("paddle")
        self.__version__ = "3.1.1"
        self._compiled_with_cuda = compiled_with_cuda
        self.device = types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=lambda: device_count)
        )

    def is_compiled_with_cuda(self) -> bool:
        return self._compiled_with_cuda


def test_resolve_paddle_device_prefers_explicit_option(monkeypatch) -> None:
    monkeypatch.setenv("MINERU_PADDLE_DEVICE", "gpu:3")
    monkeypatch.setattr(
        paddle_table,
        "_auto_select_paddle_gpu",
        lambda: (_raise_assertion("auto selection should not run"), {}),
    )

    assert paddle_table._resolve_paddle_device({"paddle_device": "gpu:7"}) == "gpu:7"


def test_resolve_paddle_device_prefers_env_over_auto(monkeypatch) -> None:
    monkeypatch.setenv("MINERU_PADDLE_DEVICE", "gpu:5")
    monkeypatch.setattr(
        paddle_table,
        "_auto_select_paddle_gpu",
        lambda: (_raise_assertion("auto selection should not run"), {}),
    )

    assert paddle_table._resolve_paddle_device({}) == "gpu:5"


def test_auto_select_paddle_gpu_prefers_highest_free_memory(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        paddle_table,
        "_run_nvidia_smi_gpu_query",
        lambda: "\n".join(
            [
                "0, GPU-000, 24576, 12000",
                "1, GPU-111, 24576, 4000",
                "2, GPU-222, 24576, 10000",
            ]
        ),
    )
    monkeypatch.setattr(paddle_table, "_run_nvidia_smi_process_query", lambda: "")

    device, details = paddle_table._auto_select_paddle_gpu()

    assert device == "gpu:1"
    assert [item["host_index"] for item in details["candidates"]] == [1, 2, 0]


def test_auto_select_paddle_gpu_breaks_ties_by_process_count(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        paddle_table,
        "_run_nvidia_smi_gpu_query",
        lambda: "\n".join(
            [
                "0, GPU-000, 24576, 4096",
                "1, GPU-111, 24576, 4096",
            ]
        ),
    )
    monkeypatch.setattr(
        paddle_table,
        "_run_nvidia_smi_process_query",
        lambda: "\n".join(
            [
                "GPU-000, 1234",
                "GPU-000, 1235",
                "GPU-111, 2234",
            ]
        ),
    )

    device, details = paddle_table._auto_select_paddle_gpu()

    assert device == "gpu:1"
    assert [item["process_count"] for item in details["candidates"]] == [1, 2]


def test_auto_select_paddle_gpu_uses_visible_gpu_mapping(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setattr(
        paddle_table,
        "_run_nvidia_smi_gpu_query",
        lambda: "\n".join(
            [
                "0, GPU-000, 24576, 1024",
                "1, GPU-111, 24576, 2048",
                "3, GPU-333, 24576, 4096",
            ]
        ),
    )
    monkeypatch.setattr(paddle_table, "_run_nvidia_smi_process_query", lambda: "")

    device, details = paddle_table._auto_select_paddle_gpu()

    assert device == "gpu:1"
    assert [(item["host_index"], item["local_index"]) for item in details["candidates"]] == [
        (1, 1),
        (3, 0),
    ]


def test_resolve_paddle_device_falls_back_when_auto_selection_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("MINERU_PADDLE_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "paddle",
        _FakePaddle(compiled_with_cuda=True, device_count=2),
    )
    monkeypatch.setattr(
        paddle_table,
        "_auto_select_paddle_gpu",
        lambda: (None, {"reason": "nvidia-smi unavailable"}),
    )

    assert paddle_table._resolve_paddle_device({}) is None


def test_parse_nvidia_smi_gpu_query_output_skips_invalid_lines() -> None:
    records = paddle_table._parse_nvidia_smi_gpu_query_output(
        "\n".join(
            [
                "",
                "0, GPU-000, 24576, 1024",
                "broken line",
                "1, GPU-111, bad, 512",
                "2, GPU-222, 16384, 2048",
            ]
        )
    )

    assert [(item.host_index, item.free_memory_mib) for item in records] == [
        (0, 23552),
        (2, 14336),
    ]


def test_build_ppstructure_v3_uses_resolved_device(monkeypatch) -> None:
    fake_paddleocr = types.ModuleType("paddleocr")

    class FakePPStructureV3:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_paddleocr.PPStructureV3 = FakePPStructureV3
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)
    monkeypatch.setattr(paddle_table, "_resolve_paddle_device", lambda options: "gpu:4")

    model = paddle_table._build_ppstructure_v3({"paddle_table_disable_pipeline_cache": True})

    assert model.kwargs["device"] == "gpu:4"


def test_build_table_structure_model_uses_resolved_device(monkeypatch) -> None:
    fake_paddleocr = types.ModuleType("paddleocr")

    class FakeTableStructureRecognition:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    fake_paddleocr.TableStructureRecognition = FakeTableStructureRecognition
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)
    monkeypatch.setattr(paddle_table, "_resolve_paddle_device", lambda options: "gpu:2")

    model = paddle_table._build_table_structure_model({"paddle_table_disable_pipeline_cache": True})

    assert model.kwargs["device"] == "gpu:2"


def _raise_assertion(message: str) -> None:
    raise AssertionError(message)
