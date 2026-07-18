import logging

logger = logging.getLogger("device-manager")

import sys
_reconf = getattr(sys.stdout, "reconfigure", None)
if callable(_reconf):
    _reconf(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import openvino as ov
    OV_AVAILABLE = True
except Exception:
    OV_AVAILABLE = False

import psutil


class _CPUFallbackManager:
    def pick(self, model_type: str = "personal") -> str:
        return "CPU"

    @property
    def summary(self) -> dict:
        return {"available_devices": ["CPU"], "assigned": {}}

    @property
    def has_acceleration(self) -> bool:
        return False


if not OV_AVAILABLE:
    device_manager = _CPUFallbackManager()
else:

    class DeviceManager:
        _PREFERENCES = {
            "personal":  ["NPU", "GPU", "CPU"],
            "analysis":  ["NPU", "GPU", "CPU"],
            "embedding": ["NPU", "GPU", "CPU"],
        }

        def __init__(self):
            self._core = ov.Core()
            self._ram_gb = 0.0
            self._available = ["CPU"]
            self._npu_name = "N/A"
            self._npu_family = "N/A"
            self.refresh()

        def refresh(self):
            self._ram_gb = psutil.virtual_memory().total / 1024**3
            try:
                self._available = self._core.available_devices
            except Exception:
                self._available = ["CPU"]

            if "NPU" in self._available:
                try:
                    self._npu_name = self._core.get_property("NPU", "FULL_DEVICE_NAME")
                    self._npu_family = self._npu_name.lower()
                except Exception:
                    self._npu_name = "Intel NPU (generic)"
                    self._npu_family = "generic"
            else:
                self._npu_name = "N/A"
                self._npu_family = "N/A"

        def pick(self, model_type: str) -> str:
            self.refresh()
            base = self._PREFERENCES.get(model_type, ["CPU"])

            if model_type == "personal":
                if self._ram_gb >= 16:
                    chain = [d for d in base if d in self._available]
                elif self._ram_gb >= 12:
                    chain = [d for d in base if d in self._available and d != "NPU"]
                else:
                    chain = ["CPU"]
            elif model_type == "analysis":
                # INT4 LLM：Intel 核显（iGPU）+ INT4 已知会编译通过但推理挂死不输出
                # （参考 Intel 官方 KB: "Unable To Get Output With INT8/INT4 ... on GPU"）。
                # 仅当有 NPU 或 RAM>=12（可能有独显）时才用 GPU；否则强制 CPU 避免 iGPU 坑。
                if "NPU" in self._available:
                    chain = [d for d in base if d in self._available]
                elif self._ram_gb >= 12:
                    chain = [d for d in base if d in self._available]
                else:
                    chain = ["CPU"]
            else:
                # embedding (BGE INT8) — iGPU 通常 OK，保持原链
                chain = [d for d in base if d in self._available]

            if not chain:
                chain = ["CPU"]

            device_str = f"AUTO:{','.join(chain)}"
            logger.info(
                "[%s] RAM=%.0fGB NPU=%s → %s",
                model_type, self._ram_gb, self._npu_family, device_str,
            )
            return device_str

        @property
        def summary(self) -> dict:
            return {
                "system_ram_gb": round(self._ram_gb, 1),
                "available_devices": self._available,
                "npu": self._npu_name,
                "assigned": {mt: self.pick(mt) for mt in self._PREFERENCES},
            }

        @property
        def has_acceleration(self) -> bool:
            return any(d in self._available for d in ["GPU", "NPU"])

    try:
        device_manager = DeviceManager()
    except Exception:
        logger.warning("OpenVINO init failed, falling back to CPU.")
        device_manager = _CPUFallbackManager()
