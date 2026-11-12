import torch
from transformers.utils import is_torch_npu_available


def is_npu_available() -> bool:
    """Detect Ascend NPU availability"""
    try:
        return is_torch_npu_available()
    except (ImportError, RuntimeError, ModuleNotFoundError, AttributeError):
        return False


def get_current_device() -> str:
    """Get the current accelerator device string (e.g. 'cuda:0', 'npu:0')."""
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return "cuda:0"
    device_idx = torch.accelerator.current_device_index()
    return f"{acc.type}:{device_idx}"


def get_device_name(idx: int) -> str:
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return "NO ACCELERATOR"

    mod = torch.get_device_module(acc)
    if hasattr(mod, "get_device_name"):
        return mod.get_device_name(idx)
    else:
        return f"{str(acc).upper()} DEVICE"


def mem_get_info():
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return (0, 0)

    mod = torch.get_device_module(acc)
    if hasattr(mod, "mem_get_info"):
        return mod.mem_get_info()
    else:
        return (0, 0)


def empty_cache():
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return

    mod = torch.get_device_module(acc)
    if hasattr(mod, "empty_cache"):
        mod.empty_cache()
    return


def accelerator_synchronize() -> None:
    """torch.accelerator.synchronize() routed through the device module."""
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return

    mod = torch.get_device_module(acc)
    if hasattr(mod, "synchronize"):
        mod.synchronize()


def accelerator_memory_allocated() -> int:
    """torch.accelerator.memory_allocated() with an NPU-safe dispatch.

    torch_npu's allocator does not implement the DeviceAllocator interface
    that torch.accelerator.memory_* asserts on ("Allocator for npu is not a
    DeviceAllocator" INTERNAL ASSERT FAILED at CachingDeviceAllocator.h), so
    route through the device module (torch.npu on NPU) instead.
    """
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return 0

    mod = torch.get_device_module(acc)
    if hasattr(mod, "memory_allocated"):
        return mod.memory_allocated()
    return 0
