import pytest
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("aios_process", "os/core/aios_process.py")
aios_process = importlib.util.module_from_spec(spec)
sys.modules["aios_process"] = aios_process
spec.loader.exec_module(aios_process)

def test_mem_copy_overlap_normal():
    dst = bytearray(10)
    src = b"0123456789"
    aios_process.mem_copy_overlap(dst, 0, src, 0, 5)
    assert dst == b"01234\x00\x00\x00\x00\x00"

def test_mem_copy_overlap_zero_size():
    dst = bytearray(10)
    src = b"0123456789"
    aios_process.mem_copy_overlap(dst, 0, src, 0, 0)
    assert dst == b"\x00" * 10

def test_mem_copy_overlap_negative_size():
    dst = bytearray(10)
    src = b"0123456789"
    aios_process.mem_copy_overlap(dst, 0, src, 0, -1)
    assert dst == b"\x00" * 10

def test_mem_copy_overlap_too_large():
    dst = bytearray(100)
    src = b"0" * 100
    with pytest.raises(ValueError, match="> 64B limit"):
        aios_process.mem_copy_overlap(dst, 0, src, 0, 65)

def test_mem_copy_overlap_same_array_forward():
    arr = bytearray(b"0123456789")
    aios_process.mem_copy_overlap(arr, 2, arr, 0, 5)
    assert arr == b"0101234789"

def test_mem_copy_overlap_same_array_backward():
    arr = bytearray(b"0123456789")
    aios_process.mem_copy_overlap(arr, 0, arr, 2, 5)
    assert arr == b"2345656789"
