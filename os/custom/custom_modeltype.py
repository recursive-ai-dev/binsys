“””
USAGE
————————————————————————
 
from custom_format import save_file, load_file
 
# Save
tensors = {"weight": torch.randn(1000, 1000), "bias": torch.randn(1000)}
save_file(tensors, "model.ctf", metadata={"version": "1.0"})
 
# Load
loaded = load_file("model.ctf", device="cuda")
 
————————————————————————
 
from custom_format import safe_open
 
with safe_open("10gb_model.ctf") as f:
    # Only load the layers you need
    layer0 = f.get_tensor("layers.0.weight")
    layer1 = f.get_tensor("layers.1.weight")
    # 10GB file stays on disk, only these tensors in RAM
 
————————————————————————
 
from custom_format import CTFHeader
 
header = CTFHeader.from_file("model.ctf")
for name, info in header.tensors.items():
    print(f"{name}: {info.dtype} {info.shape}")
 
———————————————————————-
 
 
Custom Tensor Format (CTF) - A SafeTensors/GGUF-inspired model format
Features:
- Memory-mapped zero-copy loading
- JSON metadata header with 8-byte size prefix
- Contiguous binary tensor storage
- Security: No code execution, validation checks
- Support for partial/lazy loading
"""
 
import json
import struct
import os
import mmap
from typing import Dict, List, Tuple, Optional, Union, BinaryIO
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
 
 
@dataclass
class TensorInfo:
    """Metadata for a single tensor"""
    name: str
    dtype: str  # 'F32', 'F16', 'BF16', 'I32', 'I64', etc.
    shape: List[int]
    data_offsets: Tuple[int, int]  # (start, end) in bytes
    
    def to_dict(self) -> dict:
        return {
            "dtype": self.dtype,
            "shape": self.shape,
            "data_offsets": list(self.data_offsets)
        }
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> "TensorInfo":
        return cls(
            name=name,
            dtype=data["dtype"],
            shape=data["shape"],
            data_offsets=tuple(data["data_offsets"])
        )
    
    @property
    def nbytes(self) -> int:
        """Calculate size in bytes"""
        return self.data_offsets[1] - self.data_offsets[0]
 
 
class CTFHeader:
    """Manages the JSON header section"""
    
    MAX_HEADER_SIZE = 100 * 1024 * 1024  # 100MB limit (DOS protection)
    
    def __init__(self, tensors: Dict[str, TensorInfo], metadata: Optional[Dict[str, str]] = None):
        self.tensors = tensors
        self.metadata = metadata or {}
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        header_dict = {
            name: info.to_dict() 
            for name, info in self.tensors.items()
        }
        if self.metadata:
            header_dict["__metadata__"] = self.metadata
        
        header_json = json.dumps(header_dict, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8')
        
        if len(header_bytes) > self.MAX_HEADER_SIZE:
            raise ValueError(f"Header too large: {len(header_bytes)} bytes")
        
        # 8-byte little-endian size prefix
        size_prefix = struct.pack('<Q', len(header_bytes))
        return size_prefix + header_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "CTFHeader":
        """Parse header from bytes"""
        if len(data) < 8:
            raise ValueError("File too small for header")
        
        header_size = struct.unpack('<Q', data[:8])[0]
        
        if header_size > cls.MAX_HEADER_SIZE:
            raise ValueError(f"Header size {header_size} exceeds maximum")
        
        if len(data) < 8 + header_size:
            raise ValueError("File truncated")
        
        header_json = data[8:8+header_size].decode('utf-8')
        header_dict = json.loads(header_json)
        
        metadata = header_dict.pop("__metadata__", {})
        
        tensors = {}
        for name, info_dict in header_dict.items():
            tensors[name] = TensorInfo.from_dict(name, info_dict)
        
        return cls(tensors, metadata)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "CTFHeader":
        """Read only header from file (fast inspection)"""
        with open(filepath, 'rb') as f:
            # Read just the size prefix first
            size_prefix = f.read(8)
            if len(size_prefix) != 8:
                raise ValueError("Invalid file format")
            
            header_size = struct.unpack('<Q', size_prefix)[0]
            if header_size > cls.MAX_HEADER_SIZE:
                raise ValueError("Header too large")
            
            # Read the rest of header
            header_data = size_prefix + f.read(header_size)
            return cls.from_bytes(header_data)
 
 
class CTFSerializer:
    """Handles conversion between tensors and CTF format"""
    
    # Mapping between numpy/torch dtypes and our format
    DTYPE_MAP = {
        'float32': 'F32',
        'float16': 'F16',
        'bfloat16': 'BF16',
        'float64': 'F64',
        'int32': 'I32',
        'int64': 'I64',
        'int16': 'I16',
        'int8': 'I8',
        'uint8': 'U8',
        'bool': 'BOOL',
    }
    
    REVERSE_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}
    
    @classmethod
    def numpy_to_ctf_dtype(cls, arr: np.ndarray) -> str:
        dtype_str = str(arr.dtype)
        if dtype_str not in cls.DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return cls.DTYPE_MAP[dtype_str]
    
    @classmethod
    def ctf_to_numpy_dtype(cls, ctf_dtype: str) -> str:
        if ctf_dtype not in cls.REVERSE_DTYPE_MAP:
            raise ValueError(f"Unknown CTF dtype: {ctf_dtype}")
        return cls.REVERSE_DTYPE_MAP[ctf_dtype]
    
    @classmethod
    def save_file(
        cls, 
        tensors: Dict[str, Union[np.ndarray, torch.Tensor]], 
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Save tensors to CTF format file.
        
        Args:
            tensors: Dictionary mapping names to tensor objects
            filepath: Output file path
            metadata: Optional string metadata dict
        """
        filepath = Path(filepath)
        
        # Convert all to numpy and calculate offsets
        numpy_tensors = {}
        current_offset = 0
        tensor_infos = {}
        
        for name, tensor in tensors.items():
            # Convert to numpy if needed
            if isinstance(tensor, torch.Tensor):
                # Ensure contiguous and on CPU
                tensor = tensor.detach().cpu().contiguous().numpy()
            elif not isinstance(tensor, np.ndarray):
                tensor = np.array(tensor)
            
            # Ensure C-contiguous
            tensor = np.ascontiguousarray(tensor)
            
            dtype = cls.numpy_to_ctf_dtype(tensor)
            shape = list(tensor.shape)
            nbytes = tensor.nbytes
            
            numpy_tensors[name] = tensor
            tensor_infos[name] = TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                data_offsets=(current_offset, current_offset + nbytes)
            )
            current_offset += nbytes
        
        # Build header
        header = CTFHeader(tensor_infos, metadata)
        header_bytes = header.to_bytes()
        
        # Write file
        with open(filepath, 'wb') as f:
            f.write(header_bytes)
            
            # Write tensor data contiguously
            for name in tensor_infos:
                f.write(numpy_tensors[name].tobytes())
        
        return filepath
    
    @classmethod
    def load_file(
        cls, 
        filepath: Union[str, Path], 
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Load all tensors from CTF file.
        
        Args:
            filepath: Path to .ctf file
            device: Target device ('cpu', 'cuda', etc.)
        
        Returns:
            Dictionary of tensors
        """
        with CTFLoader(filepath) as loader:
            return {
                name: loader.get_tensor(name, device=device)
                for name in loader.tensor_names()
            }
 
 
class CTFLoader:
    """
    Memory-mapped loader for CTF files.
    Supports lazy loading and partial tensor access.
    """
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self._file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        self._header: Optional[CTFHeader] = None
        self._data_start: int = 0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def open(self):
        """Open file and memory map it"""
        if self._file is not None:
            return
        
        file_size = self.filepath.stat().st_size
        self._file = open(self.filepath, 'rb')
        
        # Memory map the entire file
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse header
        self._header = CTFHeader.from_bytes(self._mmap)
        
        # Calculate where data section starts
        header_size = struct.unpack('<Q', self._mmap[:8])[0]
        self._data_start = 8 + header_size
        
        # Validate file size matches expected
        last_tensor_end = max(
            (info.data_offsets[1] for info in self._header.tensors.values()),
            default=0
        )
        expected_size = self._data_start + last_tensor_end
        
        if file_size != expected_size:
            raise ValueError(
                f"File size mismatch: expected {expected_size}, got {file_size}. "
                "File may be corrupted or truncated."
            )
    
    def close(self):
        """Close file and release resources"""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._header = None
    
    def is_open(self) -> bool:
        return self._file is not None
    
    @property
    def header(self) -> CTFHeader:
        if self._header is None:
            raise RuntimeError("Loader not opened")
        return self._header
    
    def tensor_names(self) -> List[str]:
        """Get list of available tensor names"""
        return list(self.header.tensors.keys())
    
    def get_tensor_info(self, name: str) -> TensorInfo:
        """Get metadata for a specific tensor"""
        if name not in self.header.tensors:
            raise KeyError(f"Tensor '{name}' not found")
        return self.header.tensors[name]
    
    def get_tensor(
        self, 
        name: str, 
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Load a specific tensor by name.
        
        This uses zero-copy when possible (CPU), or copies to device.
        """
        info = self.get_tensor_info(name)
        
        # Calculate absolute offsets in file
        abs_start = self._data_start + info.data_offsets[0]
        abs_end = self._data_start + info.data_offsets[1]
        
        # Get raw bytes from mmap (zero-copy view)
        raw_bytes = self._mmap[abs_start:abs_end]
        
        # Convert to numpy array (zero-copy via buffer protocol)
        dtype = CTFSerializer.ctf_to_numpy_dtype(info.dtype)
        arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(info.shape)
        
        # Convert to torch tensor
        tensor = torch.from_numpy(arr)
        
        # Move to device if needed
        if device != "cpu":
            tensor = tensor.to(device)
        
        return tensor
    
    def get_slice(self, name: str, indices: Tuple[int, ...]) -> torch.Tensor:
        """
        Load a slice of a tensor efficiently without loading the whole tensor.
        Note: This performs a copy for the slice, but doesn't load full tensor.
        """
        info = self.get_tensor_info(name)
        
        # For now, implement simple row slicing for first dimension
        # Full ND slicing would require more complex stride calculations
        if len(indices) != len(info.shape):
            raise ValueError("Indices must match tensor dimensions")
        
        # Calculate byte offsets for the slice
        dtype = CTFSerializer.ctf_to_numpy_dtype(info.dtype)
        itemsize = np.dtype(dtype).itemsize
        
        # Simple implementation: slice first dimension only
        start_idx = indices[0] if isinstance(indices[0], slice) else slice(indices[0], indices[0]+1)
        
        if isinstance(start_idx, slice):
            row_start = start_idx.start or 0
            row_end = start_idx.stop or info.shape[0]
            row_step = start_idx.step or 1
            
            rows = list(range(row_start, row_end, row_step))
            if len(rows) == 0:
                raise ValueError("Empty slice")
            
            # Calculate bytes per row
            bytes_per_row = itemsize * np.prod(info.shape[1:], dtype=np.int64)
            
            sliced_data = bytearray()
            for row in rows:
                row_offset = info.data_offsets[0] + (row * bytes_per_row)
                abs_start = self._data_start + row_offset
                abs_end = abs_start + bytes_per_row
                sliced_data.extend(self._mmap[abs_start:abs_end])
            
            new_shape = [len(rows)] + info.shape[1:]
            arr = np.frombuffer(sliced_data, dtype=dtype).reshape(new_shape)
            return torch.from_numpy(arr)
        else:
            raise NotImplementedError("Only slicing supported currently")
    
    def get_metadata(self) -> Dict[str, str]:
        """Get file metadata"""
        return self.header.metadata.copy()
    
    def inspect(self) -> Dict:
        """Get full file inspection info"""
        return {
            "filepath": str(self.filepath),
            "num_tensors": len(self.header.tensors),
            "tensors": {
                name: {
                    "dtype": info.dtype,
                    "shape": info.shape,
                    "size_mb": info.nbytes / (1024 * 1024)
                }
                for name, info in self.header.tensors.items()
            },
            "metadata": self.header.metadata,
            "total_size_mb": (self._data_start + 
                            sum(t.nbytes for t in self.header.tensors.values())) / (1024 * 1024)
        }
 
 
# Convenience functions matching safetensors API
def save_file(
    tensors: Dict[str, Union[np.ndarray, torch.Tensor]], 
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None
):
    """Save tensors to CTF format (SafeTensors-compatible API)"""
    return CTFSerializer.save_file(tensors, filepath, metadata)
 
def load_file(
    filepath: Union[str, Path], 
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load all tensors from CTF file (SafeTensors-compatible API)"""
    return CTFSerializer.load_file(filepath, device)
 
def safe_open(filepath: Union[str, Path], framework: str = "pt"):
    """
    Context manager for lazy loading (SafeTensors-compatible API).
    
    Usage:
        with safe_open("model.ctf") as f:
            tensor = f.get_tensor("weight")
    """
    return CTFLoader(filepath)
 
 
# Example usage and testing
if __name__ == "__main__":
    # Create test tensors
    print("Creating test tensors...")
    tensors = {
        "embedding.weight": torch.randn(50000, 768, dtype=torch.float32),
        "transformer.layers.0.attention.qkv.weight": torch.randn(768, 2304, dtype=torch.float16),
        "transformer.layers.0.attention.qkv.bias": torch.randn(2304, dtype=torch.float32),
        "lm_head.weight": torch.randn(50000, 768, dtype=torch.bfloat16),
    }
    
    # Add some metadata
    metadata = {
        "model_type": "transformer",
        "num_layers": "12",
        "author": "custom_loader_demo"
    }
    
    filepath = "test_model.ctf"
    
    # Save
    print(f"\nSaving to {filepath}...")
    save_file(tensors, filepath, metadata)
    
    # Inspect without loading
    print("\nInspecting file header...")
    header = CTFHeader.from_file(filepath)
    print(f"Found {len(header.tensors)} tensors")
    for name, info in header.tensors.items():
        print(f"  {name}: {info.dtype} {info.shape}")
    
    # Lazy loading
    print("\nLazy loading specific tensors...")
    with safe_open(filepath) as loader:
        print(f"Available tensors: {loader.tensor_names()}")
        
        # Load only what we need
        emb = loader.get_tensor("embedding.weight")
        print(f"Loaded embedding: {emb.shape}, {emb.dtype}")
        
        # Check metadata
        print(f"Metadata: {loader.get_metadata()}")
        
        # Full inspection
        info = loader.inspect()
        print(f"\nTotal file size: {info['total_size_mb']:.2f} MB")
    
    # Full load
    print("\nLoading all tensors...")
    loaded = load_file(filepath, device="cpu")
    for name, tensor in loaded.items():
        print(f"  {name}: {tensor.shape}, {tensor.dtype}")
    
    # Verify correctness
    print("\nVerifying data integrity...")
    for name in tensors:
        original = tensors[name]
        loaded_t = loaded[name]
        
        # Convert to same dtype for comparison (bfloat16 might have precision differences)
        if original.dtype == torch.bfloat16:
            original = original.float()
            loaded_t = loaded_t.float()
        
        if torch.allclose(original, loaded_t):
            print(f"  ✓ {name}: OK")
        else:
            print(f"  ✗ {name}: MISMATCH")
    
    # Cleanup
    os.remove(filepath)
    print(f"\nCleaned up {filepath}")
    print("Demo complete!")
