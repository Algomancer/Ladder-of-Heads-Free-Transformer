"""
text8_loader.py
Text8 DataLoader with jagged batch support.

Mirrors the image loader design:
- JaggedBatch structure with offsets for variable-length sequences
- Fixed sequence length mode (like max_patches)
- Simple character-level tokenization (27 vocab: a-z + space)

BPC (bits per character) = cross_entropy_loss / ln(2)
"""

import torch
from typing import Iterator, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import zipfile
import urllib.request


# Text8 character set: 'a'-'z' + ' ' (space)
VOCAB_SIZE = 27
CHAR_TO_IDX = {chr(ord('a') + i): i for i in range(26)}
CHAR_TO_IDX[' '] = 26
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}


def download_text8(data_dir: str = "./data") -> Path:
    """Download text8 dataset if not present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    text8_path = data_dir / "text8"
    zip_path = data_dir / "text8.zip"
    
    if text8_path.exists():
        return text8_path
    
    if not zip_path.exists():
        print("Downloading text8...")
        url = "http://mattmahoney.net/dc/text8.zip"
        urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting text8...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    
    return text8_path


def load_text8(data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load text8 and return train/valid/test splits as uint8 arrays.
    
    Standard split:
    - Train: first 90M characters
    - Valid: next 5M characters  
    - Test: last 5M characters
    
    Returns:
        train, valid, test: uint8 numpy arrays of character indices
    """
    text8_path = download_text8(data_dir)
    
    with open(text8_path, 'r') as f:
        text = f.read()
    
    # Convert to indices
    indices = np.array([CHAR_TO_IDX[c] for c in text], dtype=np.uint8)
    
    # Standard splits
    train = indices[:90_000_000]
    valid = indices[90_000_000:95_000_000]
    test = indices[95_000_000:100_000_000]
    
    return train, valid, test


@dataclass
class TextBatch:
    """A batch of text sequences packed together."""
    tokens: torch.Tensor       # (total_tokens,) input tokens
    targets: torch.Tensor      # (total_tokens,) target tokens (shifted by 1)
    offsets: torch.Tensor      # (batch_size + 1,) cumulative offsets
    coords: torch.Tensor       # (total_tokens, 2) grid coordinates (0, position) for ND-RoPE compat
    
    @property
    def batch_size(self) -> int:
        return len(self.offsets) - 1
    
    @property
    def total_tokens(self) -> int:
        return self.tokens.shape[0]
    
    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "TextBatch":
        return TextBatch(
            tokens=self.tokens.to(device),
            targets=self.targets.to(device),
            offsets=self.offsets.to(device),
            coords=self.coords.to(device),
        )
    
    def decode(self, idx: int = 0) -> str:
        """Decode a single sequence from the batch."""
        start = self.offsets[idx].item()
        end = self.offsets[idx + 1].item()
        tokens = self.tokens[start:end].tolist()
        return ''.join(IDX_TO_CHAR[t] for t in tokens)


class Text8Dataset:
    """Simple indexable dataset for text8 chunks."""
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 256,
        stride: Optional[int] = None,
    ):
        """
        Args:
            data: uint8 array of character indices
            seq_len: length of each sequence (need seq_len + 1 for target)
            stride: step between sequences (default: seq_len for non-overlapping)
        """
        self.data = data
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        
        # Number of complete sequences we can extract
        # Need seq_len + 1 chars for input + target
        self.n_sequences = max(0, (len(data) - seq_len - 1) // self.stride + 1)
    
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            tokens: (seq_len,) uint8 input tokens
            targets: (seq_len,) uint8 target tokens
        """
        start = idx * self.stride
        end = start + self.seq_len + 1
        
        chunk = self.data[start:end]
        tokens = chunk[:-1]
        targets = chunk[1:]
        
        return tokens, targets


def collate_text_batch(items: List[Tuple[np.ndarray, np.ndarray]]) -> TextBatch:
    """
    Collate text sequences into a jagged batch.
    
    Args:
        items: List of (tokens, targets) numpy array tuples
        
    Returns:
        TextBatch with torch tensors
    """
    if not items:
        return TextBatch(
            tokens=torch.empty(0, dtype=torch.long),
            targets=torch.empty(0, dtype=torch.long),
            offsets=torch.zeros(1, dtype=torch.long),
            coords=torch.empty(0, 2, dtype=torch.long),
        )
    
    # Pre-compute sizes
    lengths = [t.shape[0] for t, _ in items]
    total_tokens = sum(lengths)
    
    # Allocate
    all_tokens = np.empty(total_tokens, dtype=np.int64)
    all_targets = np.empty(total_tokens, dtype=np.int64)
    all_coords = np.empty((total_tokens, 2), dtype=np.int64)
    
    # Fill
    offset = 0
    for tokens, targets in items:
        n = tokens.shape[0]
        all_tokens[offset:offset + n] = tokens
        all_targets[offset:offset + n] = targets
        # coords: (0, position) for 1D sequences - compatible with 2D ND-RoPE
        all_coords[offset:offset + n, 0] = 0
        all_coords[offset:offset + n, 1] = np.arange(n)
        offset += n
    
    # Build offsets
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    
    return TextBatch(
        tokens=torch.from_numpy(all_tokens),
        targets=torch.from_numpy(all_targets),
        offsets=torch.from_numpy(offsets),
        coords=torch.from_numpy(all_coords),
    )


class Text8Loader:
    """
    DataLoader for text8 with jagged batch support.
    
    Similar to SPDLImageLoader - prefetches batches in background thread.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 256,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
        prefetch_factor: int = 4,
        stride: Optional[int] = None,
    ):
        """
        Args:
            data: uint8 array of character indices
            seq_len: sequence length
            batch_size: number of sequences per batch
            shuffle: shuffle sequences each epoch
            drop_last: drop incomplete final batch
            prefetch_factor: number of batches to prefetch
            stride: step between sequences (default: seq_len)
        """
        self.dataset = Text8Dataset(data, seq_len=seq_len, stride=stride)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
    
    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
    
    def _get_indices(self) -> List[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        return indices
    
    def __iter__(self) -> Iterator[TextBatch]:
        from queue import Queue
        import threading
        
        indices = self._get_indices()
        
        if self.drop_last:
            n_batches = len(indices) // self.batch_size
            indices = indices[:n_batches * self.batch_size]
        
        prefetch_queue: Queue[Optional[TextBatch]] = Queue(maxsize=self.prefetch_factor)
        stop_event = threading.Event()
        
        def prefetch_worker():
            try:
                for batch_start in range(0, len(indices), self.batch_size):
                    if stop_event.is_set():
                        break
                    
                    batch_indices = indices[batch_start:batch_start + self.batch_size]
                    items = [self.dataset[idx] for idx in batch_indices]
                    batch = collate_text_batch(items)
                    prefetch_queue.put(batch)
                
                prefetch_queue.put(None)
            except Exception as e:
                prefetch_queue.put(None)
                raise
        
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()
        
        try:
            while True:
                batch = prefetch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            while not prefetch_queue.empty():
                try:
                    prefetch_queue.get_nowait()
                except:
                    pass


class Text8FixedLoader:
    """
    Loader that yields batches with exactly max_tokens tokens.
    
    Uses rolling buffer - sequences that don't fit carry to next batch.
    Mirrors create_fixed_loader for images.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        max_tokens: int,
        seq_len: int = 256,
        batch_size: int = 64,
        shuffle: bool = True,
        prefetch_factor: int = 4,
    ):
        self.base_loader = Text8Loader(
            data=data,
            seq_len=seq_len,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            prefetch_factor=prefetch_factor,
        )
        self.max_tokens = max_tokens
    
    def __iter__(self) -> Iterator[TextBatch]:
        from collections import deque
        
        buffer_tokens: deque[torch.Tensor] = deque()
        buffer_targets: deque[torch.Tensor] = deque()
        buffer_coords: deque[torch.Tensor] = deque()
        buffer_count: int = 0
        
        def emit_batch() -> TextBatch:
            nonlocal buffer_count
            
            out_tokens = []
            out_targets = []
            out_coords = []
            out_offsets = [0]
            used = 0
            
            while buffer_tokens and used < self.max_tokens:
                seq_tokens = buffer_tokens[0]
                seq_targets = buffer_targets[0]
                seq_coords = buffer_coords[0]
                seq_len = seq_tokens.shape[0]
                
                if used + seq_len <= self.max_tokens:
                    out_tokens.append(seq_tokens)
                    out_targets.append(seq_targets)
                    out_coords.append(seq_coords)
                    used += seq_len
                    out_offsets.append(used)
                    buffer_tokens.popleft()
                    buffer_targets.popleft()
                    buffer_coords.popleft()
                    buffer_count -= seq_len
                else:
                    take = self.max_tokens - used
                    out_tokens.append(seq_tokens[:take])
                    out_targets.append(seq_targets[:take])
                    out_coords.append(seq_coords[:take])
                    used += take
                    out_offsets.append(used)
                    buffer_tokens[0] = seq_tokens[take:]
                    buffer_targets[0] = seq_targets[take:]
                    buffer_coords[0] = seq_coords[take:]
                    buffer_count -= take
            
            return TextBatch(
                tokens=torch.cat(out_tokens),
                targets=torch.cat(out_targets),
                offsets=torch.tensor(out_offsets, dtype=torch.long),
                coords=torch.cat(out_coords),
            )
        
        while True:
            for batch in self.base_loader:
                # Add sequences to buffer
                for i in range(batch.batch_size):
                    start = batch.offsets[i].item()
                    end = batch.offsets[i + 1].item()
                    buffer_tokens.append(batch.tokens[start:end])
                    buffer_targets.append(batch.targets[start:end])
                    buffer_coords.append(batch.coords[start:end])
                    buffer_count += end - start
                
                # Emit fixed batches
                while buffer_count >= self.max_tokens:
                    yield emit_batch()


def create_loaders(
    seq_len: int = 256,
    batch_size: int = 64,
    data_dir: str = "./data",
    shuffle_train: bool = True,
) -> Tuple[Text8Loader, Text8Loader, Text8Loader]:
    """
    Create train/valid/test loaders for text8.
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    train_data, valid_data, test_data = load_text8(data_dir)
    
    train_loader = Text8Loader(
        train_data, seq_len=seq_len, batch_size=batch_size,
        shuffle=shuffle_train, drop_last=True
    )
    valid_loader = Text8Loader(
        valid_data, seq_len=seq_len, batch_size=batch_size,
        shuffle=False, drop_last=False
    )
    test_loader = Text8Loader(
        test_data, seq_len=seq_len, batch_size=batch_size,
        shuffle=False, drop_last=False
    )
    
    return train_loader, valid_loader, test_loader


def bpc_from_loss(loss: float) -> float:
    """Convert cross-entropy loss (nats) to bits per character."""
    return loss / np.log(2)


def loss_from_bpc(bpc: float) -> float:
    """Convert bits per character to cross-entropy loss (nats)."""
    return bpc * np.log(2)

    
    print("\nDone!")
