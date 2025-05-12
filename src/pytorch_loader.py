import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Any
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import mmap
import threading
from dataclasses import dataclass
import queue
import time

@dataclass
class DatasetConfig:
    cache_size: int = 1000
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    transform_prob: float = 0.5

class MemoryMappedFile:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.Lock()
        self._mmap = None
        self._h5 = None
        
    def __enter__(self):
        self._lock.acquire()
        self._h5 = h5py.File(self.file_path, 'r')
        self._mmap = mmap.mmap(self._h5.fileno(), 0, access=mmap.ACCESS_READ)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mmap:
            self._mmap.close()
        if self._h5:
            self._h5.close()
        self._lock.release()

class DataPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.next_meta = None
        self._preload()
        
    def _preload(self):
        try:
            self.next_data, self.next_meta = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_meta = None
            return
            
        with torch.cuda.stream(self.stream):
            self.next_data = {k: v.to(self.device, non_blocking=True) 
                            for k, v in self.next_data.items()}
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        meta = self.next_meta
        if data is not None:
            self._preload()
        return data, meta

class DatasetLoader(Dataset):
    def __init__(self, 
                 data_dir: str,
                 config: Optional[DatasetConfig] = None,
                 transform: Optional[callable] = None,
                 device: Optional[torch.device] = None):
        self.data_dir = Path(data_dir)
        self.config = config or DatasetConfig()
        self.transform = transform
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.h5_files = list(self.data_dir.glob("**/*.h5"))
        self._file_cache = {}
        self._cache_lock = threading.Lock()
        self._prefetch_queue = queue.Queue(maxsize=self.config.cache_size)
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()
        
    def _prefetch_worker(self):
        while True:
            try:
                idx = self._prefetch_queue.get()
                if idx is None:
                    break
                self._load_and_cache(idx)
            except Exception as e:
                print(f"Prefetch error: {e}")
            finally:
                self._prefetch_queue.task_done()
                
    def _load_and_cache(self, idx: int):
        with MemoryMappedFile(self.h5_files[idx]) as mmf:
            data = {
                'belief': torch.from_numpy(mmf._h5['belief'][:]).float(),
                'map_slice': torch.from_numpy(mmf._h5['map_slice'][:]).float(),
                'goal_mask': torch.from_numpy(mmf._h5['goal_mask'][:]).float(),
                'sensor_flag': torch.from_numpy(mmf._h5['sensor_flag'][:]).long(),
                'traj': torch.from_numpy(mmf._h5['traj'][:]).float(),
                'sigma': torch.from_numpy(mmf._h5['sigma'][:]).float(),
                'meta': mmf._h5['meta'][()]
            }
            
        with self._cache_lock:
            self._file_cache[idx] = data
            if len(self._file_cache) > self.config.cache_size:
                oldest_key = next(iter(self._file_cache))
                del self._file_cache[oldest_key]
                
    def __len__(self) -> int:
        return len(self.h5_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with self._cache_lock:
            if idx in self._file_cache:
                data = self._file_cache[idx]
            else:
                self._load_and_cache(idx)
                data = self._file_cache[idx]
                
        if self.transform and np.random.random() < self.config.transform_prob:
            data = self.transform(data)
            
        return data
        
    def get_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers
        )
        
    def get_prefetcher(self, batch_size: int) -> DataPrefetcher:
        return DataPrefetcher(self.get_dataloader(batch_size), self.device) 