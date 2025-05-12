import jax
import jax.numpy as jnp
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Any, Callable
from flax import jax_utils
from flax.training import train_state
import numpy as np
from dataclasses import dataclass
import threading
import queue
import mmap
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class JAXDatasetConfig:
    batch_size: int = 32
    num_workers: int = 4
    prefetch_size: int = 2
    cache_size: int = 1000
    shuffle_buffer_size: int = 10000
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

class JAXDataPrefetcher:
    def __init__(self, loader: 'JAXDatasetLoader', config: JAXDatasetConfig):
        self.loader = loader
        self.config = config
        self._queue = queue.Queue(maxsize=config.prefetch_size)
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()
        
    def _prefetch_worker(self):
        while True:
            try:
                batch = self.loader._get_next_batch()
                self._queue.put(batch)
            except Exception as e:
                print(f"Prefetch error: {e}")
                
    def get_next(self) -> Dict:
        return self._queue.get()

class JAXDatasetLoader:
    def __init__(self, 
                 data_dir: str,
                 config: Optional[JAXDatasetConfig] = None,
                 transform: Optional[Callable] = None,
                 key: Optional[jax.random.PRNGKey] = None):
        self.data_dir = Path(data_dir)
        self.config = config or JAXDatasetConfig()
        self.transform = transform
        self.key = key or jax.random.PRNGKey(0)
        
        self.h5_files = list(self.data_dir.glob("**/*.h5"))
        self._file_cache = {}
        self._cache_lock = threading.Lock()
        self._current_idx = 0
        self._shuffle_buffer = []
        self._shuffle_buffer_lock = threading.Lock()
        
        self._init_shuffle_buffer()
        
    def _init_shuffle_buffer(self):
        with self._shuffle_buffer_lock:
            self._shuffle_buffer = list(range(len(self.h5_files)))
            self.key, subkey = jax.random.split(self.key)
            self._shuffle_buffer = jax.random.permutation(subkey, np.array(self._shuffle_buffer))
            
    def _load_and_cache(self, idx: int) -> Dict:
        with MemoryMappedFile(self.h5_files[idx]) as mmf:
            data = {
                'belief': jnp.array(mmf._h5['belief'][:]),
                'map_slice': jnp.array(mmf._h5['map_slice'][:]),
                'goal_mask': jnp.array(mmf._h5['goal_mask'][:]),
                'sensor_flag': jnp.array(mmf._h5['sensor_flag'][:]),
                'traj': jnp.array(mmf._h5['traj'][:]),
                'sigma': jnp.array(mmf._h5['sigma'][:]),
                'meta': mmf._h5['meta'][()]
            }
            
        with self._cache_lock:
            self._file_cache[idx] = data
            if len(self._file_cache) > self.config.cache_size:
                oldest_key = next(iter(self._file_cache))
                del self._file_cache[oldest_key]
                
        return data
        
    def _get_next_batch(self) -> Dict:
        batch_indices = []
        with self._shuffle_buffer_lock:
            for _ in range(self.config.batch_size):
                if self._current_idx >= len(self._shuffle_buffer):
                    self._init_shuffle_buffer()
                    self._current_idx = 0
                batch_indices.append(self._shuffle_buffer[self._current_idx])
                self._current_idx += 1
                
        batch_data = []
        for idx in batch_indices:
            with self._cache_lock:
                if idx in self._file_cache:
                    data = self._file_cache[idx]
                else:
                    data = self._load_and_cache(idx)
                    
            if self.transform and np.random.random() < self.config.transform_prob:
                data = self.transform(data)
                
            batch_data.append(data)
            
        return jax.tree_map(lambda *x: jnp.stack(x), *batch_data)
        
    def get_prefetcher(self) -> JAXDataPrefetcher:
        return JAXDataPrefetcher(self, self.config)
        
    def get_batch(self) -> Dict:
        return self._get_next_batch()
        
    def get_batched_dataset(self) -> Dict:
        return jax_utils.prefetch_to_device(
            self.get_batch,
            size=self.config.prefetch_size,
            devices=jax.local_devices()
        ) 