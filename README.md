# B-COD Dataset

This repository contains an anonymized subset of a marine navigation dataset, providing multimodal sensor data for the B-COD project.

Please download the dataset from [this Google Drive link](https://drive.google.com/drive/folders/1-hn1hIVhsf1EiL8lQO-WpRQ42gzn73id?usp=sharing).

## Original Dataset Structure

Each data snippet contains:
- Belief raster (64×64×5 float16)
- Map slice (64×64×3 semantic image)
- Goal mask (64×64 binary)
- Sensor flags (5-bit uint8 vector)
- Ground truth trajectory (∆x,∆y,∆ψ for 8 waypoints)
- Waypoint log-variances
- Metadata (latitude/longitude, weather, clip ID)

## Dataset Statistics

- **Modalities**: 100% contain LiDAR and IMU data
  - Day camera: 72%
  - Night camera: 28%
  - GNSS: 64%
  - Sonde: 18%

- **Belief spread**: 
  - Median planar 1σ = 0.38m
  - 95th percentile = 2.1m

- **Lighting conditions**: 0.2-55 kLux
- **Obstacle ranges**: 
  - Mean: 14.2m
  - Minimum: 0.8m

## Current Publicly Available Dataset Directory Structure (Please note that this is a subset of the original dataset to maintain anonymity of the review process)

```
.
├── dataset_20250226_150909/
│   └── zEeZKgHmazs1R6C.h5
├── dataset_20250228_063657/
│   └── sA0JsWGpDDFMUMK.h5
...
└── dataset_lidar_json_untagged/
    └── *.json
```

## Download

To download the dataset, please follow this [this Google Drive link](https://drive.google.com/drive/folders/1-hn1hIVhsf1EiL8lQO-WpRQ42gzn73id?usp=sharing).

## Usage

```python
# PyTorch
from src.pytorch_loader import DatasetLoader
dataset = DatasetLoader("path/to/dataset")

# JAX
from src.jax_loader import DatasetLoader
loader = DatasetLoader("path/to/dataset", batch_size=32)
batch = loader.get_batch(0)
```

## License

License will be released after the review process. We plan to release the whole dataset (280GB) under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
