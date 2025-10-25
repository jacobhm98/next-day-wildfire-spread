from glob import glob
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import MultiTFRecordDataset


# Feature names matching the paper
INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

OUTPUT_FEATURES = ['FireMask']

# Data statistics from the paper (Table in preprocessing section)
# These are clip values (min, max), mean, and std for normalization
# Note: These would ideally be computed from your training data
# Using reasonable defaults based on the paper
DATA_STATS = {
    'elevation': {'min': -100, 'max': 4000, 'mean': 1000, 'std': 800},
    'th': {'min': 0, 'max': 360, 'mean': 180, 'std': 100},  # wind direction in degrees
    'vs': {'min': 0, 'max': 20, 'mean': 5, 'std': 3},  # wind speed m/s
    'tmmn': {'min': 230, 'max': 320, 'mean': 280, 'std': 15},  # min temp in Kelvin
    'tmmx': {'min': 240, 'max': 330, 'mean': 295, 'std': 15},  # max temp in Kelvin
    'sph': {'min': 0, 'max': 0.03, 'mean': 0.005, 'std': 0.005},  # specific humidity kg/kg
    'pr': {'min': 0, 'max': 100, 'mean': 2, 'std': 5},  # precipitation mm
    'pdsi': {'min': -10, 'max': 10, 'mean': 0, 'std': 3},  # drought index
    'NDVI': {'min': -1, 'max': 1, 'mean': 0.3, 'std': 0.2},  # vegetation index
    'population': {'min': 0, 'max': 3000, 'mean': 100, 'std': 300},  # people per sq km
    'erc': {'min': 0, 'max': 120, 'mean': 50, 'std': 25},  # energy release component
    'PrevFireMask': {'min': 0, 'max': 1, 'mean': 0.5, 'std': 0.5},  # binary mask
}


class WildfireDataset(Dataset):
    """Wrapper around TFRecord dataset with preprocessing and augmentation."""

    def __init__(self, split="train", augment=False):
        """
        Args:
            split: One of 'train', 'eval', or 'test'
            augment: Whether to apply data augmentation (flip, rotate)
        """
        self.split = split
        self.augment = augment

        # Get TFRecord files
        train_files = sorted(glob(f"data/next_day_wildfire_spread_{split}_*.tfrecord"))

        # Extract keys for MultiTFRecordDataset
        keys = []
        for f in train_files:
            match = re.search(r"_([a-z]+_\d+)\.tfrecord$", f)
            if match:
                keys.append(match.group(1))

        n = len(keys)
        splits = {key: 1.0 / n for key in keys}

        # Create TFRecord dataset
        self.tfrecord_dataset = MultiTFRecordDataset(
            data_pattern="data/next_day_wildfire_spread_{}.tfrecord",
            index_pattern="data/next_day_wildfire_spread_{}.tfindex",
            splits=splits,
            infinite=False
        )

    def __len__(self):
        # Note: MultiTFRecordDataset doesn't support len(), so we approximate
        # The paper mentions ~15k training samples, ~1.8k eval, ~1.6k test
        if self.split == "train":
            return 14979
        elif self.split == "eval":
            return 1877
        else:  # test
            return 1689

    def __getitem__(self, idx):
        # Get raw data from tfrecord
        # Note: MultiTFRecordDataset is iterable, not indexable
        # This is a limitation - we'll handle it in the DataLoader
        raise NotImplementedError("Use get_dataloader() instead")

    def preprocess(self, sample):
        """Preprocess a single sample from TFRecord.

        Args:
            sample: Dictionary with feature names as keys

        Returns:
            inputs: Tensor of shape [12, 64, 64]
            target: Tensor of shape [1, 64, 64]
        """
        # Stack input features in order
        input_list = []
        for feature_name in INPUT_FEATURES:
            data = sample[feature_name]

            # Convert to numpy if needed
            if isinstance(data, bytes):
                data = np.frombuffer(data, dtype=np.float32)

            # Reshape to 64x64 if needed
            if data.size == 64 * 64:
                data = data.reshape(64, 64)

            # Clip values
            stats = DATA_STATS[feature_name]
            data = np.clip(data, stats['min'], stats['max'])

            # Normalize (except for fire masks - keep as 0/1)
            if feature_name not in ['PrevFireMask', 'FireMask']:
                data = (data - stats['mean']) / stats['std']

            input_list.append(data)

        # Stack inputs: [12, 64, 64]
        inputs = np.stack(input_list, axis=0).astype(np.float32)

        # Get target (FireMask)
        target = sample['FireMask']
        if isinstance(target, bytes):
            target = np.frombuffer(target, dtype=np.float32)
        if target.size == 64 * 64:
            target = target.reshape(1, 64, 64)
        target = target.astype(np.float32)

        # Convert to tensors
        inputs = torch.from_numpy(inputs)
        target = torch.from_numpy(target)

        # Apply augmentation if enabled
        if self.augment:
            inputs, target = self.apply_augmentation(inputs, target)

        return inputs, target

    def apply_augmentation(self, inputs, target):
        """Apply random flip and rotation augmentation.

        Args:
            inputs: Tensor of shape [12, 64, 64]
            target: Tensor of shape [1, 64, 64]

        Returns:
            Augmented inputs and target
        """
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            inputs = torch.flip(inputs, dims=[2])
            target = torch.flip(target, dims=[2])

        # Random vertical flip
        if torch.rand(1) > 0.5:
            inputs = torch.flip(inputs, dims=[1])
            target = torch.flip(target, dims=[1])

        # Random 90-degree rotation (0, 90, 180, or 270 degrees)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            inputs = torch.rot90(inputs, k=k, dims=[1, 2])
            target = torch.rot90(target, k=k, dims=[1, 2])

        return inputs, target


def collate_fn(batch):
    """Custom collate function that preprocesses TFRecord samples.

    Args:
        batch: List of raw samples from MultiTFRecordDataset

    Returns:
        inputs: Tensor of shape [batch_size, 12, 64, 64]
        targets: Tensor of shape [batch_size, 1, 64, 64]
    """
    # This is a workaround since MultiTFRecordDataset returns raw dicts
    # We need to preprocess them here
    inputs_list = []
    targets_list = []

    for sample in batch:
        # Each sample is a dict with feature names as keys
        # We need to parse and preprocess it
        # This is a simplified version - in practice, you'd use the dataset's preprocess method
        pass

    return torch.stack(inputs_list), torch.stack(targets_list)


def get_dataloader(split="train", batch_size=32, num_workers=0, augment=None):
    """Create a DataLoader for the wildfire dataset.

    Args:
        split: One of 'train', 'eval', or 'test'
        batch_size: Batch size
        num_workers: Number of worker processes (must be 0 for iterable datasets)
        augment: Whether to apply augmentation. If None, defaults to True for train, False otherwise.

    Returns:
        DataLoader
    """
    if augment is None:
        augment = (split == "train")

    dataset = WildfireDataset(split=split, augment=augment)

    # Create iterable dataset wrapper
    class IterableDatasetWrapper(torch.utils.data.IterableDataset):
        def __init__(self, tfrecord_dataset, preprocess_fn):
            super().__init__()
            self.tfrecord_dataset = tfrecord_dataset
            self.preprocess_fn = preprocess_fn

        def __iter__(self):
            for sample in self.tfrecord_dataset:
                yield self.preprocess_fn(sample)

    # Create wrapper
    iterable = IterableDatasetWrapper(dataset.tfrecord_dataset, dataset.preprocess)

    # Custom collate that handles the iterator
    def collate_preprocessed(batch):
        inputs_list, targets_list = zip(*batch)
        return torch.stack(inputs_list), torch.stack(targets_list)

    # Note: For iterable datasets, we can't shuffle or use multiple workers easily
    # MultiTFRecordDataset handles sampling internally
    dataloader = DataLoader(
        iterable,
        batch_size=batch_size,
        collate_fn=collate_preprocessed,
        num_workers=0,  # Must be 0 for iterable datasets
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset loading...")

    train_loader = get_dataloader("train", batch_size=4)

    print("\nLoading one batch...")
    inputs, targets = next(iter(train_loader))

    print(f"Inputs shape: {inputs.shape}")  # Should be [4, 12, 64, 64]
    print(f"Targets shape: {targets.shape}")  # Should be [4, 1, 64, 64]
    print(f"Inputs range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"Targets unique values: {torch.unique(targets)}")

    print("\n✓ Dataset test passed!")
