from glob import glob
import re
import torch
from tfrecord.torch.dataset import MultiTFRecordDataset


DATA_PATTERN = "data/next_day_wildfire_spread_{}.tfrecord"
INDEX_PATTERN = "data/next_day_wildfire_spread_{}.tfindex"
BATCH_SIZE = 32

INPUT_FEATURES = ['elevation', 'th', 'vs',  'tmmn', 'tmmx', 'sph', 
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']

OUTPUT_FEATURES = ['FireMask', ]


def get_dataset(split="train"):
    """Create a TFRecord dataset for the given split (train/eval/test)."""
    train_files = sorted(glob(f"data/next_day_wildfire_spread_{split}_*.tfrecord"))

    keys = []
    for f in train_files:
        match = re.search(r"_([a-z]+_\d+)\.tfrecord$", f)
        if match:
            keys.append(match.group(1))

    n = len(keys)
    splits = {key: 1.0/n for key in keys}

    dataset = MultiTFRecordDataset(
        data_pattern=DATA_PATTERN,
        index_pattern=INDEX_PATTERN,
        splits=splits,
        infinite=False
    )

    return dataset


def get_data_loader(split="train"):
    dataset = get_dataset(split)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    train_data_loader = get_data_loader("train")
    data_point = next(iter(train_data_loader))
    for key, values in data_point.items():
        print(f"name: {key}, tensor shape: {values.shape}")


