# next-day-wildfire-spread

Using the "next day wildfire spread" dataset to train a CNN

## Original paper:

- https://ieeexplore.ieee.org/abstract/document/9840400

## Install dependencies

- Make sure you have `uv` installed: https://docs.astral.sh/uv/
- Run `uv sync`

## Prepare dataset

- Run `./download_dataset.sh`
- Create index for .tfrecord files
- Prepare tensor flow indices `uv run create_index.py`
