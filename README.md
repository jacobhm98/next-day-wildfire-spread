# Next-Day Wildfire Spread Prediction

CNN autoencoder for predicting wildfire spread based on the paper "Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data" (Huot et al., 2022).

## Dataset

The dataset contains ~18,500 fire events from 2012-2020 across the United States:
- **Training**: 14,979 samples
- **Evaluation**: 1,877 samples
- **Test**: 1,689 samples

Each sample is a 64×64 km region at 1 km resolution with:
- **12 input features**: elevation, wind direction, wind speed, min/max temperature, humidity, precipitation, drought index, vegetation (NDVI), population density, energy release component (ERC), and previous fire mask
- **1 output**: Next-day fire mask (binary prediction per pixel)

## Model Architecture

CNN Autoencoder following the paper's specifications:
- **Initial conv**: 12 → 16 filters
- **Encoder**: 2 residual blocks with max pooling (16 → 16 → 32 filters)
- **Bottleneck**: 1 residual block (32 filters)
- **Decoder**: 2 residual blocks with upsampling (32 → 16 → 16 filters)
- **Output conv**: 16 → 1 filter

**Total parameters**: ~52k (very lightweight!)

## Training Configuration

Matching the paper:
- **Loss**: Weighted binary cross-entropy (weight=3 on fire class)
- **Optimizer**: Adam with learning rate 0.0001
- **Batch size**: 32
- **Dropout**: 0.1
- **Data augmentation**: Random flip and rotation
- **Metrics**: AUC-PR, precision, recall

## Installation

Make sure you have `uv` installed: https://docs.astral.sh/uv/

```bash
uv sync
```

## Prepare Dataset

```bash
# Download dataset
./download_dataset.sh

# Create index files for .tfrecord files
uv run python -m tfrecord.tools.tfrecord2idx data/next_day_wildfire_spread_train_00.tfrecord data/next_day_wildfire_spread_train_00.tfindex
# (repeat for all tfrecord files, or create a script to automate)
```

## Usage

### Test the dataset loading

```bash
uv run python dataset.py
```

### Test the model

```bash
uv run python model.py
```

### Train the model

```bash
# Train with default parameters (100 epochs)
uv run python train.py

# Train with custom parameters
uv run python train.py --epochs 1000 --batch-size 64 --lr 0.0001

# Full training as in paper (1000 epochs)
uv run python train.py --epochs 1000 --batch-size 32 --lr 0.0001 --pos-weight 3.0
```

### Monitor training with TensorBoard

```bash
uv run tensorboard --logdir logs
```

## Files

- `model.py`: CNN autoencoder architecture
- `dataset.py`: Dataset loading and preprocessing
- `train.py`: Training loop with metrics and evaluation
- `main.py`: Original exploration script

## Checkpoints and Logs

After each epoch, the training script saves:
- **`models/epoch_XXXX.pt`**: Checkpoint for each epoch with full metadata
- **`models/best_model.pt`**: Best model based on validation AUC-PR
- **`models/latest.pt`**: Most recent checkpoint
- **`logs/run_TIMESTAMP/`**: TensorBoard logs

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'train_loss': float,
    'eval_loss': float,
    'auc_pr': float,
    'precision': float,
    'recall': float,
    'hyperparameters': {
        'batch_size': int,
        'learning_rate': float,
        'pos_weight': float,
    },
    'timestamp': str,
    'run_name': str,
}
```

## Paper Results

From the paper (Table II):
- **AUC-PR**: 28.4%
- **Precision**: 33.6%
- **Recall**: 43.1%

These are the targets to match/exceed with this implementation.

## References

Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., & Chen, Y. F. (2022). Next day wildfire spread: A machine learning dataset to predict wildfire spreading from remote-sensing data. *IEEE Transactions on Geoscience and Remote Sensing*, 60, 1-13.

- Paper: https://ieeexplore.ieee.org/abstract/document/9840400
- Dataset: https://www.kaggle.com/fantineh/next-day-wildfire-spread
- Code: https://github.com/google-research/google-research/tree/master/simulation_research/next_day_wildfire_spread
