import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with optional max pooling or upsampling."""

    def __init__(self, in_channels, out_channels, pool=False, upsample=False, dropout_rate=0.1):
        super().__init__()
        self.pool = pool
        self.upsample = upsample

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

        # Pooling or upsampling
        if pool:
            self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        out = self.dropout2(out)

        if self.pool:
            out = self.pool_layer(out)
        if self.upsample:
            out = self.upsample_layer(out)

        return out


class CNNAutoencoder(nn.Module):
    """CNN Autoencoder for wildfire spread prediction.

    Architecture from "Next Day Wildfire Spread: A Machine Learning Dataset to Predict
    Wildfire Spreading From Remote-Sensing Data" (Huot et al., 2022).

    The model uses:
    - 16 filters for most blocks
    - 32 filters for the middle two residual blocks (bottleneck)
    - 3x3 convolutions with stride 1x1
    - 2x2 pooling
    - Dropout rate of 0.1
    """

    def __init__(self, in_channels=12, out_channels=1, dropout_rate=0.1):
        """
        Args:
            in_channels: Number of input feature channels (default: 12 for the 11 features + PrevFireMask)
            out_channels: Number of output channels (default: 1 for FireMask prediction)
            dropout_rate: Dropout rate (default: 0.1)
        """
        super().__init__()

        # Initial convolution
        self.conv_initial = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn_initial = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_initial = nn.Dropout2d(dropout_rate)

        # Encoder (downsampling path)
        self.encoder1 = ResidualBlock(16, 16, pool=True, dropout_rate=dropout_rate)
        self.encoder2 = ResidualBlock(16, 32, pool=True, dropout_rate=dropout_rate)  # Middle block 1

        # Bottleneck
        self.bottleneck = ResidualBlock(32, 32, dropout_rate=dropout_rate)  # Middle block 2

        # Decoder (upsampling path)
        self.decoder1 = ResidualBlock(32, 16, upsample=True, dropout_rate=dropout_rate)
        self.decoder2 = ResidualBlock(16, 16, upsample=True, dropout_rate=dropout_rate)

        # Final convolution to get output
        self.conv_final = nn.Conv2d(16, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # Initial convolution
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = self.relu(x)
        x = self.dropout_initial(x)

        # Encoder
        x = self.encoder1(x)  # 32x32 (if input is 64x64) or 16x16 (if input is 32x32)
        x = self.encoder2(x)  # 16x16 (if input is 64x64) or 8x8 (if input is 32x32)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder1(x)  # 32x32 or 16x16
        x = self.decoder2(x)  # 64x64 or 32x32

        # Final output
        x = self.conv_final(x)

        return x


def create_model(in_channels=12, out_channels=1, dropout_rate=0.1):
    """Factory function to create the CNN autoencoder model.

    Args:
        in_channels: Number of input feature channels
        out_channels: Number of output channels
        dropout_rate: Dropout rate

    Returns:
        CNNAutoencoder model
    """
    return CNNAutoencoder(in_channels, out_channels, dropout_rate)


if __name__ == "__main__":
    # Test the model
    model = create_model(in_channels=12, out_channels=1)

    # Print model summary
    print(model)
    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with dummy input
    batch_size = 4
    input_size = 32  # After random crop from 64x64
    dummy_input = torch.randn(batch_size, 12, input_size, input_size)

    print(f"\nInput shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, 1, input_size, input_size), "Output shape mismatch!"
    print("\n✓ Model test passed!")
