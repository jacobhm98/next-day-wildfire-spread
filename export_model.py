"""Export trained model to various portable formats."""

import torch
import argparse
import json
from pathlib import Path
from model import CNNAutoencoder


def export_to_torchscript(checkpoint_path, output_path):
    """Export model to TorchScript (portable, no Python dependencies).

    TorchScript can be loaded in C++, production environments, etc.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model
    model = CNNAutoencoder(in_channels=12, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create example input
    example_input = torch.randn(1, 12, 64, 64)

    # Trace the model
    print("Tracing model with TorchScript...")
    traced_model = torch.jit.trace(model, example_input)

    # Save
    traced_model.save(output_path)
    print(f"✓ Saved TorchScript model to {output_path}")

    # Save metadata separately
    metadata = {
        'epoch': checkpoint['epoch'],
        'auc_pr': checkpoint['auc_pr'],
        'precision': checkpoint['precision'],
        'recall': checkpoint['recall'],
        'train_loss': checkpoint['train_loss'],
        'eval_loss': checkpoint['eval_loss'],
        'hyperparameters': checkpoint['hyperparameters'],
        'timestamp': checkpoint['timestamp'],
        'run_name': checkpoint['run_name'],
    }

    metadata_path = output_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")

    return traced_model, metadata


def export_to_onnx(checkpoint_path, output_path):
    """Export model to ONNX format (interoperable with TensorFlow, etc.).

    ONNX can be used with many frameworks and deployment tools.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model
    model = CNNAutoencoder(in_channels=12, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create example input
    example_input = torch.randn(1, 12, 64, 64)

    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✓ Saved ONNX model to {output_path}")

    # Save metadata separately
    metadata = {
        'epoch': checkpoint['epoch'],
        'auc_pr': checkpoint['auc_pr'],
        'precision': checkpoint['precision'],
        'recall': checkpoint['recall'],
        'train_loss': checkpoint['train_loss'],
        'eval_loss': checkpoint['eval_loss'],
        'hyperparameters': checkpoint['hyperparameters'],
        'timestamp': checkpoint['timestamp'],
        'run_name': checkpoint['run_name'],
    }

    metadata_path = output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")


def export_weights_only(checkpoint_path, output_path):
    """Export just the weights as a simple state dict (still needs architecture).

    This is the most compact format but still requires the model class.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Save just the state dict
    torch.save(checkpoint['model_state_dict'], output_path)
    print(f"✓ Saved weights to {output_path}")

    # Save metadata separately
    metadata = {
        'epoch': checkpoint['epoch'],
        'auc_pr': checkpoint['auc_pr'],
        'precision': checkpoint['precision'],
        'recall': checkpoint['recall'],
        'train_loss': checkpoint['train_loss'],
        'eval_loss': checkpoint['eval_loss'],
        'hyperparameters': checkpoint['hyperparameters'],
        'timestamp': checkpoint['timestamp'],
        'run_name': checkpoint['run_name'],
    }

    metadata_path = output_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")


def test_load_torchscript(model_path):
    """Test loading the TorchScript model (no Python code needed!)."""
    print(f"\nTesting TorchScript model from {model_path}...")

    # Load model (no Python model definition needed!)
    loaded_model = torch.jit.load(model_path)
    loaded_model.eval()

    # Test inference
    example_input = torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        output = loaded_model(example_input)

    print(f"✓ TorchScript model loaded successfully!")
    print(f"  Input shape: {example_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    return loaded_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model to portable formats")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file (e.g., models/best_model.pt)")
    parser.add_argument("--format", type=str, choices=["torchscript", "onnx", "weights"],
                        default="torchscript", help="Export format")
    parser.add_argument("--output", type=str, help="Output path (optional)")
    parser.add_argument("--test", action="store_true", help="Test loading the exported model")

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        checkpoint_stem = Path(args.checkpoint).stem
        if args.format == "torchscript":
            output_path = f"models/{checkpoint_stem}_torchscript.pt"
        elif args.format == "onnx":
            output_path = f"models/{checkpoint_stem}.onnx"
        else:  # weights
            output_path = f"models/{checkpoint_stem}_weights_only.pt"

    # Export
    if args.format == "torchscript":
        traced_model, metadata = export_to_torchscript(args.checkpoint, output_path)

        # Test if requested
        if args.test:
            test_load_torchscript(output_path)

    elif args.format == "onnx":
        export_to_onnx(args.checkpoint, output_path)

        if args.test:
            print("\nTo test ONNX model, install onnxruntime: pip install onnxruntime")

    else:  # weights
        export_weights_only(args.checkpoint, output_path)
        print("\nNote: Weights-only format still requires model.py to load")

    print(f"\n✓ Export complete!")
    print(f"\nTo use the exported model:")
    if args.format == "torchscript":
        print(f"  model = torch.jit.load('{output_path}')")
        print(f"  model.eval()")
        print(f"  output = model(input_tensor)")
    elif args.format == "onnx":
        print(f"  import onnxruntime as ort")
        print(f"  session = ort.InferenceSession('{output_path}')")
        print(f"  output = session.run(None, {{'input': input_array}})")
    else:
        print(f"  from model import CNNAutoencoder")
        print(f"  model = CNNAutoencoder(in_channels=12, out_channels=1)")
        print(f"  model.load_state_dict(torch.load('{output_path}'))")
