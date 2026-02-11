#!/usr/bin/env python3
"""
Quick test script to verify the project setup is working correctly.
This script tests data loading and basic model instantiation without full training.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import tensorflow as tf
import numpy as np
from pathlib import Path

print("=" * 70)
print("PROJECT SETUP VERIFICATION TEST")
print("=" * 70)

# Check TensorFlow installation
print(f"\n✓ TensorFlow version: {tf.__version__}")
print(f"✓ Keras version: {tf.keras.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"✓ GPUs available: {len(gpus)} {'(CPU-only mode)' if len(gpus) == 0 else ''}")

# Check data directory
project_dir = Path(__file__).parent
data_dir = project_dir / 'data'
model_dir = project_dir / 'model'

print(f"\n✓ Project directory: {project_dir}")
print(f"✓ Data directory: {data_dir}")
print(f"✓ Model directory: {model_dir}")

# Verify data files exist
required_files = ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy']
print("\nChecking data files:")
for file in required_files:
    file_path = data_dir / file
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {file}: {size_mb:.2f} MB")
    else:
        print(f"  ✗ {file}: MISSING")
        print("\nPlease run: python src/data_loader.py")
        exit(1)

# Load a small sample of data
print("\nLoading data samples...")
x_train = np.load(data_dir / 'x_train.npy')
y_train = np.load(data_dir / 'y_train.npy')
x_test = np.load(data_dir / 'x_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

print(f"  ✓ Train data shape: {x_train.shape}")
print(f"  ✓ Test data shape: {x_test.shape}")
print(f"  ✓ Train labels shape: {y_train.shape}")
print(f"  ✓ Test labels shape: {y_test.shape}")
print(f"  ✓ Pixel value range: [{x_train.min():.2f}, {x_train.max():.2f}]")

# Test model instantiation
print("\nTesting MobileNetV2 instantiation...")
try:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    print(f"  ✓ MobileNetV2 loaded successfully")
    print(f"  ✓ Total parameters: {base_model.count_params():,}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# Test basic inference with a dummy model
print("\nTesting basic model inference...")
try:
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    test_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Test with a single batch
    sample_batch = x_train[:1]
    sample_labels = y_train[:1]
    predictions = test_model.predict(sample_batch, verbose=0)
    print(f"  ✓ Model inference successful")
    print(f"  ✓ Output shape: {predictions.shape}")
    print(f"  ✓ Prediction sum: {predictions.sum():.4f} (should be ~1.0)")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - PROJECT SETUP IS WORKING!")
print("=" * 70)
print("\nNext steps:")
print("  1. Train baseline model: python src/final_training_pipeline.py")
print("  2. Or convert existing model: python src/convert_and_evaluate.py")
print("=" * 70)
