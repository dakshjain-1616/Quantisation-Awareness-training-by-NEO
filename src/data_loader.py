import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import sys

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print(f"Original train shape: {x_train.shape}, test shape: {x_test.shape}")
    
    def preprocess_images(images):
        images = tf.image.resize(images, (224, 224))
        images = tf.cast(images, tf.float32)
        images = (images / 127.5) - 1.0
        return images.numpy()
    
    print("Using subset for memory efficiency (5000 train, 1000 test samples)")
    x_train_subset = x_train[:5000]
    x_test_subset = x_test[:1000]
    y_train_subset = y_train[:5000]
    y_test_subset = y_test[:1000]
    
    print("Resizing and normalizing images to 224x224 and [-1, 1] range...")
    x_train_processed = preprocess_images(x_train_subset)
    x_test_processed = preprocess_images(x_test_subset)
    
    y_train_cat = to_categorical(y_train_subset, 10)
    y_test_cat = to_categorical(y_test_subset, 10)
    
    print(f"Processed train shape: {x_train_processed.shape}")
    print(f"Processed test shape: {x_test_processed.shape}")
    print(f"Train labels shape: {y_train_cat.shape}")
    print(f"Test labels shape: {y_test_cat.shape}")
    print(f"Pixel value range: [{x_train_processed.min():.2f}, {x_train_processed.max():.2f}]")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, 'x_train.npy'), x_train_processed)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train_cat)
    np.save(os.path.join(data_dir, 'x_test.npy'), x_test_processed)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test_cat)
    
    print(f"\nData saved to {data_dir}")
    print("Files created:")
    print(f"  - x_train.npy: {os.path.getsize(os.path.join(data_dir, 'x_train.npy')) / (1024**3):.2f} GB")
    print(f"  - x_test.npy: {os.path.getsize(os.path.join(data_dir, 'x_test.npy')) / (1024**3):.2f} GB")
    
    return x_train_processed, y_train_cat, x_test_processed, y_test_cat

if __name__ == "__main__":
    load_and_preprocess_data()
    print("\nData pipeline setup complete!")