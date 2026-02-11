import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import sys

print("=" * 60)
print("QUANTIZATION-AWARE TRAINING FOR MOBILENETV2")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU Available: {len(gpus) > 0}")
if gpus:
    print(f"GPU Device: {gpus[0].name}")
    tf.config.experimental.set_memory_growth(gpus[0], True)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model')
os.makedirs(model_dir, exist_ok=True)

print(f"\nLoading preprocessed data from {data_dir}...")
x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

print(f"Train data shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels: {y_test.shape}")

print("\n" + "=" * 60)
print("STEP 1: Load Pre-trained MobileNetV2 (Float32)")
print("=" * 60)

inputs = tf.keras.Input(shape=(224, 224, 3))

base_model = MobileNetV2(
    input_tensor=inputs,
    include_top=False,
    weights='imagenet'
)

x = base_model.output
x = GlobalAveragePooling2D(name='global_avg_pool')(x)
predictions = Dense(10, activation='softmax', name='predictions')(x)

model_float32 = tf.keras.Model(inputs=inputs, outputs=predictions)

for layer in base_model.layers[:-30]:
    layer.trainable = False

model_float32.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nModel created with {len(model_float32.layers)} layers")
print(f"Trainable parameters: {model_float32.count_params()}")

print("\nFine-tuning baseline model (5 epochs for better accuracy)...")
history_baseline = model_float32.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=1
)

baseline_acc = history_baseline.history['val_accuracy'][-1]
print(f"\nBaseline Float32 Model Validation Accuracy: {baseline_acc:.4f}")

baseline_path = os.path.join(model_dir, 'mobilenet_float32.keras')
model_float32.save(baseline_path)
print(f"Baseline model saved to: {baseline_path}")

baseline_size = os.path.getsize(baseline_path) / (1024 * 1024)
print(f"Baseline model size: {baseline_size:.2f} MB")

print("\n" + "=" * 60)
print("STEP 2: Apply Quantization-Aware Training")
print("=" * 60)

print("Cloning model to create functional API compatible version...")
cloned_model = tf.keras.models.clone_model(model_float32)
cloned_model.set_weights(model_float32.get_weights())

quantize_model = tfmot.quantization.keras.quantize_model

q_aware_model = quantize_model(cloned_model)

q_aware_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nQuantization-aware model created")
print(f"Model now contains fake quantization nodes")

print("\nFine-tuning with QAT (3 epochs)...")
history_qat = q_aware_model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=3,
    validation_data=(x_test, y_test),
    verbose=1
)

qat_acc = history_qat.history['val_accuracy'][-1]
print(f"\nQAT Model Validation Accuracy: {qat_acc:.4f}")
print(f"Accuracy difference: {abs(baseline_acc - qat_acc):.4f} ({abs(baseline_acc - qat_acc) * 100:.2f}%)")

qat_model_path = os.path.join(model_dir, 'mobilenet_qat.keras')
q_aware_model.save(qat_model_path)
print(f"\nQAT model saved to: {qat_model_path}")

print("\n" + "=" * 60)
print("QAT TRAINING COMPLETE")
print("=" * 60)
print(f"Baseline Accuracy: {baseline_acc:.4f}")
print(f"QAT Accuracy: {qat_acc:.4f}")
print(f"Accuracy Drop: {(baseline_acc - qat_acc) * 100:.2f}%")
print("\nNext step: Convert to TFLite and measure quantized model size")