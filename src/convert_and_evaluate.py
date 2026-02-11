import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import tensorflow as tf
import numpy as np
import json
from pathlib import Path

print("=" * 70)
print("TFLITE CONVERSION WITH FULL INTEGER QUANTIZATION")
print("=" * 70)

project_dir = Path(__file__).parent.parent
data_dir = project_dir / 'data'
model_dir = project_dir / 'model'

print(f"\nLoading preprocessed data from {data_dir}...")
x_train = np.load(data_dir / 'x_train.npy')
y_train = np.load(data_dir / 'y_train.npy')
x_test = np.load(data_dir / 'x_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

print(f"Train data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

baseline_model_path = model_dir / 'mobilenet_augmented.keras'
print(f"\nLoading baseline model from {baseline_model_path}...")
model_float32 = tf.keras.models.load_model(baseline_model_path)

print("\n" + "=" * 70)
print("STEP 1: Evaluate Baseline Float32 Model")
print("=" * 70)

baseline_results = model_float32.evaluate(x_test, y_test, verbose=0)
baseline_loss = baseline_results[0]
baseline_accuracy = baseline_results[1]

print(f"Baseline Float32 Model:")
print(f"  Loss: {baseline_loss:.4f}")
print(f"  Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")

baseline_size = baseline_model_path.stat().st_size / (1024 * 1024)
print(f"  Model Size: {baseline_size:.2f} MB")

print("\n" + "=" * 70)
print("STEP 2: Convert to TFLite with Full Integer Quantization")
print("=" * 70)

def representative_dataset_gen():
    num_calibration_samples = 200
    for i in range(num_calibration_samples):
        sample = x_train[i:i+1].astype(np.float32)
        yield [sample]

print("Creating TFLite converter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model_float32)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting to TFLite with full integer quantization...")
print("  - Using 100 calibration samples from training set")
print("  - Target: INT8 operations")
print("  - Input/Output: INT8")

tflite_quant_model = converter.convert()

tflite_model_path = model_dir / 'mobilenet_quantized_final.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

tflite_size = tflite_model_path.stat().st_size / (1024 * 1024)
size_reduction = baseline_size / tflite_size

print(f"\nQuantized model saved to: {tflite_model_path}")
print(f"  Quantized Model Size: {tflite_size:.2f} MB")
print(f"  Size Reduction: {size_reduction:.2f}x")
print(f"  Size Saved: {baseline_size - tflite_size:.2f} MB ({(1 - tflite_size/baseline_size) * 100:.1f}%)")

print("\n" + "=" * 70)
print("STEP 3: Evaluate Quantized TFLite Model")
print("=" * 70)

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input details:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")
print(f"  Quantization: scale={input_details[0]['quantization'][0]}, zero_point={input_details[0]['quantization'][1]}")

print(f"\nOutput details:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")
print(f"  Quantization: scale={output_details[0]['quantization'][0]}, zero_point={output_details[0]['quantization'][1]}")

print("\nRunning inference on test set...")

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

correct_predictions = 0
total_predictions = len(x_test)

for i in range(total_predictions):
    input_data = x_test[i:i+1].astype(np.float32)
    
    input_quantized = (input_data / input_scale + input_zero_point).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], input_quantized)
    interpreter.invoke()
    
    output_quantized = interpreter.get_tensor(output_details[0]['index'])
    
    output_dequantized = (output_quantized.astype(np.float32) - output_zero_point) * output_scale
    
    predicted_class = np.argmax(output_dequantized)
    true_class = np.argmax(y_test[i])
    
    if predicted_class == true_class:
        correct_predictions += 1
    
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{total_predictions} samples...")

quantized_accuracy = correct_predictions / total_predictions
accuracy_drop = baseline_accuracy - quantized_accuracy

print(f"\nQuantized INT8 Model:")
print(f"  Accuracy: {quantized_accuracy:.4f} ({quantized_accuracy * 100:.2f}%)")
print(f"  Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop * 100:.2f}%)")

print("\n" + "=" * 70)
print("STEP 4: Generate Performance Report")
print("=" * 70)

report = {
    "model": "MobileNetV2",
    "task": "CIFAR-10 Classification (10 classes)",
    "dataset": {
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "input_shape": [224, 224, 3],
        "preprocessing": "Normalized to [-1, 1]"
    },
    "baseline_float32": {
        "accuracy": float(baseline_accuracy),
        "accuracy_percent": f"{baseline_accuracy * 100:.2f}%",
        "loss": float(baseline_loss),
        "model_size_mb": float(baseline_size),
        "format": "Keras (.keras)"
    },
    "quantized_int8": {
        "accuracy": float(quantized_accuracy),
        "accuracy_percent": f"{quantized_accuracy * 100:.2f}%",
        "model_size_mb": float(tflite_size),
        "format": "TFLite (.tflite)",
        "quantization_method": "Full Integer Quantization",
        "calibration_samples": 100
    },
    "comparison": {
        "accuracy_difference": float(accuracy_drop),
        "accuracy_difference_percent": f"{accuracy_drop * 100:.2f}%",
        "size_reduction_factor": f"{size_reduction:.2f}x",
        "size_saved_mb": float(baseline_size - tflite_size),
        "size_saved_percent": f"{(1 - tflite_size/baseline_size) * 100:.1f}%",
        "meets_accuracy_criteria": accuracy_drop < 0.02,
        "meets_size_criteria": size_reduction >= 4.0
    },
    "quantization_details": {
        "input_type": "INT8",
        "output_type": "INT8",
        "operations": "TFLITE_BUILTINS_INT8",
        "input_quantization": {
            "scale": float(input_scale),
            "zero_point": int(input_zero_point)
        },
        "output_quantization": {
            "scale": float(output_scale),
            "zero_point": int(output_zero_point)
        }
    }
}

report_path = project_dir / 'performance_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nPerformance report saved to: {report_path}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Baseline Float32 Accuracy: {baseline_accuracy * 100:.2f}%")
print(f"✓ Quantized INT8 Accuracy: {quantized_accuracy * 100:.2f}%")
print(f"✓ Accuracy Drop: {accuracy_drop * 100:.2f}% {'✓ PASS' if accuracy_drop < 0.02 else '✗ FAIL'} (Target: <2%)")
print(f"✓ Model Size Reduction: {size_reduction:.2f}x {'✓ PASS' if size_reduction >= 4.0 else '✗ FAIL'} (Target: ≥4x)")
print(f"✓ Baseline Size: {baseline_size:.2f} MB")
print(f"✓ Quantized Size: {tflite_size:.2f} MB")
print("\n" + "=" * 70)
print("QUANTIZATION PIPELINE COMPLETE!")
print("=" * 70)