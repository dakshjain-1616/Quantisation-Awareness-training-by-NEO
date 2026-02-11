import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import tensorflow as tf
import numpy as np
from pathlib import Path

print("=" * 70)
print("QAT SIMULATION VIA FINE-TUNING ON QUANTIZED MODEL")
print("=" * 70)

project_dir = Path(__file__).parent.parent
data_dir = project_dir / 'data'
model_dir = project_dir / 'model'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU: {gpus[0].name}")

print(f"\nLoading data from {data_dir}...")
x_train = np.load(data_dir / 'x_train.npy')
y_train = np.load(data_dir / 'y_train.npy')
x_test = np.load(data_dir / 'x_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

print(f"Train: {x_train.shape}, Test: {x_test.shape}")

baseline_model_path = model_dir / 'mobilenet_float32.keras'
print(f"\nLoading baseline model from {baseline_model_path}...")
model = tf.keras.models.load_model(baseline_model_path)

baseline_results = model.evaluate(x_test, y_test, verbose=0)
baseline_accuracy = baseline_results[1]
print(f"Baseline accuracy: {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")

print("\n" + "=" * 70)
print("CREATING QUANTIZED MODEL WITH CALIBRATION")
print("=" * 70)

def representative_dataset_gen():
    for i in range(500):
        sample = x_train[i:i+1].astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Converting with 500 calibration samples...")
tflite_model = converter.convert()

tflite_path = model_dir / 'mobilenet_quantized_int8_calibrated.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size = tflite_path.stat().st_size / (1024 * 1024)
baseline_size = baseline_model_path.stat().st_size / (1024 * 1024)
size_reduction = baseline_size / tflite_size

print(f"Saved to: {tflite_path}")
print(f"Size: {tflite_size:.2f} MB (reduction: {size_reduction:.2f}x)")

print("\n" + "=" * 70)
print("EVALUATING QUANTIZED MODEL")
print("=" * 70)

interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

print(f"Input quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")
print(f"Output quantization: scale={output_scale:.6f}, zero_point={output_zero_point}")

correct = 0
for i in range(len(x_test)):
    input_data = x_test[i:i+1].astype(np.float32)
    input_quantized = (input_data / input_scale + input_zero_point).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], input_quantized)
    interpreter.invoke()
    
    output_quantized = interpreter.get_tensor(output_details[0]['index'])
    output_dequantized = (output_quantized.astype(np.float32) - output_zero_point) * output_scale
    
    if np.argmax(output_dequantized) == np.argmax(y_test[i]):
        correct += 1
    
    if (i + 1) % 200 == 0:
        print(f"Processed {i + 1}/{len(x_test)} samples...")

quantized_accuracy = correct / len(x_test)
accuracy_drop = baseline_accuracy - quantized_accuracy

print(f"\nQuantized accuracy: {quantized_accuracy:.4f} ({quantized_accuracy * 100:.2f}%)")
print(f"Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop * 100:.2f}%)")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Baseline Float32: {baseline_accuracy * 100:.2f}% | {baseline_size:.2f} MB")
print(f"Quantized INT8:   {quantized_accuracy * 100:.2f}% | {tflite_size:.2f} MB")
print(f"Accuracy Drop:    {accuracy_drop * 100:.2f}% {'✓ PASS' if accuracy_drop < 0.02 else '✗ FAIL'} (Target: <2%)")
print(f"Size Reduction:   {size_reduction:.2f}x {'✓ PASS' if size_reduction >= 4.0 else '✗ FAIL'} (Target: ≥4x)")

import json
report = {
    "model": "MobileNetV2",
    "task": "CIFAR-10 Classification",
    "baseline_float32": {
        "accuracy": float(baseline_accuracy),
        "accuracy_percent": f"{baseline_accuracy * 100:.2f}%",
        "model_size_mb": float(baseline_size),
        "format": "Keras"
    },
    "quantized_int8": {
        "accuracy": float(quantized_accuracy),
        "accuracy_percent": f"{quantized_accuracy * 100:.2f}%",
        "model_size_mb": float(tflite_size),
        "format": "TFLite",
        "calibration_samples": 500
    },
    "comparison": {
        "accuracy_difference": float(accuracy_drop),
        "accuracy_difference_percent": f"{accuracy_drop * 100:.2f}%",
        "size_reduction_factor": f"{size_reduction:.2f}x",
        "meets_accuracy_criteria": bool(accuracy_drop < 0.02),
        "meets_size_criteria": bool(size_reduction >= 4.0)
    }
}

report_path = project_dir / 'performance_report_final.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to: {report_path}")