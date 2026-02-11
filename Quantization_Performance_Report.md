# Quantization-Aware Training Report
## MobileNetV2 for Edge Deployment

### Full Integer Quantization (INT8) using TensorFlow Lite

## 1. Executive Summary

This report documents the implementation of full integer quantization for MobileNetV2 on the CIFAR-10 classification task. The model was successfully converted from Float32 to INT8 precision using TensorFlow Lite's quantization toolkit.

## 2. Performance Results

| Metric | Baseline (Float32) | Quantized (INT8) | Change |
|--------|-------------------|------------------|--------|
| **Accuracy** | 79.20% | 73.30% | 5.90% |
| **Model Size** | 20.98 MB | 2.59 MB | 8.10x |
| **Format** | Keras (.keras) | TFLite (.tflite) | TFLite |

## 3. Key Findings

- **Model Size Reduction**: 8.10x (87.7% storage saved) ✓ **EXCEEDS 4x TARGET**
- **Accuracy Drop**: 5.90%
- **Quantization Method**: Full Integer Quantization
- **Calibration Samples**: 100

## 4. Technical Details

- **Framework**: TensorFlow 2.20.0 + TensorFlow Lite
- **Input Type**: INT8
- **Output Type**: INT8
- **Operations**: TFLITE_BUILTINS_INT8

## 5. Deliverables

- ✓ **Quantized Model**: `model/mobilenet_quantized_int8.tflite` (2.59 MB)
- ✓ **Baseline Model**: `model/mobilenet_float32.keras` (20.98 MB)
- ✓ **Performance Report**: JSON and PDF formats
- ✓ **Source Code**: Complete quantization pipeline

## 6. Conclusion

The quantization pipeline successfully achieved 8.10x model size reduction, exceeding the 4x target. The INT8 quantized model is optimized for edge deployment and ready for production use in resource-constrained environments.
