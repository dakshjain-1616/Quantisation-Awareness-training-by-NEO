# Quantization-Aware Training - Final Report

## MobileNetV2 with Data Augmentation for Edge Deployment

### Performance Summary

| Metric | Baseline (Float32) | Quantized (INT8) | Result |
|--------|-------------------|------------------|--------|
| **Accuracy** | 81.00% | 77.20% | 3.80% drop |
| **Model Size** | 23.52 MB | 2.59 MB | 9.08x reduction |

### Acceptance Criteria

- ✓ **File Format**: .tflite
- ✓ **Size Reduction**: 9.08x (Target: ≥4x)
- ✗ **Accuracy Drop**: 3.80% (Target: <2%)
- ✓ **INT8 Quantization**: Confirmed (input/output types INT8)

### Technical Details

- **Training Strategy**: 8 epochs with data augmentation
- **Augmentation**: Random flip, rotation (±10°), zoom (±10%)
- **Dropout**: 0.2 before final layer
- **Calibration**: 200 representative samples
- **Quantization**: Full integer (INT8 ops, weights, activations)

### Deliverables

- ✓ Quantized Model: `model/mobilenet_quantized_final.tflite` (2.59 MB)
- ✓ Baseline Model: `model/mobilenet_augmented.keras` (23.52 MB)
- ✓ Performance Reports: JSON and Markdown formats
