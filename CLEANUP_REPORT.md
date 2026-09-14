# Phase 14: Cleanup Report

## Deleted Files & Directories
- `COAST-VNet-1/` directory: Removed. Contained the obsolete COAST-VNet-1 ONNX model and old inference wrappers which produced poor velocity agreement on IO-VNBD data.
- `evaluate_coast_iovnbd.py`: Removed. Obsolete evaluation script for COAST-VNet-1.
- `test_coast_iovnbd.py`: Removed. Obsolete evaluation script.
- `coast_all_outputs.png`, `coast_output0_mps_vs_gps.png`, `coast_output0_vs_gps.png`: Removed obsolete evaluation plots for the COAST model.

## Retained Files
- `src/ai_models/train_phase6.py`: Retained as the definitive training pipeline for the new CNN-GRU model.
- `mobile_app/wayfinder_app/lib/core/speed_estimator.dart`: Retained as the ONNX wrapper for the CNN-GRU model.
- `src/preprocessing/pipeline.py`: Retained as the definitive preprocessing and sliding-window logic.
- `data/IO-VNBD-master/`: Retained as the core training dataset.
- `mobile_app/wayfinder_app/lib/core/ekf_navigation.dart`: Retained as the core EKF implementation.
- `mobile_app/wayfinder_app/lib/core/navigation_service.dart`: Retained as the core DR loop and sensor integration.

## Dependencies
- Removed `onnxruntime` bindings referencing COAST-VNet-1 specific tensor shapes.
- Retained `onnxruntime` for the new CNN-GRU model (`phase6_speed_model.onnx`).
