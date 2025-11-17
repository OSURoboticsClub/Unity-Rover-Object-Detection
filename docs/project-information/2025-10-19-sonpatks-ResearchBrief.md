# Background Research Findings 1 – Mars Rover Autonomy Capstone

## Slice
**Evaluation of You-Only-Look-Once (YOLO) models** that support development on macOS and Linux, with deployment targeting **Ubuntu 22.04 + ROS2 + GStreamer** for the rover’s object detection pipeline.

## Why it Matters
The rover’s object detection stack will be built across **macOS M1** and **Ubuntu 22.04** systems using **ROS2** (robot middleware) and **GStreamer** (audio/video pipeline).  
A **cross-platform workflow** is essential to avoid mismatched environments, repeated debugging, and deployment slowdowns.  
If the chosen model format is incompatible across systems, the team risks major delays.

---

## Key Findings

- **Cross-Platform Support:**  
  YOLO models can be trained in PyTorch and exported to **ONNX**, enabling identical inference workflows on macOS M1 and Ubuntu systems [1][2].

- **Performance Optimization:**  
  Benchmarks show significantly lower latency on NVIDIA GPUs when YOLO models are compiled with **TensorRT**, compared to ONNX/PyTorch inference alone—critical for real-time detection [3].

- **Integration with ROS2 + GStreamer:**  
  Existing ROS2 pipelines (Stereolabs, dusty-nv) show low-latency video streaming and detection using ONNX/TensorRT, confirming a viable integration path for the rover [4][5].

---

## Model Comparison Table

| **Model**        | **Short Overview** | **Pros** | **Cons** | **Recommended Use Case** |
|------------------|--------------------|----------|-----------|---------------------------|
| **YOLOv5-nano** | Smallest YOLOv5 variant; PyTorch; ONNX exportable | Extremely lightweight; very high FPS; easy to train on Mac; ONNX-friendly | Lower accuracy; struggles with small/occluded objects | **Month 1 Prototype:** Early inference tests on Mac and ROS2/GStreamer integration |
| **YOLOv5-small** | Larger YOLOv5 variant; more parameters | Balanced accuracy/speed; still lightweight; ONNX exportable | Slower than nano; moderate latency | **Month 2–3 Upgrade:** Use if accuracy needs increase; minimal ROS2 complications |
| **YOLOv8-nano** | Latest YOLO architecture; ONNX support | Lightweight; stable modern architecture | Slightly less community support; minor integration tweaks may be needed | **Alternative Prototype:** Leverages newest YOLO features while staying lightweight |
| **YOLOv8-small** | Larger v8 variant with more parameters | Better accuracy than v8-nano; improved training | Higher latency; slower on embedded GPUs | **Month 3–4 Accuracy Upgrade:** Intended for rover deployment with ROS2 |

---

## Implications

### 1. Cross-Platform Design Requirement
Standardize on **ONNX-exportable YOLO models** (v5 or v8) to ensure a unified workflow between macOS M1 and Ubuntu 22.04.

### 2. Performance Risk Mitigation
Prototype ONNX models with **TensorRT** early. Real-time detection depends on it; skipping TensorRT risks high latency and missed detections during rover navigation.

### 3. Scoped Integration Strategy
Adopt a phased model approach:

- Begin with **nano** variants for rapid prototyping.  
- Progress to **small** variants for improved accuracy.  
- Align each phase with ROS2 + GStreamer integration milestones.  

This approach enables measurable benchmarks (FPS, accuracy) and prevents scope creep.

---

## References

1. PyTorch, *Introducing Accelerated PyTorch Training on Mac* (2022).  
2. Ultralytics, *YOLOv5 / YOLOv8 Model Export & Benchmark* (2023).  
3. NVIDIA, *TensorRT Developer Guide & Performance Benchmarks* (2025).  
4. Stereolabs, *ROS2 Object Detection Integration* (2024).  
5. dusty-nv, *ROS Deep Learning Repository* (2024).  
