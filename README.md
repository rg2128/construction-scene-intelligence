# Construction Scene Intelligence

A multimodal computer vision system for construction-site scene understanding, worker-level PPE analysis, and spatial safety reasoning.

This project combines modern object detection, segmentation, and ensemble inference strategies to analyze construction-site imagery and generate structured worker-level safety reports.

---

# Current Features

## Worker Detection

- YOLO11-based worker/person detection
- Multi-worker scene analysis
- Bounding-box visualization

---

## PPE Segmentation

Fine-tuned YOLO11 segmentation model for:

- helmet
- vest
---

## Two-Stage PPE Pipeline

Pipeline architecture:

```text
Input Image
    ↓
YOLO11 Person Detection
    ↓
Worker Cropping
    ↓
PPE Segmentation
    ↓
Worker-Level Safety Analysis
````

This improves PPE analysis by focusing segmentation only inside detected worker regions.

---

# Ensemble PPE Inference

The project implements a padding-based ensemble inference strategy.

Instead of relying on a single crop around a worker, the system tests multiple crop paddings:

```text
padding = [0, 10, 20, 40, 60]
```

Each crop is analyzed independently and aggregated into a final worker-level prediction.

This improves robustness for:

* tiny workers
* occlusions
* crowded scenes
* nested detections
* partial PPE visibility

---

# Depth-Aware PPE Filtering

The final `PersonPPE.py` pipeline optionally uses monocular depth estimation to improve worker-level PPE assignment in crowded or overlapping scenes.

Depth is used as a consistency filter after PPE segmentation. For each detected worker, the pipeline estimates relative depth inside the worker box, then compares it with the median depth of each PPE segmentation mask. PPE masks with depth values inconsistent with the target worker can be filtered before aggregation.

This helps in cases where padding includes nearby or nested workers.

---

# Example Outputs

The pipeline generates:

```text
outputs/
├── worker crops & depth maps
├── PPE segmentation masks
├── worker-level annotated images
├── JSON safety summaries
└── human-readable PPE reports
```

Example worker report:

```json
{
  "worker_id": 1,
  "helmet_detected": true,
  "vest_detected": false,
  "violations": [
    "missing_vest"
  ]
}
```

---

# Repository Structure

```text
construction-scene-intelligence/
├── configs/
├── data/
├── outputs/
├── scripts/
├── requirements.txt
└── README.md
```

---

# Script

## Detection

```text
scripts/run_yolo_detection.py
```

Runs YOLO11 object detection on images.

---

## Segmentation

```text
scripts/run_yolo_segmentation.py
```

Runs YOLO11 segmentation and exports polygon masks.

---



## Depth Estimation

```text
scripts/run_depth_estimation.py
'''
---

## Worker-Level PPE Pipeline


```text
scripts/PersonPPE.py
```

Runs:

* person detection
* worker cropping
* PPE segmentation
* worker-level PPE classification

Supports:

* adaptive padding

Generates:

* segmentation visualizations
* JSON summaries
* text reports

---

# Configuration System

The project uses YAML configuration files.

Main config:

```text
configs/ppe_pipeline.yaml
```

Example:

```yaml
input_path: data/sample_images

output_dir: outputs/person_ppe_padding_ensemble

person_model: yolo11n.pt

ppe_model: runs/segment/outputs/training/yolo11n_seg_construction_ppe/weights/best.pt

person_conf: 0.25
ppe_conf: 0.05

ppe_imgsz: 640

small_crop_target_height: 256

padding_values:
  - 0
  - 10
  - 20
  - 40
  - 60

ppe_classes:
  - helmet
  - vest
  - without_helmet
  - without_vest

use_depth_filtering: true
depth_model: DPT_Hybrid
depth_tolerance_std: 1.0
---

# Installation

## Clone Repository

```bash
git clone https://github.com/rg2128/construction-scene-intelligence.git

cd construction-scene-intelligence
```

---

## Create Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Training

## PPE Segmentation Training

```bash
yolo segment train \
model=yolo11n-seg.pt \
data=data/construction_person/data.yaml \
epochs=50 \
imgsz=640 \
batch=8 \
project=outputs/training \
name=yolo11n_seg_construction_ppe
```

---

