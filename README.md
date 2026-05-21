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
- without_helmet
- without_vest

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

# Example Outputs

The pipeline generates:

```text
outputs/
├── worker crops
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

# Key Scripts

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

## Worker-Level PPE Pipeline

```text
scripts/run_person_ppe_pipeline.py
```

Runs:

* person detection
* worker cropping
* PPE segmentation
* worker-level PPE classification

Supports:

* adaptive padding
* crop resizing
* configurable inference

---

## Padding Ensemble PPE Pipeline

```text
scripts/run_person_ppe_padding_ensemble.py
```

Runs multiple crop paddings per worker and aggregates predictions into a final ensemble decision.

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
```

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

# Running the Ensemble PPE Pipeline

```bash
python scripts/run_person_ppe_padding_ensemble.py \
--config configs/ppe_pipeline.yaml
```

---

# Generated Outputs

```text
outputs/person_ppe_padding_ensemble/
├── *_crop.jpg
├── *_seg.jpg
├── *_pipeline_ensemble.jpg
├── person_ppe_padding_ensemble_summary.json
└── ppe_ensemble_report.txt
```

---

# Current Research / Engineering Ideas

## Short-Term

* Streamlit visualization dashboard
* Video support
* Multi-frame tracking
* Worker trajectory analysis
* PPE temporal consistency

---

## Mid-Term

* SAM2 integration
* Depth estimation
* Spatial hazard reasoning
* Scene graph generation
* Vision-language scene understanding

---

## Long-Term

* Autonomous construction-site inspection
* Agentic multimodal reasoning
* Safety analytics over time
* Real-time deployment pipelines

---

# Technologies Used

* YOLO11
* OpenCV
* PyTorch
* Ultralytics
* YAML configuration pipelines
* Python

---

# Author

Rahul Garg

Systems neuroscience → multimodal AI / computer vision transition project focused on scene understanding, safety analysis, and spatial reasoning systems.

```
