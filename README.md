# Shoplifting Detection Project

This repository contains a YOLOv8-based computer vision project for detecting shoplifting or suspicious shoplifter activity from images and video sources.

The workspace currently appears to be centered around training a custom object detection model using UltraLytics YOLOv8 and running inference with a saved best checkpoint.

## Project Files

The main files currently present in the workspace are:

- `train_model.py` – Training entry point. Creates a YOLO model object from the configured base model (`yolov8n.pt`) and runs `model.train(...)` with dataset, hardware, project, and augmentation settings.
- `test_model.py` – Inference/testing entry point. Loads a trained YOLO weights file, runs `model.predict(...)` against a source video/image path, and writes JSON Lines detection logs.
- `extract_frames.py` – Utility script that extracts frames from downloaded shoplifting videos into a YOLO-style image dataset structure under `data/images/train` and `data/images/val`.
- `shoplift_data.yaml` – Dataset configuration file describing the custom dataset roots and class names.
- `yolov8n.pt` – Base YOLOv8 weights used as the pretrained backbone/model source.

## Dataset Layout

The custom dataset is expected to live under the workspace as a YOLO-style directory structure:

```text
data/
  images/
    train/
      non_shoplifter/
      shoplifter/
    val/
      non_shoplifter/
      shoplifter/
```

The YAML file currently points to a dataset root that was previously located at:

```text
C:\Users\Focus Computer\Downloads\shoplifting.v1i.yolov8
```

The dataset YAML currently declares:

```yaml
path: C:\Users\Focus Computer\Downloads\shoplifting.v1i.yolov8
train: train/images
val: val/images
nc: 2
names:
  0: not-steal
  1: steal
```

The class names are therefore configured for a 2-class task, even though the current working directory uses folder names like `non_shoplifter` and `shoplifter`.

## Training Configuration

The training script defaults to a CPU-only training configuration unless overridden from the command line:

```bash
python train_model.py --data data.yaml --model yolov8n.pt --epochs 50 --imgsz 416 --batch 4 --device cpu
```

Its main training options include:

- `epochs = 50`
- `imgsz = 416`
- `batch = 4`
- `device = cpu`
- deterministic training enabled
- HSV augmentation, translation, scale, horizontal flip, mosaic, and mixup enabled

The generated training artifacts are written under `output/` and also under the `runs/` directory by Ultralytics.

## Inference and Detection Output

The runtime inference pipeline reads the existing trained model from `runs/detect/train8/weights/best.pt` by default and writes inference results to a `project` output folder. Testing does not require training again, but you must provide the path to an existing input image or video with `--source`.

The detection log writer stores structured JSON Lines records under:

```text
output/detection_logs/
```

The detection logs appear to have names like:

- `detections_20260514_205219.jsonl`
- `detections_20260514_205434.jsonl`
- `detections_20260514_205536.jsonl`
- `detections_20260514_210048.jsonl`

Each JSON Lines row stores things such as:

- source video path
- input shape
- list of detections
- class ID, canonicalized class name, confidence, and bounding box coordinates

## Output and Run Artifacts

The project currently contains a number of training and prediction outputs:

```text
output/
  detect/
  detect2/
  detect3/
  detect4/
  detection_logs/
  train/
  train2/
  train3/
  train4/
  train5/
  train6/
```

The `runs/` directory contains multiple curated prediction and training run folders, including multiple `predict*` folders and `train*` folders. The visible folders include:

```text
runs/
  detect/
    predict/
    predict2/
    predict3/
    ...
    train/
    train2/
    train3/
    train4/
    train5/
    train6/
    train7/
    train8/
```

Several training folders expose YAML configuration plus weight files (`best.pt` and `last.pt`), for example:

```text
output/train4/weights/
output/train6/weights/
runs/detect/train4/weights/
runs/detect/train5/weights/
runs/detect/train6/weights/
runs/detect/train7/weights/
runs/detect/train8/weights/
```

## Top-Level Folders

The current workspace root contains:

```text
.vscode/
data/
output/
runs/
train/
train2/
```

The `train/` and `train2/` folders appear to be model run directories or earlier executed training artifacts, containing `args.yaml` and weight folders.

## Implementation Notes

This project is a detection pipeline with the following high-level flow:

1. Prepare/collect image samples or frames.
2. Organize them according to the YOLO dataset schema.
3. Train a YOLOv8 model using the dataset definition.
4. Save model checkpoints under a project output folder.
5. Run inference using the saved model weights.
6. Log detection predictions to JSON Lines files for downstream analysis.

## Typical Commands

Below are example commands that match the visible scripts:

```bash
python train_model.py
```

```bash
python test_model.py
```

By default, `test_model.py` uses `C:\Users\Focus Computer\Downloads\V_sample_6_.gif` as its source. To test a different image or video, pass it with `--source`.

## Status

This is a custom YOLOv8 object detection experiment for detecting shoplifting-related classifications, using a two-class dataset configuration and model exports generated by Ultralytics.
