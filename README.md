# License Plate Detection and Recognition

This project detects vehicle license plates in video or camera footage, reads the plate text with OCR, and uses plate-format rules and repeated observations to improve recognition stability. License plate detection is performed with Ultralytics YOLO; text recognition uses EasyOCR.

## Project files

| File | Purpose |
| --- | --- |
| `detect_live.py` | Runs detection, OCR, and tracking on a video or camera feed. |
| `train.py` | Fine-tunes a YOLO model using the local license-plate dataset. |
| `algorithm.txt` | Original high-level algorithm outline. |

## Algorithm

1. Read frames from the input video or camera.
2. Pass each frame through the YOLO model to detect license plates.
3. For each detected plate, crop the region and prepare it for OCR:
   - Correct the plate angle when detectable.
   - Reduce the appearance of motion blur with unsharp masking.
   - Upscale small plate regions.
   - Convert to grayscale and improve local contrast.
   - Apply light sharpening and adaptive denoising.
   - Convert the result to BGR and add padding for OCR.
4. Run OCR on the processed plate image. For square-shaped regions, also try reading the top and bottom halves as a two-line plate.
5. Clean and validate OCR text using the expected Indian registration format:
   - The plate contains 10 alphanumeric characters.
   - The first two characters are a recognized state or union-territory code.
   - Position-based character corrections are applied to the state, district, series, and registration-number portions.
6. Associate each reading with the tracked plate detection across frames.
7. Keep recent readings for each track and use repeated observations to select a stable plate result.
8. Draw the detection box, track ID, and recognized text on the displayed frame. At the end, print the most frequent reading for each track.

## Requirements

- Python 3.9 or later
- A CUDA-capable NVIDIA GPU and compatible CUDA-enabled PyTorch installation for the current `detect_live.py` configuration. The script initializes EasyOCR with GPU enabled and requests YOLO device `0`.
- A local copy of the dataset for training. Dataset contents are intentionally excluded from version control.
- A trained license-plate detector checkpoint for inference.

Install the Python packages in a virtual environment. Install PyTorch using the command recommended for your CUDA version from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python easyocr numpy
```

## Dataset setup

Place the dataset in `dataset/` with the following YOLO directory structure:

```text
dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The supplied `dataset/dataset.yaml` currently points to a machine-specific directory. Before training on another computer, update its `path` value to the absolute path of that computer's `dataset/` directory. Keep the `train` and `val` entries relative to that path.

## Train the detector

From the project root, run:

```bash
python train.py
```

Training runs for up to 50 epochs and saves output under `runs/detect/license_plate_detector/`. The best checkpoint is written to `runs/detect/license_plate_detector/weights/best.pt`. Ultralytics may download the pretrained `yolov8n.pt` weights if they are not already available locally.

## Run detection and OCR

The inference script expects its checkpoint at `runs/detect/license_plate_detector3/weights/best.pt`. After training, either copy the checkpoint produced by `train.py` to that location or update `model_path` in `detect_live.py` to point to the checkpoint you trained.

Run with a video file:

```bash
python detect_live.py path/to/video.mp4
```

Run with the default camera (camera index `0`):

```bash
python detect_live.py
```

Press `q` in the video window to stop processing. The script displays tracked plates and prints a summary of recognized plates when it exits. The current script assumes a camera or video source with a usable frame rate and a CUDA-enabled GPU.

## Version-control exclusions

The repository ignore rules exclude datasets, test directories, generated plate crops, training outputs, caches, virtual environments, model binaries, local media, and PDF or PowerPoint files. Keep large or locally licensed data and model artifacts outside the Git repository.
