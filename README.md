# Tennis Match Analysis System

A computer-vision pipeline that detects players and the ball from broadcast tennis footage, localizes the court, and renders a tactical mini-court view for quick match insights.

## Features
- YOLOv8-based player and ball detection with tracking.
- ResNet50 model estimates 14 court keypoints for court geometry recovery.
- Mini-court renderer that maps real-world positions to a 2D schematic with player and ball traces.
- Ball trajectory interpolation to smooth occlusions and missed frames.
- Outputs annotated video with bounding boxes, keypoints, and a live mini-court overlay.

## Repository structure
- `main.py` — end-to-end pipeline orchestration (read video → detect → map to mini-court → save video).
- `trackers/` — player and ball trackers built on YOLO (`PlayerTracker`, `BallTracker`).
- `court_line_detector/` — ResNet50 model wrapper for court keypoint inference and overlay utilities.
- `mini_court/` — mini-court geometry, coordinate conversion, and drawing utilities.
- `utils/` — video I/O, bounding-box helpers, and pixel/meter conversions.
- `constants/` — court dimensions and player height assumptions used for scaling.
- `models/` — expected location for trained weights (YOLO player/ball, court keypoints).
- `input_videos/` & `output_videos/` — sample inputs and generated outputs.
- `tracker_stubs/` — cached detections to skip re-running heavy models during iteration.

## Requirements
- Python 3.9+ (GPU with CUDA recommended for speed).
- Install pinned dependencies:

```bash
pip install -r requirements.txt
```

- If you need a specific CUDA build of PyTorch, install it first from [pytorch.org](https://pytorch.org/get-started/locally/) and then install the rest:

```bash
pip install ultralytics opencv-python pandas numpy matplotlib pillow
```

## Models & assets
Place the following weights in the repository (paths assumed by `main.py`):
- Player detector: `yolo11x.pt` (already included here).
- Ball detector: `models/tennis_ball_best.pt`.
- Court keypoints: `models/keypoints_model_50.pth`.

You can swap in other YOLO weights (e.g., `yolo11n.pt`) if desired; update the paths in `main.py` accordingly.

## Quick start
1. Add your match footage to `input_videos/` and update `input__video_path` in `main.py`.
2. (Optional) Toggle cached detections: set `read_from_stub=True` to reuse `tracker_stubs/*.pkl`, or `False` to re-run the detectors and refresh the stubs.
3. Run the pipeline:
   ```bash
   python main.py
   ```
4. Find the annotated output in `output_videos/` (default `output3.avi`).

## How it works
1. **Video ingest** — frames are loaded and FPS preserved (`utils/video_utils.py`).
2. **Detection & tracking** — YOLOv8 tracks players; a YOLO ball model detects the tennis ball. Detections can be cached to pickle stubs for faster reruns.
3. **Court keypoints** — a ResNet50 regressor predicts 14 court landmarks, which are scaled back to the original resolution.
4. **Mini-court mapping** — player feet and ball positions are converted from pixels to meters using player height priors, then mapped onto a scaled 2D court (`mini_court.py`).
5. **Interpolation & visualization** — missing ball detections are interpolated; bounding boxes, keypoints, and mini-court trajectories are drawn on each frame.
6. **Export** — the annotated frames are written back to video with the original FPS.

## Using cached detections
- Player stubs: `tracker_stubs/player_detection*.pkl`
- Ball stubs: `tracker_stubs/ball_detection*.pkl`

Set `read_from_stub=True` in `PlayerTracker.detect_frames` / `BallTracker.detect_frames` calls (as shown in `main.py`) to avoid re-running the models. To regenerate stubs on new footage, set `read_from_stub=False` and provide a `stub_path`; the pipeline will save fresh detections.

## Training notes
- `tennis_ball_detector_training.ipynb` contains a YOLO training workflow for the ball detector.
- Court keypoint model loading expects a 14-point (x,y) regressor; replace `models/keypoints_model_50.pth` if you retrain.

## Tips
- GPU execution significantly speeds up YOLO and ResNet inference.
- If court keypoints look misaligned, verify the input aspect ratio and consider resizing logic consistent with the training setup.
- Adjust YOLO confidence thresholds in `trackers/ball_tracker.py` and `trackers/player_tracker.py` for your footage quality.
