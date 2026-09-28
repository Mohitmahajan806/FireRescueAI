# FireRescueAI

FireRescueAI is a camera-based fire-rescue **research prototype**. It uses OpenCV for video capture and display, Ultralytics YOLO for pretrained person detection, and Ultralytics ByteTrack for frame-to-frame tracking.

**Safety boundary:** every detection is unverified and requires human review. This is not an emergency response system. It does not control a drone, navigate autonomously, dispatch responders, or make rescue decisions. RGB footage must not be interpreted as detecting people through smoke, darkness, heat, or other obscurants. False positives, missed detections, and tracking ID switches are expected research observations.

## Project files

- `main.py`: webcam/video capture, YOLO person filtering (COCO class `0`), ByteTrack tracking, annotated display, optional recording, CSV logging, and `s`-key screenshots.
- `dashboard.py`: local read-only HTML dashboard for reviewing CSV rows and captured screenshot filenames.
- `thermal_adapter.py`: interface-only boundary for a future thermal camera adapter; no thermal implementation is included.
- `tests/test_main.py`: tests detection filtering, required ByteTrack arguments, CSV output, and prerecorded-video opening.
- `requirements.txt`: pinned major-version ranges for OpenCV, Ultralytics, and pytest.
- `sample_videos/`: place research footage here; no media is bundled.
- `screenshots/`: runtime screenshots captured with the `s` key.
- `output_recordings/`: annotated MP4 and CSV outputs.

## macOS setup in VS Code

1. Install Python 3.12 and confirm it is available:

   ```bash
   python3.12 --version
   ```

2. Open `/Users/amananand/Desktop/Drone` as the VS Code folder. Install the Microsoft Python extension if it is not already installed.

3. Create and activate the virtual environment:

   ```bash
   cd /Users/amananand/Desktop/Drone
   /opt/homebrew/bin/python3.12 -m venv .venv
   source .venv/bin/activate
   python --version
   ```

   On an Intel Mac, replace `/opt/homebrew/bin/python3.12` with the path returned by `which python3.12`.

4. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. In VS Code, select `.venv/bin/python` with **Python: Select Interpreter**. The first YOLO run downloads the selected pretrained weights, so allow network access for that first run.

### Camera permissions

If the webcam cannot open, go to **System Settings > Privacy & Security > Camera**, enable camera access for VS Code (or the terminal application launching Python), fully quit and reopen that application, then retry. Close other applications that may hold the camera. For a no-camera check, use a prerecorded file instead.

## Run the detector

Activate the environment first:

```bash
cd /Users/amananand/Desktop/Drone
source .venv/bin/activate
```

Webcam index `0`:

```bash
python main.py --source 0
```

Prerecorded video:

```bash
python main.py --source sample_videos/research_clip.mp4
```

Record annotated output and log detections:

```bash
python main.py --source sample_videos/research_clip.mp4 --record --csv
```

Controls while the preview window is focused:

- `q`: stop processing.
- `s`: save the current annotated frame under `screenshots/`.

Outputs are written to `output_recordings/annotated_output.mp4` and `output_recordings/detections.csv`. The detector passes `classes=[0]` to YOLO and calls `model.track(..., tracker="bytetrack.yaml", persist=True)` on every consecutive frame.

Webcam input is mirrored horizontally before inference so the live preview behaves like a standard self-view. Prerecorded video remains in its original orientation.

## Review dashboard

After producing a CSV, start the local dashboard in another terminal:

```bash
source .venv/bin/activate
python dashboard.py
```

Open `http://127.0.0.1:8765` in a browser. It summarizes detection rows, unique track IDs, and captured screenshots. It is a review aid, not an alerting or decision system.

## Tests with prerecorded-style input

Run the tests with:

```bash
source .venv/bin/activate
python -m pytest -q
```

The suite creates a tiny MP4 in a temporary directory to verify that OpenCV can open prerecorded footage. It also uses a fake model to verify the processing loop without requiring a live camera or downloading model weights.

When evaluating real footage, record at least these research observations:

- **False positives:** background objects, reflections, posters, mannequins, or smoke-like patterns incorrectly labeled as people.
- **Missed detections:** people not detected because of scale, occlusion, lighting, pose, motion blur, or scene composition.
- **Tracking ID switches:** one person receiving a new ID or two people temporarily sharing identity after occlusion or close interaction.

Treat every row as an observation to review against the original footage. Do not use this prototype to infer that a person is safe, trapped, deceased, or present through smoke, or to trigger an emergency action automatically.
