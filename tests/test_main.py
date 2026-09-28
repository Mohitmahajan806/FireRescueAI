from pathlib import Path

import cv2
import numpy as np

from main import Detection, extract_person_detections, parse_source, run


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[10, 12, 40, 70], [1, 2, 3, 4]])
    conf = FakeTensor([0.91, 0.99])
    cls = FakeTensor([0, 2])
    id = FakeTensor([7, 8])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self):
        self.calls = []
        self.frames = []

    def track(self, frame, **kwargs):
        self.frames.append(frame.copy())
        self.calls.append(kwargs)
        return [FakeResult()]


class FakeCapture:
    def __init__(self):
        self.frames = [np.zeros((80, 100, 3), dtype=np.uint8), np.zeros((80, 100, 3), dtype=np.uint8)]

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


def test_parse_source_supports_camera_and_file():
    assert parse_source("0") == 0
    assert parse_source("sample_videos/test.mp4") == "sample_videos/test.mp4"


def test_extract_person_detections_filters_to_coco_person_class():
    detections = extract_person_detections([FakeResult()])
    assert detections == [Detection(7, 0.91, 10, 12, 40, 70)]


def test_prerecorded_style_capture_writes_csv_and_uses_bytetrack(tmp_path: Path):
    model = FakeModel()
    result = run(0, model, capture=FakeCapture(), csv_log=True, output_dir=tmp_path, display=False)
    assert result == 0
    assert (tmp_path / "detections.csv").exists()
    assert len((tmp_path / "detections.csv").read_text().splitlines()) == 3
    assert all(call["tracker"] == "bytetrack.yaml" for call in model.calls)
    assert all(call["persist"] is True for call in model.calls)
    assert all(call["classes"] == [0] for call in model.calls)


def test_webcam_frames_are_mirrored_before_inference(tmp_path: Path):
    capture = FakeCapture()
    capture.frames[0][0, 0] = [10, 20, 30]
    model = FakeModel()
    run(0, model, capture=capture, output_dir=tmp_path, display=False)
    assert model.frames[0][0, -1].tolist() == [10, 20, 30]


def test_generated_video_can_be_opened_as_prerecorded_input(tmp_path: Path):
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert frame.shape[:2] == (24, 32)
