# utils/

Training/experiment scripts for emotion recognition and old Jetson Docker setups.
Nothing in `CHATBOX-DEMO_V4/` or `CHATBOX_V5-dev/` imports from here.

Large binaries are **not tracked in git** (`*.pt`, `*.whl` are ignored). Put them back here if you need them:

| File | Used by | Where to get it |
|---|---|---|
| `emotion_understanding/yolo_v8/yolov8m.pt` | `yolo_v8/train_script.py` | auto-downloaded by `ultralytics` on first `YOLO("yolov8m.pt")` |
| `emotion_understanding/yolo_v8/yolov8m-face.pt` | `yolo_v8/test_script.py` | YOLOv8-face weights (community release); also in git history before this commit |
| `dockerV1/torchvision-0.12.0-cp38-cp38-manylinux2014_aarch64.whl` | `dockerV1/dockerfile` (`COPY`) | build for Jetson (aarch64, py3.8); also in git history before this commit |

To pull one back out of history: `git log --oneline -- <path>` then `git show <commit>^:<path> > <path>`.
