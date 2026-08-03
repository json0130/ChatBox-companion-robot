"""
Face enrollment and recognition using facenet-pytorch.

Detector modes (how faces are *located* in the frame):
    "opencv"  — a single OpenCV Haar `detectMultiScale` pass finds every face box.
                Each box is cropped and embedded with InceptionResnetV1 for identity,
                and the *same* box is handed to the emotion detector. One detection
                feeds both pipelines (no separate MTCNN/Haar per model).  [default]
    "mtcnn"   — facenet-pytorch MTCNN detects + landmark-aligns each face before the
                embedding. Higher recognition accuracy, but MTCNN is the locator.

Embeddings are always InceptionResnetV1 (VGGFace2, 512-dim). If facenet-pytorch is
not installed, both modes fall back to a coarse Haar pixel embedding.
Runs on CPU — avoids CUDA version conflicts with the system torch install.

NOTE: enrollment and identification must use the SAME detector mode — a face
enrolled under MTCNN alignment will match poorly against an OpenCV-cropped probe
(and vice-versa). Re-enroll people after switching modes.

Usage:
    fi = FaceIdentifier(detector="opencv")
    fi.enroll("alice", bgr_frame)          # call multiple times to average embeddings
    fi.enroll("bob",   bgr_frame)
    person_id, sim, box = fi.identify(bgr_frame)   # box is (x1,y1,x2,y2) or None
    fi.save("faces.npz")
    fi.load("faces.npz")
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# ── Load facenet-pytorch models (CPU only — RTX 5060 needs sm_120, see notes) ─

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _FACENET_AVAILABLE = True
except ImportError:
    _FACENET_AVAILABLE = False

# Fallback: OpenCV Haar cascade for rough pixel-based identity
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class FaceIdentifier:
    """
    Named-face enrollment and identification.

    Each enrolled person is represented as an averaged 512-dim L2-normalised
    embedding from InceptionResnetV1 (VGGFace2 pretrained).  Identification
    uses cosine similarity; faces below `threshold` are reported as unknown.

    Enrollment is additive — calling enroll() multiple times for the same name
    updates the running average, so you can refine a profile over many frames.
    """

    UNKNOWN = "__unknown__"

    def __init__(
        self,
        threshold: float = 0.75,
        detector:  str   = "opencv",
    ) -> None:
        """
        Args:
            threshold: Minimum cosine similarity to accept a match [0, 1].
                       0.75 is conservative; lower for looser matching.
            detector:  Face locator — "opencv" (single Haar pass shared with the
                       emotion detector) or "mtcnn" (landmark-aligned, more accurate).
        """
        self.threshold = threshold
        self.detector  = detector.lower()
        if self.detector not in ("opencv", "mtcnn"):
            raise ValueError(f"detector must be 'opencv' or 'mtcnn', got {detector!r}")

        # Per-person: averaged embedding and sample count for running average
        self._embeddings: dict[str, np.ndarray] = {}
        self._counts:     dict[str, int]         = {}

        self._device = torch.device("cpu")
        self._mtcnn:     Optional[MTCNN]               = None
        self._mtcnn_all: Optional[MTCNN]               = None  # keep_all=True for multi-face
        self._resnet:    Optional[InceptionResnetV1]   = None
        self._haar:      Optional[cv2.CascadeClassifier] = None

        self._init_backend()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_backend(self) -> None:
        # Haar is always loaded when present: it is the locator for "opencv" mode
        # and the fallback when facenet-pytorch is unavailable.
        if os.path.exists(_HAAR_PATH):
            self._haar = cv2.CascadeClassifier(_HAAR_PATH)

        if _FACENET_AVAILABLE:
            self._resnet = InceptionResnetV1(pretrained="vggface2").eval()
            # MTCNN is only needed when it is the chosen locator — skip it in
            # "opencv" mode so we don't load/run a second detector.
            if self.detector == "mtcnn":
                self._mtcnn = MTCNN(
                    image_size=160, margin=20,
                    keep_all=False,       # single-face path (enroll, identify)
                    device=self._device,
                    post_process=True,
                )
                self._mtcnn_all = MTCNN(
                    image_size=160, margin=20,
                    keep_all=True,        # multi-face path (identify_all)
                    device=self._device,
                    post_process=True,
                )
        elif self._haar is not None:
            print("[FaceID] WARNING — facenet-pytorch not installed; using pixel fallback")
        else:
            print("[FaceID] ERROR — neither facenet-pytorch nor Haar cascade available")

    @property
    def backend(self) -> str:
        if self._resnet is not None:
            # facenet embeddings, located by either Haar (opencv) or MTCNN.
            return f"facenet+{self.detector}"
        if self._haar is not None:
            return "haar-pixel"
        return "none"

    # ── Internal embedding helpers ────────────────────────────────────────────

    def _embed_facenet(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        """Return (embedding_1d, box_xyxy) or (None, None) if no face found."""
        img_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # detect() returns boxes [[x1,y1,x2,y2],...] and probs
        boxes, probs = self._mtcnn.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return None, None

        # Pick the highest-confidence face
        best_idx = int(np.argmax(probs))
        box = tuple(int(v) for v in boxes[best_idx])  # (x1, y1, x2, y2)

        # forward() returns aligned, normalised tensor for the same face
        face_tensor = self._mtcnn(img_rgb)
        if face_tensor is None:
            return None, None

        with torch.no_grad():
            emb = self._resnet(face_tensor.unsqueeze(0))   # [1, 512]
        # L2-normalise so cosine_sim == dot product
        emb_np = emb.cpu().numpy()[0]
        emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)
        return emb_np, box

    def _embed_haar(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        """Pixel-level fallback embedding via resized face crop (grayscale 32×32)."""
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.15,
                                            minNeighbors=4, minSize=(50, 50))
        if len(faces) == 0:
            return None, None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        crop  = cv2.resize(gray[y:y+h, x:x+w], (32, 32)).astype(np.float32)
        emb   = crop.flatten() / (np.linalg.norm(crop) + 1e-8)
        box   = (x, y, x + w, y + h)
        return emb, box

    # ── OpenCV (Haar) locator — shared single detection ───────────────────────

    def _detect_boxes_opencv(
        self, frame_bgr: np.ndarray, scale: float = 1.0, max_faces: int = 4,
    ) -> list[tuple]:
        """One Haar pass → up to `max_faces` boxes (x1,y1,x2,y2), largest first.

        Detection runs on a `scale`-downsized grayscale image for speed; boxes are
        mapped back to original-frame coordinates.
        """
        if self._haar is None:
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, (int(gray.shape[1] * scale),
                                     int(gray.shape[0] * scale)))
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.15,
                                            minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return []
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
        inv = 1.0 / scale
        return [(int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv))
                for (x, y, w, h) in faces]

    def _crop_with_margin(
        self, frame_bgr: np.ndarray, box: tuple, margin: float = 0.2,
    ) -> np.ndarray:
        """Crop `box` from the frame with a relative margin (approximates MTCNN's
        margin so Haar-located crops embed closer to MTCNN-aligned ones)."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(w, x2 + mx), min(h, y2 + my)
        return frame_bgr[y1:y2, x1:x2]

    def _embed_crop_facenet(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Embed an already-cropped face with InceptionResnetV1.

        Applies facenet's fixed image standardization ((x-127.5)/128) — the same
        preprocessing MTCNN(post_process=True) uses — so opencv-located crops and
        mtcnn-aligned faces land in a comparable embedding space.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(rgb, (160, 160)).astype(np.float32)
        face = (face - 127.5) / 128.0
        tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)  # [1,3,160,160]
        with torch.no_grad():
            emb = self._resnet(tensor)
        emb_np = emb.cpu().numpy()[0]
        return emb_np / (np.linalg.norm(emb_np) + 1e-8)

    def _embed_opencv(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        """Single-face path for opencv mode: largest Haar box → facenet embedding."""
        boxes = self._detect_boxes_opencv(frame_bgr, scale=1.0, max_faces=1)
        if not boxes:
            return None, None
        box = boxes[0]
        emb = self._embed_crop_facenet(self._crop_with_margin(frame_bgr, box))
        if emb is None:
            return None, None
        return emb, box

    def _get_embedding_and_box(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[tuple]]:
        # opencv mode with facenet embeddings: one Haar detection, resnet embedding.
        if self.detector == "opencv" and self._resnet is not None and self._haar is not None:
            return self._embed_opencv(frame_bgr)
        if self._resnet is not None:
            return self._embed_facenet(frame_bgr)
        if self._haar is not None:
            return self._embed_haar(frame_bgr)
        return None, None

    # ── Public API ────────────────────────────────────────────────────────────

    def enroll(self, name: str, frame_bgr: np.ndarray) -> bool:
        """
        Add or update the face embedding for `name` from one BGR frame.

        Multiple calls average the embeddings, so enroll() can be called
        across several frames to build a more robust profile.

        Returns True if a face was found and enrolled, False otherwise.
        """
        emb, box = self._get_embedding_and_box(frame_bgr)
        if emb is None:
            return False

        if name in self._embeddings:
            n   = self._counts[name]
            avg = (self._embeddings[name] * n + emb) / (n + 1)
            self._embeddings[name] = avg / (np.linalg.norm(avg) + 1e-8)
            self._counts[name]     = n + 1
        else:
            self._embeddings[name] = emb
            self._counts[name]     = 1

        return True

    def enroll_from_camera(
        self,
        name: str,
        camera_index: int = 0,
        n_captures: int = 15,
        countdown: int = 3,
    ) -> bool:
        """
        Interactive enrollment: opens webcam, counts down, captures n_captures
        frames, and averages the embeddings.  Shows a live OpenCV window.

        Returns True if at least one frame was successfully enrolled.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[FaceID] Cannot open camera {camera_index}")
            return False

        captured = 0
        frame_n  = 0
        fps_est  = cap.get(cv2.CAP_PROP_FPS) or 30
        count_frames = int(countdown * fps_est)

        print(f"[FaceID] Enrolling '{name}' — look at the camera, capturing {n_captures} frames …")

        try:
            while captured < n_captures:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_n += 1

                # Countdown phase
                if frame_n <= count_frames:
                    remaining = countdown - int(frame_n / fps_est)
                    overlay = frame.copy()
                    cv2.putText(overlay, f"Enrolling: {name}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 255, 0), 2)
                    cv2.putText(overlay, f"Starting in {remaining}s — look at camera",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.imshow(f"Enroll: {name}", overlay)
                    cv2.waitKey(1)
                    continue

                # Capture phase
                if self.enroll(name, frame):
                    captured += 1
                    pct  = int(captured / n_captures * 100)
                    bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    overlay = frame.copy()
                    cv2.putText(overlay, f"Capturing '{name}': {captured}/{n_captures}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
                    cv2.putText(overlay, f"[{bar}] {pct}%",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    cv2.imshow(f"Enroll: {name}", overlay)
                    cv2.waitKey(1)
                else:
                    overlay = frame.copy()
                    cv2.putText(overlay, f"No face found — adjust position",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                    cv2.imshow(f"Enroll: {name}", overlay)
                    cv2.waitKey(1)

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if captured > 0:
            print(f"[FaceID] '{name}' enrolled with {self._counts[name]} total samples.")
        else:
            print(f"[FaceID] Enrollment failed — no faces captured for '{name}'.")
        return captured > 0

    def identify(
        self, frame_bgr: np.ndarray
    ) -> tuple[Optional[str], float, Optional[tuple]]:
        """
        Identify the most prominent face in frame_bgr.

        Returns:
            person_id  — matched name, or None if no face / no match above threshold
            similarity — cosine similarity of best match (0–1); 0.0 if no face
            box        — (x1, y1, x2, y2) in pixels, or None if no face found
        """
        if not self._embeddings:
            emb, box = self._get_embedding_and_box(frame_bgr)
            return None, 0.0, box

        emb, box = self._get_embedding_and_box(frame_bgr)
        if emb is None:
            return None, 0.0, None

        best_name = None
        best_sim  = -1.0
        for name, stored in self._embeddings.items():
            sim = _cosine_sim(emb, stored)
            if sim > best_sim:
                best_sim  = sim
                best_name = name

        if best_sim >= self.threshold:
            return best_name, best_sim, box
        return None, best_sim, box   # below threshold → unknown face present

    def identify_all(
        self,
        frame_bgr:  np.ndarray,
        max_faces:  int   = 4,
        scale:      float = 0.5,
    ) -> list[tuple[Optional[str], float, tuple]]:
        """
        Identify every face visible in the frame.

        Returns:
            List of (person_id, similarity, box) — one entry per detected face.
            person_id is None for faces below the similarity threshold.
            Faces are ordered by detection confidence (highest first).
            Returns [] when no faces are found.
        """
        # ── OpenCV mode: one Haar pass locates every face, facenet embeds each ──
        if self.detector == "opencv" and self._resnet is not None and self._haar is not None:
            boxes = self._detect_boxes_opencv(frame_bgr, scale=scale, max_faces=max_faces)
            results = []
            for box in boxes:
                emb = self._embed_crop_facenet(self._crop_with_margin(frame_bgr, box))
                if emb is None:
                    continue
                if not self._embeddings:
                    results.append((None, 0.0, box))
                    continue
                best_name, best_sim = None, -1.0
                for name, stored in self._embeddings.items():
                    s = _cosine_sim(emb, stored)
                    if s > best_sim:
                        best_sim, best_name = s, name
                pid = best_name if best_sim >= self.threshold else None
                results.append((pid, float(best_sim), box))
            return results

        # ── Haar fallback: single face only ──────────────────────────────────
        if self._mtcnn_all is None:
            if self._haar is None:
                return []
            emb, box = self._embed_haar(frame_bgr)
            if box is None:
                return []
            if not self._embeddings:
                return [(None, 0.0, box)]
            best_name, best_sim = None, -1.0
            for name, stored in self._embeddings.items():
                s = _cosine_sim(emb, stored)
                if s > best_sim:
                    best_sim, best_name = s, name
            pid = best_name if best_sim >= self.threshold else None
            return [(pid, best_sim, box)]

        # ── FaceNet path ─────────────────────────────────────────────────────
        # Downscale for faster MTCNN (boxes scaled back to original coords)
        if scale != 1.0:
            h0, w0 = frame_bgr.shape[:2]
            small  = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)))
        else:
            small  = frame_bgr

        img_rgb = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        boxes, probs = self._mtcnn_all.detect(img_rgb)
        if boxes is None or len(boxes) == 0:
            return []

        face_tensors = self._mtcnn_all(img_rgb)   # [N,3,160,160] or [3,160,160] or None
        if face_tensors is None:
            return []
        if face_tensors.dim() == 3:               # single face edge case
            face_tensors = face_tensors.unsqueeze(0)

        with torch.no_grad():
            embs = self._resnet(face_tensors)     # [N, 512]

        results = []
        n = min(len(boxes), embs.shape[0], max_faces)
        for i in range(n):
            prob = float(probs[i]) if probs[i] is not None else 0.0
            if prob < 0.90:                       # skip low-confidence detections
                continue
            # Scale box coordinates back to original frame size
            inv = 1.0 / scale
            box = tuple(int(v * inv) for v in boxes[i])

            emb_np = embs[i].cpu().numpy()
            emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)

            if not self._embeddings:
                results.append((None, 0.0, box))
                continue

            best_name, best_sim = None, -1.0
            for name, stored in self._embeddings.items():
                s = _cosine_sim(emb_np, stored)
                if s > best_sim:
                    best_sim, best_name = s, name

            pid = best_name if best_sim >= self.threshold else None
            results.append((pid, float(best_sim), box))

        return results

    def known_people(self) -> list[str]:
        return sorted(self._embeddings.keys())

    def forget(self, name: str) -> None:
        self._embeddings.pop(name, None)
        self._counts.pop(name, None)

    def rename(self, old_name: str, new_name: str) -> bool:
        """Move a person's enrolled embedding from `old_name` to `new_name` — used
        when a provisionally-enrolled face (e.g. 'guest_3') tells us their real name.
        If `new_name` already exists, the two profiles are averaged (weighted by
        sample count) so no enrolment work is lost. Returns True if it existed."""
        if old_name == new_name or old_name not in self._embeddings:
            return False
        emb_old = self._embeddings.pop(old_name)
        n_old = self._counts.pop(old_name, 1)
        if new_name in self._embeddings:
            emb_new, n_new = self._embeddings[new_name], self._counts.get(new_name, 1)
            avg = (emb_new * n_new + emb_old * n_old) / (n_new + n_old)
            self._embeddings[new_name] = avg / (np.linalg.norm(avg) + 1e-8)
            self._counts[new_name]     = n_new + n_old
        else:
            self._embeddings[new_name] = emb_old
            self._counts[new_name]     = n_old
        return True

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save enrolled embeddings and counts to a .npz file."""
        np.savez(
            path,
            names=np.array(list(self._embeddings.keys())),
            embeddings=np.array(list(self._embeddings.values())),
            counts=np.array([self._counts[n] for n in self._embeddings]),
        )
        print(f"[FaceID] saved {len(self._embeddings)} people → {os.path.abspath(path)}")

    def load(self, path: str) -> bool:
        """Load enrolled embeddings from a .npz file. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=False)
            names      = data["names"].tolist()
            embeddings = data["embeddings"]
            counts     = data["counts"].tolist()
            self._embeddings = {n: e for n, e in zip(names, embeddings)}
            self._counts     = {n: int(c) for n, c in zip(names, counts)}
            print(f"[FaceID] loaded {len(self._embeddings)} people from {path}")
            return True
        except Exception as exc:
            print(f"[FaceID] load failed: {exc}")
            return False
