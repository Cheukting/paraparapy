"""Recorded landmark clips: save/load a sequence of :class:`PoseSample`.

Recordings are the ground truth for Steps 3–4: they verify MediaPipe's
world-coordinate axes, tune the One Euro filter, and become test fixtures for
feature extraction and scoring. They are written by the guided recorder
(``uv run parapara-record``).

Format: one ``.npz`` file per clip (NumPy's zip of arrays, loadable without
pickle, so a recording file can never execute code):

=================  =================  ========================================
array              shape / dtype      meaning
=================  =================  ========================================
``frame_index``    (N,) int64         camera frame counter
``captured_ns``    (N,) int64         ``time.monotonic_ns()`` at camera read
``processed_ns``   (N,) int64         ... when landmarks were ready
``pose_ms``        (N,) float32       pose inference time
``has_pose``       (N,) bool          whether a person was detected
``pose_image``     (N, 33, 5) f32     normalized image landmarks (NaN if none)
``pose_world``     (N, 33, 5) f32     world landmarks in metres (NaN if none)
``metadata``       () str             JSON: clip id, bpm, beat times, ...
=================  =================  ========================================

Only landmarks are stored, **never camera images**: recordings are small and
contain no pictures of the player.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from parapara.pose.landmarks import NUM_POSE_LANDMARKS, PoseSample

FORMAT_VERSION = 1
_POSE_SHAPE = (NUM_POSE_LANDMARKS, 5)


@dataclass(frozen=True)
class Recording:
    """A clip of landmark samples plus free-form JSON metadata.

    ``metadata`` must contain ``frame_size`` (``[width, height]``) so samples
    can be rebuilt; the recorder also stores the clip id, tempo, beat times
    and camera details.
    """

    frame_index: npt.NDArray[np.int64]
    captured_ns: npt.NDArray[np.int64]
    processed_ns: npt.NDArray[np.int64]
    pose_ms: npt.NDArray[np.float32]
    has_pose: npt.NDArray[np.bool_]
    pose_image: npt.NDArray[np.float32]
    pose_world: npt.NDArray[np.float32]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame_index)

    @classmethod
    def from_samples(cls, samples: list[PoseSample], metadata: dict[str, Any]) -> Self:
        n = len(samples)
        nan_pose = np.full(_POSE_SHAPE, np.nan, np.float32)
        meta = dict(metadata)
        if samples and "frame_size" not in meta:
            meta["frame_size"] = list(samples[0].frame_size)
        return cls(
            frame_index=np.array([s.frame_index for s in samples], np.int64).reshape(n),
            captured_ns=np.array([s.captured_ns for s in samples], np.int64).reshape(n),
            processed_ns=np.array([s.processed_ns for s in samples], np.int64).reshape(n),
            pose_ms=np.array([s.pose_ms for s in samples], np.float32).reshape(n),
            has_pose=np.array([s.has_pose for s in samples], np.bool_).reshape(n),
            pose_image=np.array(
                [nan_pose if s.pose_image is None else s.pose_image for s in samples], np.float32
            ).reshape(n, *_POSE_SHAPE),
            pose_world=np.array(
                [nan_pose if s.pose_world is None else s.pose_world for s in samples], np.float32
            ).reshape(n, *_POSE_SHAPE),
            metadata=meta,
        )

    def samples(self) -> list[PoseSample]:
        """Rebuild the :class:`PoseSample` objects (e.g. to replay a clip)."""
        width, height = self.metadata.get("frame_size", (0, 0))
        out = []
        for i in range(len(self)):
            has = bool(self.has_pose[i])
            out.append(
                PoseSample(
                    frame_index=int(self.frame_index[i]),
                    captured_ns=int(self.captured_ns[i]),
                    processed_ns=int(self.processed_ns[i]),
                    frame_size=(int(width), int(height)),
                    pose_image=self.pose_image[i].copy() if has else None,
                    pose_world=self.pose_world[i].copy() if has else None,
                    pose_ms=float(self.pose_ms[i]),
                )
            )
        return out

    @property
    def person_found_pct(self) -> float:
        return 100.0 * float(self.has_pose.mean()) if len(self) else 0.0

    @property
    def duration_s(self) -> float:
        return (int(self.captured_ns[-1]) - int(self.captured_ns[0])) / 1e9 if len(self) > 1 else 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format_version": FORMAT_VERSION, **self.metadata}
        np.savez_compressed(
            path,
            frame_index=self.frame_index,
            captured_ns=self.captured_ns,
            processed_ns=self.processed_ns,
            pose_ms=self.pose_ms,
            has_pose=self.has_pose,
            pose_image=self.pose_image,
            pose_world=self.pose_world,
            metadata=np.array(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            version = meta.pop("format_version", None)
            if version != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported recording format_version {version!r}")
            return cls(
                frame_index=data["frame_index"],
                captured_ns=data["captured_ns"],
                processed_ns=data["processed_ns"],
                pose_ms=data["pose_ms"],
                has_pose=data["has_pose"],
                pose_image=data["pose_image"],
                pose_world=data["pose_world"],
                metadata=meta,
            )
