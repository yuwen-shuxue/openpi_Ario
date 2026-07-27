"""Ario streaming dataset that reads directly from OSS without pre-conversion."""

from __future__ import annotations

import io
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ACTION_DIM = 31
PT_FILES = ["eef_torso.pt", "head.pt", "eef_left.pt", "gripper_cmd.pt", "eef_right.pt"]
IMAGE_SIZE = (320, 240)

CAMERA_VIEWS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DISK_CACHE_FORMAT_VERSION = 2


@dataclass
class ArioConfig:
    s3_prefixes: str = ""
    s3_endpoint: str = "https://oss-cn-wulanchabu-internal.aliyuncs.com"
    video_downsample_rate: int = 6
    min_frames: int = 1885
    image_size: tuple[int, int] = IMAGE_SIZE
    task: str = "fold clothes"
    load_instructions: bool = False
    skip_video: bool = False
    cache_size: int = 32
    max_episodes: int | None = None
    disk_cache_dir: str = "/tmp/ario_disk_cache"
    disk_cache_max_gb: float = 200.0
    multi_view: bool = True


class ArioStreamingDataset:
    """Dataset that streams Ario-format episodes directly from S3/OSS.

    Each __getitem__ returns a single training sample with action chunking applied.
    Episodes are cached in an LRU manner to avoid redundant downloads.
    """

    def __init__(self, config: ArioConfig, action_horizon: int):
        self._config = config
        self._action_horizon = action_horizon
        self._s3 = None
        self._pid: int | None = None

        # LRU cache for decoded episodes: ep_key -> (state_action, frames_dict)
        self._cache: OrderedDict[str, tuple[np.ndarray, dict[str, list[np.ndarray]]]] = OrderedDict()
        self._cache_size = config.cache_size

        # Per-episode instructions loaded from instructions.json
        self._instructions: dict[str, str] = {}

        # Discover episodes and build global frame index
        episodes = self._discover_episodes()
        if config.max_episodes is not None:
            episodes = episodes[: config.max_episodes]
        self._episodes = episodes  # list of (bucket, prefix)

        # Build frame index: for each episode, count usable (downsampled) frames.
        # We need to download .pt to know the length, so do a lightweight pass.
        self._episode_lengths: list[int] = []  # downsampled frame count per episode
        self._cumulative: list[int] = []  # cumulative sum for global index lookup
        self._build_index()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_s3"] = None
        state["_pid"] = None
        state["_cache"] = OrderedDict()
        return state

    def _get_s3(self):
        # Recreate client after fork (pid changes)
        pid = os.getpid()
        if self._s3 is None or self._pid != pid:
            import boto3
            from botocore.config import Config

            self._pid = pid
            ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
            sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            cfg = Config(s3={"addressing_style": "virtual"}, signature_version="s3v4")
            kwargs = {}
            if self._config.s3_endpoint:
                kwargs["endpoint_url"] = self._config.s3_endpoint
            self._s3 = boto3.client(
                "s3",
                aws_access_key_id=ak,
                aws_secret_access_key=sk,
                region_name="cn-wulanchabu",
                config=cfg,
                **kwargs,
            )
            self._cache.clear()
        return self._s3

    def _discover_episodes(self) -> list[tuple[str, str]]:
        s3 = self._get_s3()
        episodes: set[tuple[str, str]] = set()
        for uri in self._config.s3_prefixes.split(","):
            uri = uri.strip()
            if not uri:
                continue
            bucket, prefix = self._parse_s3_uri(uri)
            paginator = s3.get_paginator("list_objects_v2")
            discovered_views: dict[str, set[str]] = {}
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if self._config.multi_view:
                        for cam in CAMERA_VIEWS:
                            suffix = f"raw_video/{cam}.mp4"
                            if key.endswith(suffix):
                                ep_prefix = key[: -len(suffix)]
                                discovered_views.setdefault(ep_prefix, set()).add(cam)
                                break
                    elif key.endswith("/video.mp4"):
                        ep_prefix = key[: key.rfind("/video.mp4") + 1]
                        episodes.add((bucket, ep_prefix))

            if self._config.multi_view:
                required_views = set(CAMERA_VIEWS)
                for ep_prefix, views in discovered_views.items():
                    if views == required_views:
                        episodes.add((bucket, ep_prefix))
                    else:
                        missing = sorted(required_views - views)
                        print(
                            f"[WARN] Skipping incomplete multi-view episode {ep_prefix}: "
                            f"missing {missing}",
                            flush=True,
                        )

        episodes = sorted(episodes, key=lambda x: (x[0], x[1]))
        if self._config.max_episodes is not None:
            episodes = episodes[: self._config.max_episodes]
        print(f"ArioStreamingDataset: found {len(episodes)} episodes on S3")
        return episodes

    def _build_index(self):
        """Download eef_torso.pt from each episode to determine its length."""
        import json

        from tqdm import tqdm

        s3 = self._get_s3()
        cumulative = 0
        valid_episodes = []
        rate = self._config.video_downsample_rate

        for bucket, prefix in tqdm(self._episodes, desc="Building frame index"):
            try:
                data = self._s3_download_bytes(s3, bucket, prefix + "eef_torso.pt")
                tensor = torch.load(io.BytesIO(data), map_location="cpu")
                raw_len = tensor.shape[0]
            except Exception:
                continue

            if raw_len < self._config.min_frames:
                continue

            # Load per-episode instruction from instructions.json
            if self._config.load_instructions:
                try:
                    instr_data = self._s3_download_bytes(s3, bucket, prefix + "instructions.json")
                    instr_json = json.loads(instr_data)
                    sub_instructions = instr_json.get("sub_instructions", [])
                    if sub_instructions:
                        self._instructions[prefix] = sub_instructions[0]["instruction"]
                    else:
                        self._instructions[prefix] = self._config.task
                except Exception:
                    self._instructions[prefix] = self._config.task

            n_frames = len(range(0, raw_len, rate))
            valid_episodes.append((bucket, prefix))
            self._episode_lengths.append(n_frames)
            cumulative += n_frames
            self._cumulative.append(cumulative)

        self._episodes = valid_episodes
        print(f"ArioStreamingDataset: {len(self._episodes)} valid episodes, {cumulative} total frames")

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int, _retries: int = 3, _timeout: float = 60.0) -> dict:
        import random
        import signal

        if index < 0:
            index += len(self)

        for attempt in range(_retries):
            ep_idx, frame_idx = self._global_to_local(index)
            bucket, prefix = self._episodes[ep_idx]
            cache_key = f"{bucket}/{prefix}"

            try:
                state_action, frames_dict = self._get_episode_with_timeout(bucket, prefix, cache_key, _timeout)

                state = state_action[frame_idx]
                actions = self._get_action_chunk(state_action, frame_idx)

                prompt = self._instructions.get(prefix, self._config.task)

                result = {
                    "observation/image": frames_dict["cam_high"][frame_idx],
                    "observation/state": state,
                    "actions": actions,
                    "prompt": prompt,
                }

                if self._config.multi_view:
                    for cam in CAMERA_VIEWS:
                        result[f"observation/{cam}"] = frames_dict[cam][frame_idx]

                return result
            except (TimeoutError, Exception) as e:
                print(f"[WARN] Episode fetch failed (attempt {attempt+1}/{_retries}), ep={ep_idx}: {e}. Skipping.", flush=True)
                index = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Failed to fetch any episode after {_retries} retries")

    def _get_episode_with_timeout(self, bucket, prefix, cache_key, timeout):
        """Wrap _get_episode with a timeout. Falls back to no-timeout on non-main threads."""
        import signal
        import threading

        if threading.current_thread() is not threading.main_thread():
            return self._get_episode(bucket, prefix, cache_key)

        def _handler(signum, frame):
            raise TimeoutError(f"Episode fetch timed out after {timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout))
        try:
            result = self._get_episode(bucket, prefix, cache_key)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result

    def _global_to_local(self, index: int) -> tuple[int, int]:
        """Convert global frame index to (episode_idx, local_frame_idx)."""
        import bisect
        ep_idx = bisect.bisect_right(self._cumulative, index)
        local = index - (self._cumulative[ep_idx - 1] if ep_idx > 0 else 0)
        return ep_idx, local

    def _get_action_chunk(self, state_action: np.ndarray, frame_idx: int) -> np.ndarray:
        """Get action_horizon consecutive actions starting at frame_idx, clamping at end."""
        n = len(state_action)
        indices = [min(frame_idx + i, n - 1) for i in range(self._action_horizon)]
        return state_action[indices]

    def _get_episode(
        self, bucket: str, prefix: str, cache_key: str
    ) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
        """Get episode data, using memory LRU cache backed by disk cache."""
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Try disk cache
        disk_path = self._disk_cache_path(cache_key)
        if disk_path.exists():
            try:
                with np.load(disk_path, allow_pickle=False) as data:
                    state_action = data["state_action"]
                    missing_cameras = [cam for cam in CAMERA_VIEWS if cam not in data]
                    if missing_cameras:
                        raise ValueError(
                            f"Cache {disk_path} is missing camera arrays {missing_cameras}"
                        )
                    frames_dict = {cam: list(data[cam]) for cam in CAMERA_VIEWS}
                disk_path.stat()
                os.utime(disk_path, None)
                self._cache[cache_key] = (state_action, frames_dict)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                return state_action, frames_dict
            except Exception:
                disk_path.unlink(missing_ok=True)

        # Download and decode
        s3 = self._get_s3()
        state_action = self._build_state_action(s3, bucket, prefix)

        if self._config.skip_video:
            # Only need state/actions (e.g. for norm stats), skip expensive video download
            rate = self._config.video_downsample_rate
            indices = list(range(0, len(state_action), rate))
            state_action = state_action[indices].astype(np.float32)
            target_w, target_h = self._config.image_size
            dummy_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            frames_dict = {cam: [dummy_frame] * len(state_action) for cam in CAMERA_VIEWS}
        else:
            if self._config.multi_view:
                frames_dict = self._extract_multi_view_frames(s3, bucket, prefix)
            else:
                frames = self._extract_video_frames(s3, bucket, prefix + "video.mp4")
                frames_dict = {cam: frames for cam in CAMERA_VIEWS}

            # Align lengths and downsample
            min_len = len(state_action)
            for cam in frames_dict:
                min_len = min(min_len, len(frames_dict[cam]))
            rate = self._config.video_downsample_rate
            indices = list(range(0, min_len, rate))

            state_action = state_action[indices].astype(np.float32)
            frames_dict = {cam: [frames_dict[cam][i] for i in indices] for cam in frames_dict}

        # Save to disk cache
        self._save_to_disk_cache(disk_path, state_action, frames_dict)

        self._cache[cache_key] = (state_action, frames_dict)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return state_action, frames_dict

    def _disk_cache_path(self, cache_key: str) -> Path:
        import hashlib

        cache_identity = "|".join(
            [
                f"v{DISK_CACHE_FORMAT_VERSION}",
                cache_key,
                f"rate={self._config.video_downsample_rate}",
                f"size={self._config.image_size}",
                f"multi_view={self._config.multi_view}",
                f"skip_video={self._config.skip_video}",
            ]
        )
        key_hash = hashlib.md5(cache_identity.encode()).hexdigest()
        cache_dir = Path(self._config.disk_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key_hash}.npz"

    def _save_to_disk_cache(self, path: Path, state_action: np.ndarray, frames_dict: dict[str, list[np.ndarray]]):
        try:
            self._evict_disk_cache_if_needed()
            save_data = {"state_action": state_action}
            for cam, frames in frames_dict.items():
                save_data[cam] = np.stack(frames)
            np.savez(path, **save_data)
        except Exception:
            path.unlink(missing_ok=True)

    def _evict_disk_cache_if_needed(self):
        cache_dir = Path(self._config.disk_cache_dir)
        if not cache_dir.exists():
            return
        max_bytes = int(self._config.disk_cache_max_gb * 1024**3)
        files = list(cache_dir.glob("*.npz"))
        total = sum(f.stat().st_size for f in files)
        if total <= max_bytes:
            return
        files.sort(key=lambda f: f.stat().st_atime)
        for f in files:
            if total <= max_bytes * 0.8:
                break
            total -= f.stat().st_size
            f.unlink(missing_ok=True)

    def _build_state_action(self, s3, bucket: str, prefix: str) -> np.ndarray:
        tensors = {}
        for fname in PT_FILES:
            data = self._s3_download_bytes(s3, bucket, prefix + fname)
            tensors[fname] = torch.load(io.BytesIO(data), map_location="cpu")

        state_action = torch.cat(
            [
                tensors["eef_torso.pt"],
                tensors["head.pt"],
                tensors["eef_left.pt"],
                tensors["gripper_cmd.pt"][:, 0:1],
                tensors["eef_right.pt"],
                tensors["gripper_cmd.pt"][:, 1:2],
            ],
            dim=-1,
        )
        return state_action.numpy()

    def _extract_video_frames(self, s3, bucket: str, video_key: str) -> list[np.ndarray]:
        data = self._s3_download_bytes(s3, bucket, video_key)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(data)
        tmp.close()
        try:
            frames = self._decode_video(Path(tmp.name))
        finally:
            os.unlink(tmp.name)
        return frames

    def _extract_multi_view_frames(
        self, s3, bucket: str, prefix: str
    ) -> dict[str, list[np.ndarray]]:
        """Load frames from raw_video/ for each camera view."""
        frames_dict = {}
        for cam in CAMERA_VIEWS:
            video_key = prefix + f"raw_video/{cam}.mp4"
            try:
                frames_dict[cam] = self._extract_video_frames(s3, bucket, video_key)
            except Exception as e:
                raise RuntimeError(f"Failed to load required camera view {cam}: {e}") from e
        return frames_dict

    def _decode_video(self, path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        target_w, target_h = self._config.image_size
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def _s3_download_bytes(s3, bucket: str, key: str) -> bytes:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        path = uri.split("://", 1)[1]
        bucket, _, key = path.partition("/")
        return bucket, key
