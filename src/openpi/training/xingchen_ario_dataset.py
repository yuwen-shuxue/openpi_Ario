"""Custom dataset for loading Xingchen robot data in ARIO format from OSS/S3.

Data structure per episode:
    raw_video/cam_torso.mp4 (1280x720)
    raw_video/cam_left_wrist.mp4 (640x360)
    raw_video/cam_right_wrist.mp4 (640x360)
    torso.pt (N, 4) - waist joint positions
    head.pt (N, 2) - head joint positions
    eef_left_xyzw.pt (N, 7) - left arm end-effector pose
    eef_right_xyzw.pt (N, 7) - right arm end-effector pose
    gripper_cmd.pt (N, 2) - gripper commands [left, right]
    instructions.json - task instructions
"""

import io
import json
import logging
import os
import pathlib
import tempfile
from typing import Sequence

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class XingchenArioDataset:
    """Loads Xingchen ARIO episodes from local or S3-compatible storage.

    Sampling strategy:
        - video_downsample_rate: take every Nth frame from raw video
        - num_frames: total number of downsampled frames per window
        - video_action_freq: action label every M-th downsampled frame
        - action_horizon: number of action steps per sample = (num_frames - 1) // video_action_freq
    """

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 13,
        video_downsample_rate: int = 6,
        video_action_freq: int = 2,
        episode_ids: Sequence[str] | None = None,
    ):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.video_downsample_rate = video_downsample_rate
        self.video_action_freq = video_action_freq
        self.action_horizon = (num_frames - 1) // video_action_freq  # = 6

        self._use_s3 = data_dir.startswith("s3://") or data_dir.startswith("oss://")

        if self._use_s3:
            prefix = data_dir.replace("oss://", "s3://")
            if episode_ids is not None:
                self.episodes = [f"{prefix}/{eid}" for eid in episode_ids]
            else:
                self.episodes = self._discover_episodes_s3(prefix)
        else:
            if episode_ids is not None:
                self.episodes = [os.path.join(data_dir, eid) for eid in episode_ids]
            else:
                self.episodes = self._discover_episodes_local(data_dir)

        self._samples = self._build_sample_index()
        logger.info(f"XingchenArioDataset: {len(self.episodes)} episodes, {len(self._samples)} samples")

    def _discover_episodes_s3(self, prefix: str) -> list[str]:
        import megfile
        entries = megfile.smart_listdir(prefix)
        episodes = []
        for entry in sorted(entries):
            path = f"{prefix}/{entry}"
            if megfile.smart_isdir(path):
                episodes.append(path)
        return episodes

    def _discover_episodes_local(self, data_dir: str) -> list[str]:
        base = pathlib.Path(data_dir)
        episodes = []
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                episodes.append(str(entry))
        return episodes

    def _get_episode_length(self, episode_path: str) -> int:
        """Get number of frames in an episode by loading torso.pt shape."""
        torso_path = f"{episode_path}/torso.pt"
        tensor = self._load_pt(torso_path)
        return tensor.shape[0]

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """Build (episode_idx, start_frame) pairs for all valid windows."""
        samples = []
        for ep_idx, ep_path in enumerate(self.episodes):
            try:
                ep_len = self._get_episode_length(ep_path)
            except Exception as e:
                logger.warning(f"Skipping episode {ep_path}: {e}")
                continue

            # Window spans num_frames * video_downsample_rate raw frames
            window_raw_frames = self.num_frames * self.video_downsample_rate
            # Slide by action_horizon * video_action_freq * video_downsample_rate
            stride = self.action_horizon * self.video_action_freq * self.video_downsample_rate

            for start in range(0, max(1, ep_len - window_raw_frames + 1), stride):
                if start + window_raw_frames <= ep_len:
                    samples.append((ep_idx, start))

        return samples

    def _load_pt(self, path: str) -> torch.Tensor:
        if self._use_s3:
            import megfile
            with megfile.smart_open(path, "rb") as f:
                buf = io.BytesIO(f.read())
                return torch.load(buf, map_location="cpu", weights_only=True)
        else:
            return torch.load(path, map_location="cpu", weights_only=True)

    def _load_json(self, path: str) -> dict:
        if self._use_s3:
            import megfile
            with megfile.smart_open(path, "rb") as f:
                return json.load(f)
        else:
            with open(path, "r") as f:
                return json.load(f)

    def _load_video_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Load a single frame from a video file."""
        if self._use_s3:
            import megfile
            # Download to temp file for cv2
            with megfile.smart_open(video_path, "rb") as f:
                data = f.read()
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(data)
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            os.unlink(tmp.name)
        else:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _get_instruction(self, episode_path: str, frame_idx: int) -> str:
        """Get the instruction text for a given frame."""
        try:
            instructions = self._load_json(f"{episode_path}/instructions.json")
            # Find the sub-instruction that covers this frame
            if "sub_instructions" in instructions:
                for sub in instructions["sub_instructions"]:
                    if sub.get("start_frame", 0) <= frame_idx <= sub.get("end_frame", float("inf")):
                        return sub.get("text", sub.get("instruction", ""))
            # Fallback to top-level instruction
            if "instruction" in instructions:
                return instructions["instruction"]
            if "text" in instructions:
                return instructions["text"]
        except Exception:
            pass
        return "manipulate the blocks"

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index) -> dict:
        ep_idx, start_frame = self._samples[index.__index__() if hasattr(index, '__index__') else index]
        ep_path = self.episodes[ep_idx]

        # Load action tensors for the full episode
        torso_qpos = self._load_pt(f"{ep_path}/torso.pt").numpy()  # (N, 4)
        head = self._load_pt(f"{ep_path}/head.pt").numpy()  # (N, 2)
        eef_left = self._load_pt(f"{ep_path}/eef_left_xyzw.pt").numpy()  # (N, 7)
        eef_right = self._load_pt(f"{ep_path}/eef_right_xyzw.pt").numpy()  # (N, 7)
        gripper_cmd = self._load_pt(f"{ep_path}/gripper_cmd.pt").numpy()  # (N, 2)

        # Current observation frame (first frame in window)
        obs_frame_idx = start_frame

        # Load observation image
        cam_torso = self._load_video_frame(f"{ep_path}/raw_video/cam_torso.mp4", obs_frame_idx)
        cam_left = self._load_video_frame(f"{ep_path}/raw_video/cam_left_wrist.mp4", obs_frame_idx)
        cam_right = self._load_video_frame(f"{ep_path}/raw_video/cam_right_wrist.mp4", obs_frame_idx)

        # Build action sequence: sample at video_action_freq intervals after observation
        # Frame indices in the downsampled sequence: 0 is obs, then 1,2,...,num_frames-1
        # Action at downsampled frame i corresponds to raw frame: start_frame + i * video_downsample_rate
        # We take actions at every video_action_freq downsampled frames
        action_frames = []
        for i in range(1, self.num_frames, self.video_action_freq):
            raw_idx = start_frame + i * self.video_downsample_rate
            if raw_idx < len(torso_qpos):
                action_frames.append(raw_idx)
            else:
                action_frames.append(len(torso_qpos) - 1)

        # Truncate to action_horizon
        action_frames = action_frames[:self.action_horizon]

        # Build action array (action_horizon, 22)
        actions = []
        for af in action_frames:
            action = np.concatenate([
                torso_qpos[af],           # (4,)
                head[af],                 # (2,)
                eef_left[af],             # (7,)
                gripper_cmd[af, 0:1],     # (1,) left gripper
                eef_right[af],            # (7,)
                gripper_cmd[af, 1:2],     # (1,) right gripper
            ]).astype(np.float32)
            actions.append(action)

        actions = np.stack(actions, axis=0)  # (action_horizon, 22)

        # Build current state (same format as action, from obs frame)
        state = np.concatenate([
            torso_qpos[obs_frame_idx],
            head[obs_frame_idx],
            eef_left[obs_frame_idx],
            gripper_cmd[obs_frame_idx, 0:1],
            eef_right[obs_frame_idx],
            gripper_cmd[obs_frame_idx, 1:2],
        ]).astype(np.float32)

        # Get instruction
        prompt = self._get_instruction(ep_path, obs_frame_idx)

        return {
            "cam_torso": cam_torso,
            "cam_left_wrist": cam_left,
            "cam_right_wrist": cam_right,
            "torso_qpos": torso_qpos[obs_frame_idx].astype(np.float32),
            "head": head[obs_frame_idx].astype(np.float32),
            "eef_left": eef_left[obs_frame_idx].astype(np.float32),
            "eef_right": eef_right[obs_frame_idx].astype(np.float32),
            "gripper_left": gripper_cmd[obs_frame_idx, 0:1].astype(np.float32),
            "gripper_right": gripper_cmd[obs_frame_idx, 1:2].astype(np.float32),
            "actions": actions,
            "prompt": prompt,
            "state": state,
        }
