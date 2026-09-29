# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Video Utilities for Unicode, Decord, OpenCV, and Tensor conversion.
Provides rock-solid handling of Vietnamese Unicode filepaths on Windows.
"""

import os
import cv2
import numpy as np
import torch
import shutil
import tempfile
from typing import List, Tuple, Optional, Union
from PIL import Image

def read_image_unicode(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Safely reads an image from a path that may contain non-ASCII / Unicode characters
    (e.g., Vietnamese diacritics on Windows).
    Returns BGR numpy array matching cv2.imread behavior, or raises FileNotFoundError / ValueError.
    """
    str_path = str(path)
    if not os.path.exists(str_path):
        raise FileNotFoundError(f"Image file not found: {str_path}")
    
    # 1. Try reading as raw binary buffer and decoding with OpenCV
    try:
        data = np.fromfile(str_path, dtype=np.uint8)
        if len(data) > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    # 2. Try PIL fallback
    try:
        pil_img = Image.open(str_path).convert('RGB')
        # PIL is RGB -> convert to BGR for OpenCV standard
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    # 3. Standard cv2.imread
    img = cv2.imread(str_path)
    if img is not None:
        return img

    raise ValueError(f"Failed to read image from path: {str_path}")


def write_image_unicode(path: Union[str, os.PathLike], img_bgr: np.ndarray) -> bool:
    """
    Safely writes an image to a path that may contain non-ASCII / Unicode characters on Windows.
    img_bgr: BGR numpy array
    """
    str_path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(str_path)), exist_ok=True)
    ext = os.path.splitext(str_path)[1]
    if not ext:
        ext = '.png'
    
    success, buf = cv2.imencode(ext, img_bgr)
    if not success:
        return False
    
    with open(str_path, 'wb') as f:
        f.write(buf.tobytes())
    return True


def safe_open_video_capture(video_path: Union[str, os.PathLike]) -> Tuple[cv2.VideoCapture, Optional[str]]:
    """
    Opens cv2.VideoCapture safely on Windows with Unicode path support.
    Returns (cap, temp_file_path).
    If a temporary copy was created to bypass OpenCV Unicode limitations, temp_file_path is returned
    so it can be cleaned up after reading.
    """
    str_path = str(video_path)
    if not os.path.exists(str_path):
        raise FileNotFoundError(f"Video file not found: {str_path}")

    # First attempt direct open
    cap = cv2.VideoCapture(str_path)
    if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        return cap, None

    cap.release()

    # Second attempt: check if path contains non-ascii characters
    try:
        str_path.encode('ascii')
        is_ascii = True
    except UnicodeEncodeError:
        is_ascii = False

    if not is_ascii:
        # Create temp ASCII file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"wan_eval_tmp_{os.getpid()}_{np.random.randint(10000, 99999)}.mp4")
        shutil.copy2(str_path, temp_file)
        cap = cv2.VideoCapture(temp_file)
        if cap.isOpened():
            return cap, temp_file
        cap.release()
        if os.path.exists(temp_file):
            os.remove(temp_file)

    raise RuntimeError(f"Could not open video capture for: {str_path}")


def load_video_frames(
    video_path: Union[str, os.PathLike],
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None
) -> Tuple[List[np.ndarray], float]:
    """
    Loads video frames as a list of RGB numpy arrays [H, W, 3] (uint8).
    Supports decord VideoReader if available, with robust OpenCV fallback.
    Returns (frames_rgb, original_fps).
    """
    str_path = str(video_path)
    
    # Try decord first if installed
    try:
        import decord
        from decord import VideoReader, cpu
        vr = VideoReader(str_path, ctx=cpu(0))
        orig_fps = vr.get_avg_fps()
        total_frames = len(vr)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        indices = list(range(total_frames))
        batch = vr.get_batch(indices).asnumpy()  # [T, H, W, 3] RGB
        frames = [batch[i] for i in range(len(batch))]
        return frames, orig_fps
    except Exception:
        pass

    # Fallback to OpenCV
    cap, tmp_path = safe_open_video_capture(str_path)
    try:
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30.0

        frames = []
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_idx += 1
        return frames, orig_fps
    finally:
        cap.release()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def frames_to_video_tensor(frames_rgb: List[np.ndarray]) -> torch.Tensor:
    """
    Converts list of RGB numpy frames [H, W, C] (uint8 0..255)
    to a PyTorch float tensor [T, H, W, C] in range [0, 1].
    """
    arr = np.stack(frames_rgb, axis=0) # [T, H, W, C]
    return torch.from_numpy(arr).float() / 255.0


def extract_joints_from_metas(pose_metas: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs 133-keypoint coordinates, confidences, and bounding boxes
    from ViTPose / AAPoseMeta format pose_metas.
    ViTPose WholeBody 133 Layout:
      - 0..16: Body & Arms (17)
      - 17..22: Left/Right foot / toe
      - 23..90: Face Landmarks (68)
      - 91..111: Left Hand (21)
      - 112..132: Right Hand (21)
    
    Each pose_meta contains:
      - 'keypoints_body': [20, 3] or [17, 3] or [133, 3]
      - 'keypoints_face': [68, 3]
      - 'keypoints_left_hand': [21, 3]
      - 'keypoints_right_hand': [21, 3]
      - 'width', 'height'
    Normalized [0, 1] coords are converted back to pixel coordinates [x, y].
    
    Returns:
      joints: np.ndarray [T, 133, 2] in pixels
      confidences: np.ndarray [T, 133]
      bboxes: np.ndarray [T, 4] in pixels
    """
    T = len(pose_metas)
    joints = np.zeros((T, 133, 2), dtype=np.float32)
    confs = np.zeros((T, 133), dtype=np.float32)
    bboxes = np.zeros((T, 4), dtype=np.float32)

    for t, meta in enumerate(pose_metas):
        w = float(meta.get('width', 1.0))
        h = float(meta.get('height', 1.0))

        # Check if raw 133 keypoints are present in meta
        if 'keypoints' in meta and np.array(meta['keypoints']).shape[0] == 133:
            raw_kp = np.array(meta['keypoints'], dtype=np.float32)
            # If coordinates are normalized (max <= 1.0), scale by w, h
            if raw_kp[:, :2].max() <= 1.05 and (w > 1.0 or h > 1.0):
                joints[t, :, 0] = raw_kp[:, 0] * w
                joints[t, :, 1] = raw_kp[:, 1] * h
            else:
                joints[t, :, :2] = raw_kp[:, :2]
            if raw_kp.shape[1] >= 3:
                confs[t] = raw_kp[:, 2]
            else:
                confs[t] = 1.0
        else:
            # Reconstruct from split components
            # Body:
            if 'keypoints_body' in meta:
                kp_body = np.array(meta['keypoints_body'], dtype=np.float32)
                num_body = min(17, len(kp_body))
                if kp_body[:, :2].max() <= 1.05 and (w > 1.0 or h > 1.0):
                    joints[t, :num_body, 0] = kp_body[:num_body, 0] * w
                    joints[t, :num_body, 1] = kp_body[:num_body, 1] * h
                else:
                    joints[t, :num_body, :2] = kp_body[:num_body, :2]
                confs[t, :num_body] = kp_body[:num_body, 2] if kp_body.shape[1] >= 3 else 1.0

            # Face: 23..90 (68 joints)
            if 'keypoints_face' in meta:
                kp_face = np.array(meta['keypoints_face'], dtype=np.float32)
                num_face = min(68, len(kp_face))
                if kp_face[:, :2].max() <= 1.05 and (w > 1.0 or h > 1.0):
                    joints[t, 23:23+num_face, 0] = kp_face[:num_face, 0] * w
                    joints[t, 23:23+num_face, 1] = kp_face[:num_face, 1] * h
                else:
                    joints[t, 23:23+num_face, :2] = kp_face[:num_face, :2]
                confs[t, 23:23+num_face] = kp_face[:num_face, 2] if kp_face.shape[1] >= 3 else 1.0

            # Left hand: 91..111 (21 joints)
            if 'keypoints_left_hand' in meta:
                kp_lh = np.array(meta['keypoints_left_hand'], dtype=np.float32)
                num_lh = min(21, len(kp_lh))
                if kp_lh[:, :2].max() <= 1.05 and (w > 1.0 or h > 1.0):
                    joints[t, 91:91+num_lh, 0] = kp_lh[:num_lh, 0] * w
                    joints[t, 91:91+num_lh, 1] = kp_lh[:num_lh, 1] * h
                else:
                    joints[t, 91:91+num_lh, :2] = kp_lh[:num_lh, :2]
                confs[t, 91:91+num_lh] = kp_lh[:num_lh, 2] if kp_lh.shape[1] >= 3 else 1.0

            # Right hand: 112..132 (21 joints)
            if 'keypoints_right_hand' in meta:
                kp_rh = np.array(meta['keypoints_right_hand'], dtype=np.float32)
                num_rh = min(21, len(kp_rh))
                if kp_rh[:, :2].max() <= 1.05 and (w > 1.0 or h > 1.0):
                    joints[t, 112:112+num_rh, 0] = kp_rh[:num_rh, 0] * w
                    joints[t, 112:112+num_rh, 1] = kp_rh[:num_rh, 1] * h
                else:
                    joints[t, 112:112+num_rh, :2] = kp_rh[:num_rh, :2]
                confs[t, 112:112+num_rh] = kp_rh[:num_rh, 2] if kp_rh.shape[1] >= 3 else 1.0

        # Compute bounding box from valid joints
        valid_mask = confs[t] > 0.1
        if np.sum(valid_mask) > 0:
            v_pts = joints[t, valid_mask]
            min_xy = np.min(v_pts, axis=0)
            max_xy = np.max(v_pts, axis=0)
            bboxes[t] = [min_xy[0], min_xy[1], max_xy[0], max_xy[1]]
        else:
            bboxes[t] = [0.0, 0.0, w, h]

    return joints, confs, bboxes
