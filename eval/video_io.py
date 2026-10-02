import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from PIL import Image


def read_image(path: Union[str, os.PathLike]) -> np.ndarray:
    str_path = str(path)
    if not os.path.exists(str_path):
        raise FileNotFoundError(f"Image not found: {str_path}")

    img = cv2.imread(str_path)
    if img is None:
        pil_img = Image.open(str_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def write_image(path: Union[str, os.PathLike], img_bgr: np.ndarray) -> bool:
    str_path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(str_path)), exist_ok=True)
    return cv2.imwrite(str_path, img_bgr)


def load_frames(
    video_path: Union[str, os.PathLike],
    max_frames: Optional[int] = None
) -> Tuple[List[np.ndarray], float]:
    str_path = str(video_path)
    if not os.path.exists(str_path):
        raise FileNotFoundError(f"Video not found: {str_path}")

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str_path, ctx=cpu(0))
        orig_fps = float(vr.get_avg_fps() or 30.0)
        total = min(len(vr), max_frames) if max_frames else len(vr)
        batch = vr.get_batch(list(range(total))).asnumpy()
        return [batch[i] for i in range(len(batch))], orig_fps
    except Exception:
        pass

    cap = cv2.VideoCapture(str_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = []
        while True:
            if max_frames and len(frames) >= max_frames:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames, fps
    finally:
        cap.release()


def frames_to_tensor(frames_rgb: List[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(frames_rgb, axis=0)).float() / 255.0


def extract_joints(pose_metas: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(pose_metas)
    joints = np.zeros((T, 133, 2), dtype=np.float32)
    confs = np.zeros((T, 133), dtype=np.float32)
    bboxes = np.zeros((T, 4), dtype=np.float32)

    for t, meta in enumerate(pose_metas):
        w = float(meta.get("width", 1.0))
        h = float(meta.get("height", 1.0))

        if "keypoints" in meta and np.array(meta["keypoints"]).shape[0] == 133:
            raw = np.array(meta["keypoints"], dtype=np.float32)
            if raw[:, :2].max() <= 1.05 and (w > 1.0 or h > 1.0):
                joints[t, :, 0] = raw[:, 0] * w
                joints[t, :, 1] = raw[:, 1] * h
            else:
                joints[t, :, :2] = raw[:, :2]
            confs[t] = raw[:, 2] if raw.shape[1] >= 3 else 1.0
        else:
            slices = [
                ("keypoints_body", 0, 17),
                ("keypoints_face", 23, 68),
                ("keypoints_left_hand", 91, 21),
                ("keypoints_right_hand", 112, 21),
            ]
            for key, start, count in slices:
                if key in meta:
                    pts = np.array(meta[key], dtype=np.float32)[:count]
                    n_pts = len(pts)
                    if pts[:, :2].max() <= 1.05 and (w > 1.0 or h > 1.0):
                        joints[t, start : start + n_pts, 0] = pts[:, 0] * w
                        joints[t, start : start + n_pts, 1] = pts[:, 1] * h
                    else:
                        joints[t, start : start + n_pts, :2] = pts[:, :2]
                    confs[t, start : start + n_pts] = pts[:, 2] if pts.shape[1] >= 3 else 1.0

        valid = confs[t] > 0.1
        if np.sum(valid) > 0:
            v_pts = joints[t, valid]
            bboxes[t] = [float(v_pts[:, 0].min()), float(v_pts[:, 1].min()), float(v_pts[:, 0].max()), float(v_pts[:, 1].max())]
        else:
            bboxes[t] = [0.0, 0.0, w, h]

    return joints, confs, bboxes
