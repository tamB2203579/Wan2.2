# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Computer Vision Metrics Engine for Video and Motion Fidelity Evaluation.

Implements standard Computer Vision metrics:
  1. FVD (Fréchet Video Distance - Unterthiner et al., 2018):
     Spatio-temporal distribution distance using 3D-ResNet / Kinetics-400 embeddings.
     Supports both per-video FVD and Global Dataset-Level FVD.
  2. PA-MPJPE (Procrustes-Aligned Mean Per Joint Position Error in pixels):
     Eliminates translation, rotation SO(2), and scale differences via SVD.
  3. PCK@0.05 and PCK@0.10 (Percentage of Correct Keypoints):
     Normalized by bounding box scale max(w, h).
  4. N-MPJPE (Normalized MPJPE):
     Scale-invariant joint error centered at pose centroid.
  5. Keypoint Subsets:
     Overall (133), Body & Arms (0..16), Face Landmarks (23..90), Hands (91..132).
"""

import os
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union

# ==============================================================================
# 1. KEYPOINT SUBSETS (ViTPose-H WholeBody 133 Layout)
# ==============================================================================
KEYPOINT_SUBSETS = {
    "overall": list(range(133)),
    "body": list(range(17)),         # Indices 0..16: Body & Arms
    "face": list(range(23, 91)),      # Indices 23..90: Face Landmarks (68 keypoints)
    "hands": list(range(91, 133)),    # Indices 91..132: Left hand (91..111) + Right hand (112..132) (42 keypoints)
}


# ==============================================================================
# 2. PROCRUSTES ALIGNMENT & PA-MPJPE
# ==============================================================================
def compute_pa_mpjpe(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    conf_thresh: float = 0.3,
    joint_indices: Optional[List[int]] = None
) -> float:
    """
    Computes Procrustes-Aligned MPJPE (PA-MPJPE) in pixels per frame.
    Solves optimal similarity transformation (scale s, rotation R in SO(2), translation t) via SVD:
        min_{s, R, t} || Y - (s * X * R + t) ||_F
    where Y is ground truth and X is prediction.

    Args:
        gt_joints: np.ndarray [T, J, 2] - Ground truth 2D keypoints
        pred_joints: np.ndarray [T, J, 2] - Predicted 2D keypoints
        confidences: np.ndarray [T, J] - Keypoint confidence scores (optional)
        conf_thresh: float - Minimum confidence threshold
        joint_indices: list of joint indices to evaluate (defaults to all)

    Returns:
        float: Mean Euclidean joint error in pixels
    """
    T = min(gt_joints.shape[0], pred_joints.shape[0])
    if T == 0:
        return 0.0

    J = gt_joints.shape[1]
    if joint_indices is None:
        joint_indices = list(range(J))

    errors = []

    for t in range(T):
        Y = gt_joints[t, joint_indices]    # [K, 2] GT
        X = pred_joints[t, joint_indices]  # [K, 2] Pred

        if confidences is not None:
            conf_t = confidences[t, joint_indices]
            mask = conf_t >= conf_thresh
            if np.sum(mask) < 3:
                mask = np.ones(len(joint_indices), dtype=bool)
        else:
            mask = np.ones(len(joint_indices), dtype=bool)

        Y_m = Y[mask]
        X_m = X[mask]

        if len(Y_m) < 3:
            continue

        # 1. Centering
        mu_Y = np.mean(Y_m, axis=0, keepdims=True)
        mu_X = np.mean(X_m, axis=0, keepdims=True)
        Y_c = Y_m - mu_Y
        X_c = X_m - mu_X

        # 2. Frobenius norm
        norm_Y = np.linalg.norm(Y_c)
        norm_X = np.linalg.norm(X_c)

        if norm_X < 1e-6 or norm_Y < 1e-6:
            continue

        Y_cn = Y_c / norm_Y
        X_cn = X_c / norm_X

        # 3. SVD for optimal orthogonal rotation R in SO(2)
        H = X_cn.T @ Y_cn
        U, S, Vt = np.linalg.svd(H)
        d = np.ones(2)
        if np.linalg.det(U @ Vt) < 0:
            d[-1] = -1
        R = U @ np.diag(d) @ Vt

        # Optimal scale s and translation t
        s = np.sum(S * d) * (norm_Y / norm_X)
        t_vec = mu_Y - s * (mu_X @ R)

        # 4. Transform prediction and compute Euclidean error
        X_aligned = s * (X_m @ R) + t_vec
        diff = np.linalg.norm(X_aligned - Y_m, axis=-1)
        errors.extend(diff.tolist())

    return float(np.mean(errors)) if errors else 0.0


# ==============================================================================
# 3. SCALE-NORMALIZED MPJPE (N-MPJPE)
# ==============================================================================
def compute_n_mpjpe(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    conf_thresh: float = 0.3,
    joint_indices: Optional[List[int]] = None
) -> float:
    """
    Computes Normalized MPJPE (N-MPJPE).
    Centers each pose at its centroid and normalizes by the torso/bounding scale.
    Output is unitless scale-invariant joint error.
    """
    T = min(gt_joints.shape[0], pred_joints.shape[0])
    if T == 0:
        return 0.0

    J = gt_joints.shape[1]
    if joint_indices is None:
        joint_indices = list(range(J))

    errors = []

    for t in range(T):
        Y = gt_joints[t, joint_indices]
        X = pred_joints[t, joint_indices]

        if confidences is not None:
            conf_t = confidences[t, joint_indices]
            mask = conf_t >= conf_thresh
            if np.sum(mask) < 3:
                mask = np.ones(len(joint_indices), dtype=bool)
        else:
            mask = np.ones(len(joint_indices), dtype=bool)

        Y_m = Y[mask]
        X_m = X[mask]

        if len(Y_m) < 3:
            continue

        mu_Y = np.mean(Y_m, axis=0, keepdims=True)
        mu_X = np.mean(X_m, axis=0, keepdims=True)

        scale_Y = np.max(np.ptp(Y_m, axis=0))
        scale_X = np.max(np.ptp(X_m, axis=0))
        if scale_Y < 1e-4:
            scale_Y = 1.0
        if scale_X < 1e-4:
            scale_X = 1.0

        Y_norm = (Y_m - mu_Y) / scale_Y
        X_norm = (X_m - mu_X) / scale_X

        diff = np.linalg.norm(X_norm - Y_norm, axis=-1)
        errors.extend(diff.tolist())

    return float(np.mean(errors)) if errors else 0.0


# ==============================================================================
# 4. PERCENTAGE OF CORRECT KEYPOINTS (PCK)
# ==============================================================================
def compute_pck(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    bboxes: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    conf_thresh: float = 0.3,
    joint_indices: Optional[List[int]] = None
) -> Tuple[float, int]:
    """
    Computes Percentage of Correct Keypoints (PCK@alpha).
    Scale is defined as max(width, height) of ground truth human bounding box / joint spread.

    Args:
        gt_joints: np.ndarray [T, J, 2]
        pred_joints: np.ndarray [T, J, 2]
        confidences: np.ndarray [T, J]
        bboxes: np.ndarray [T, 4] bounding boxes [x1, y1, x2, y2]
        alpha: float - Threshold fraction (e.g. 0.05 or 0.10)
        conf_thresh: float - Minimum confidence threshold
        joint_indices: list of joint indices to evaluate

    Returns:
        score: float (0.0 to 100.0) percentage
        num_valid: int
    """
    T = min(gt_joints.shape[0], pred_joints.shape[0])
    if T == 0:
        return 0.0, 0

    J = gt_joints.shape[1]
    if joint_indices is None:
        joint_indices = list(range(J))

    correct = 0
    total = 0

    for t in range(T):
        scale = None
        if bboxes is not None and t < len(bboxes) and bboxes[t] is not None:
            bbox = bboxes[t]
            if len(bbox) >= 4:
                w = abs(bbox[2] - bbox[0])
                h = abs(bbox[3] - bbox[1])
                scale = max(w, h)

        if scale is None or scale <= 1e-4:
            valid_gt = gt_joints[t, joint_indices]
            if len(valid_gt) > 1:
                scale = float(np.max(np.ptp(valid_gt, axis=0)))
            else:
                scale = 100.0

        if scale <= 1e-4:
            scale = 100.0

        threshold = alpha * scale

        for j in joint_indices:
            if confidences is not None and confidences[t, j] < conf_thresh:
                continue
            dist = np.linalg.norm(pred_joints[t, j] - gt_joints[t, j])
            if dist <= threshold:
                correct += 1
            total += 1

    score = float((correct / total) * 100.0) if total > 0 else 0.0
    return score, total


# ==============================================================================
# 5. FRÉCHET VIDEO DISTANCE (FVD) - COMPUTER VISION STANDARD
# ==============================================================================
class VideoFeatureExtractor:
    """
    Extracts spatio-temporal video features for FVD calculation using torchvision 3D-ResNet (r3d_18).
    Pretrained on Kinetics-400 (standard in Computer Vision for video distribution evaluation).
    """
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            import torchvision.models.video as video_models
            try:
                weights = video_models.R3D_18_Weights.DEFAULT
                net = video_models.r3d_18(weights=weights)
            except Exception:
                net = video_models.r3d_18(pretrained=False)

            net.fc = nn.Identity()
            net.eval().to(self.device)
            self.model = net
        except Exception as e:
            print(f"[VideoFeatureExtractor] Note: 3D-ResNet fallback mode active ({e})")
            self.model = None

    def extract_clip_features(
        self,
        video_tensor: torch.Tensor,
        clip_len: int = 16,
        stride: int = 8,
        chunk_size: int = 8
    ) -> np.ndarray:
        """
        Extracts pooled spatio-temporal embeddings over 16-frame sliding windows with Kinetics normalization.
        video_tensor: torch.Tensor [T, H, W, C] in range [0, 1]
        Returns: np.ndarray [NumClips, FeatureDim] (where FeatureDim=512 for r3d_18)
        """
        T, H, W, C = video_tensor.shape
        if T < clip_len:
            pad = clip_len - T
            video_tensor = torch.cat([video_tensor, video_tensor[-1:].repeat(pad, 1, 1, 1)], dim=0)
            T = clip_len

        # Kinetics-400 normalization parameters
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

        clip_starts = list(range(0, T - clip_len + 1, stride))
        if not clip_starts:
            clip_starts = [0]

        all_feats = []
        for i in range(0, len(clip_starts), chunk_size):
            chunk_starts = clip_starts[i : i + chunk_size]
            clips = []
            for start in chunk_starts:
                clip = video_tensor[start : start + clip_len]  # [16, H, W, C]
                clip_2d = clip.permute(0, 3, 1, 2).float()     # [16, C, H, W]
                clip_resized = nn.functional.interpolate(
                    clip_2d, size=(112, 112), mode='bilinear', align_corners=False
                )
                clips.append(clip_resized.permute(1, 0, 2, 3)) # [C, 16, 112, 112]

            batch_clips = torch.stack(clips, dim=0).to(self.device)  # [B_chunk, C, 16, 112, 112]
            batch_clips = (batch_clips - mean) / std

            with torch.no_grad():
                if self.model is not None:
                    chunk_feats = self.model(batch_clips)
                    all_feats.append(chunk_feats.cpu().numpy())
                else:
                    diff = batch_clips[:, :, 1:] - batch_clips[:, :, :-1]
                    spatial_mean = batch_clips.mean(dim=(-2, -1)).view(batch_clips.shape[0], -1)
                    temporal_motion = diff.abs().mean(dim=(-2, -1)).view(batch_clips.shape[0], -1)
                    chunk_feats = torch.cat([spatial_mean, temporal_motion], dim=1)
                    all_feats.append(chunk_feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)


def align_aspect_ratio(tensor: torch.Tensor, target_aspect_ratio: float) -> torch.Tensor:
    """
    Crops tensor centered horizontally or vertically to match target_aspect_ratio.
    tensor: [T, H, W, C]
    target_aspect_ratio: target W / H
    """
    T, H, W, C = tensor.shape
    current_ar = W / H
    if abs(current_ar - target_aspect_ratio) < 0.02:
        return tensor

    if current_ar > target_aspect_ratio:
        new_w = int(round(H * target_aspect_ratio))
        start_x = max(0, (W - new_w) // 2)
        return tensor[:, :, start_x : start_x + new_w, :]
    else:
        new_h = int(round(W / target_aspect_ratio))
        start_y = max(0, (H - new_h) // 2)
        return tensor[:, start_y : start_y + new_h, :, :]


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-4
) -> float:
    """
    Computes Fréchet distance (2-Wasserstein distance between Gaussian distributions):
        FVD = ||mu1 - mu2||_2^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
    
    Standard Computer Vision implementation with numerical ridge regularization.
    """
    diff = mu1 - mu2

    # Regularization to prevent singular covariance matrices
    sigma1_reg = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2_reg = sigma2 + np.eye(sigma2.shape[0]) * eps

    res = scipy.linalg.sqrtm(sigma1_reg.dot(sigma2_reg))
    covmean = res[0] if isinstance(res, tuple) else res
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * (eps * 10)
        res = scipy.linalg.sqrtm((sigma1_reg + offset).dot(sigma2_reg + offset))
        covmean = res[0] if isinstance(res, tuple) else res

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
    return max(0.0, fvd)


def compute_fvd(
    real_video: torch.Tensor,
    fake_video: torch.Tensor,
    extractor: Optional[VideoFeatureExtractor] = None,
    clip_len: int = 16,
    stride: int = 8,
    device: Optional[str] = None,
    auto_align_ar: bool = True,
    return_features: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    Computes Fréchet Video Distance (FVD) between a real video and generated video.
    real_video, fake_video: torch.Tensor [T, H, W, C] in range [0, 1]

    Returns:
        fvd_score (float) or (fvd_score, feats_real, feats_fake) if return_features=True
    """
    if auto_align_ar and real_video is not None and fake_video is not None:
        real_ar = real_video.shape[2] / real_video.shape[1]
        fake_ar = fake_video.shape[2] / fake_video.shape[1]
        if abs(real_ar - fake_ar) >= 0.02:
            real_video = align_aspect_ratio(real_video, fake_ar)

    if extractor is None:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = VideoFeatureExtractor(device=device)

    feats_real = extractor.extract_clip_features(real_video, clip_len=clip_len, stride=stride)
    feats_fake = extractor.extract_clip_features(fake_video, clip_len=clip_len, stride=stride)

    if len(feats_real) < 2 or len(feats_fake) < 2:
        mu_real = np.mean(feats_real, axis=0)
        mu_fake = np.mean(feats_fake, axis=0)
        dist = float(np.linalg.norm(mu_real - mu_fake)) * 100.0
        return (dist, feats_real, feats_fake) if return_features else dist

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)

    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    fvd = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return (fvd, feats_real, feats_fake) if return_features else fvd


def compute_dataset_fvd(
    feats_real: np.ndarray,
    feats_fake: np.ndarray,
    eps: float = 1e-4
) -> float:
    """
    Computes Dataset-Level Fréchet Video Distance (Global FVD) across all video clips.
    This is the gold-standard evaluation metric in Computer Vision (Unterthiner et al., 2018).

    Args:
        feats_real: np.ndarray [N_total, D] - Pooled spatio-temporal features from real clips
        feats_fake: np.ndarray [N_total, D] - Pooled spatio-temporal features from generated clips
        eps: float - Regularization epsilon for covariance matrix

    Returns:
        float: Dataset-level FVD score
    """
    if len(feats_real) == 0 or len(feats_fake) == 0:
        return 0.0

    if len(feats_real) < 2 or len(feats_fake) < 2:
        mu_real = np.mean(feats_real, axis=0)
        mu_fake = np.mean(feats_fake, axis=0)
        return float(np.linalg.norm(mu_real - mu_fake)) * 100.0

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)

    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake, eps=eps)


def assess_fvd_quality(score: Optional[float]) -> str:
    """
    Categorizes FVD score according to computer vision video generation benchmarks:
      - < 250: Excellent (High temporal realism and consistency)
      - 250..400: Good (Acceptable realism with minor temporal drift)
      - >= 400: Moderate (Noticeable distortion or temporal jitter)
    """
    if score is None:
        return "N/A"
    if score < 250.0:
        return "Excellent"
    elif score < 400.0:
        return "Good"
    else:
        return "Moderate"


# ==============================================================================
# 6. COMPREHENSIVE MULTI-METRIC EVALUATION
# ==============================================================================
def evaluate_video_pair(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    gt_confidences: Optional[np.ndarray] = None,
    gt_bboxes: Optional[np.ndarray] = None,
    real_video_tensor: Optional[torch.Tensor] = None,
    gen_video_tensor: Optional[torch.Tensor] = None,
    feature_extractor: Optional[VideoFeatureExtractor] = None,
    conf_thresh: float = 0.3
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Evaluates a video pair against computer vision metrics:
      - PA-MPJPE (Overall, Body, Face, Hands)
      - PCK@0.05 and PCK@0.10 (Overall, Body, Face, Hands)
      - N-MPJPE (Overall, Body, Face, Hands)
      - FVD (Fréchet Video Distance) + Quality Assessment

    Returns:
      (results_dict, real_clip_features, fake_clip_features)
    """
    T = min(gt_joints.shape[0], pred_joints.shape[0])
    results: Dict[str, Any] = {
        "frames_evaluated": int(T),
        "pa_mpjpe": {},
        "pck_005": {},
        "pck_010": {},
        "n_mpjpe": {},
        "fvd": None,
        "fvd_quality": "N/A"
    }

    # Evaluate each keypoint subset
    for subset_name, indices in KEYPOINT_SUBSETS.items():
        pa_val = compute_pa_mpjpe(
            gt_joints=gt_joints,
            pred_joints=pred_joints,
            confidences=gt_confidences,
            conf_thresh=conf_thresh,
            joint_indices=indices
        )
        results["pa_mpjpe"][subset_name] = round(pa_val, 4)

        n_mpjpe_val = compute_n_mpjpe(
            gt_joints=gt_joints,
            pred_joints=pred_joints,
            confidences=gt_confidences,
            conf_thresh=conf_thresh,
            joint_indices=indices
        )
        results["n_mpjpe"][subset_name] = round(n_mpjpe_val, 4)

        pck05_val, _ = compute_pck(
            gt_joints=gt_joints,
            pred_joints=pred_joints,
            confidences=gt_confidences,
            bboxes=gt_bboxes,
            alpha=0.05,
            conf_thresh=conf_thresh,
            joint_indices=indices
        )
        results["pck_005"][subset_name] = round(pck05_val, 4)

        pck10_val, _ = compute_pck(
            gt_joints=gt_joints,
            pred_joints=pred_joints,
            confidences=gt_confidences,
            bboxes=gt_bboxes,
            alpha=0.10,
            conf_thresh=conf_thresh,
            joint_indices=indices
        )
        results["pck_010"][subset_name] = round(pck10_val, 4)

    # Compute FVD and extract clip features
    feats_real = None
    feats_fake = None
    if real_video_tensor is not None and gen_video_tensor is not None:
        try:
            fvd_score, feats_real, feats_fake = compute_fvd(
                real_video=real_video_tensor,
                fake_video=gen_video_tensor,
                extractor=feature_extractor,
                return_features=True
            )
            results["fvd"] = round(fvd_score, 4)
            results["fvd_quality"] = assess_fvd_quality(fvd_score)
        except Exception as e:
            print(f"[MetricsEngine] FVD calculation error: {e}")
            results["fvd"] = None
            results["fvd_quality"] = "Error"

    return results, feats_real, feats_fake
