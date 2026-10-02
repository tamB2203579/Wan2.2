import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union

KEYPOINT_SUBSETS = {
    "overall": list(range(133)),
    "body": list(range(17)),
    "face": list(range(23, 91)),
    "hands": list(range(91, 133)),
}
EVAL_SUBSETS = KEYPOINT_SUBSETS



def _procrustes_align(Y_m: np.ndarray, X_m: np.ndarray) -> np.ndarray:
    """Aligns X_m to Y_m via optimal similarity transform (scale, rotation, translation)."""
    mu_Y = np.mean(Y_m, axis=0, keepdims=True)
    mu_X = np.mean(X_m, axis=0, keepdims=True)
    Y_c = Y_m - mu_Y
    X_c = X_m - mu_X

    norm_Y = np.linalg.norm(Y_c)
    norm_X = np.linalg.norm(X_c)
    if norm_X < 1e-6 or norm_Y < 1e-6:
        return X_m

    H = (X_c / norm_X).T @ (Y_c / norm_Y)
    U, S, Vt = np.linalg.svd(H)
    d = np.ones(2)
    if np.linalg.det(U @ Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    s = np.sum(S * d) * (norm_Y / norm_X)
    t = mu_Y - s * (mu_X @ R)
    return s * (X_m @ R) + t


def compute_pa_mpjpe(
    gt: np.ndarray,
    pred: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_thresh: float = 0.3,
    joint_indices: Optional[List[int]] = None
) -> float:
    T = min(gt.shape[0], pred.shape[0])
    if T == 0:
        return 0.0

    indices = joint_indices or list(range(gt.shape[1]))
    errors = []

    for t in range(T):
        Y = gt[t, indices]
        X = pred[t, indices]

        mask = conf[t, indices] >= conf_thresh if conf is not None else np.ones(len(indices), dtype=bool)
        if np.sum(mask) < 3:
            mask = np.ones(len(indices), dtype=bool)

        Y_m, X_m = Y[mask], X[mask]
        if len(Y_m) < 3:
            continue

        X_aligned = _procrustes_align(Y_m, X_m)
        errors.extend(np.linalg.norm(X_aligned - Y_m, axis=-1).tolist())

    return float(np.mean(errors)) if errors else 0.0


def compute_raw_mpjpe(
    gt: np.ndarray,
    pred: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_thresh: float = 0.3,
    joint_indices: Optional[List[int]] = None
) -> float:
    """Compute raw Mean Per Joint Position Error without Procrustes alignment."""
    T = min(gt.shape[0], pred.shape[0])
    if T == 0:
        return 0.0

    indices = joint_indices or list(range(gt.shape[1]))
    errors = []

    for t in range(T):
        Y = gt[t, indices]
        X = pred[t, indices]

        mask = conf[t, indices] >= conf_thresh if conf is not None else np.ones(len(indices), dtype=bool)
        if np.sum(mask) < 1:
            mask = np.ones(len(indices), dtype=bool)

        errors.extend(np.linalg.norm(X[mask] - Y[mask], axis=-1).tolist())

    return float(np.mean(errors)) if errors else 0.0


def compute_fve(
    gt: np.ndarray,
    pred: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_thresh: float = 0.3,
    procrustes_align: bool = True
) -> float:
    """Compute First-order Velocity Error between successive keyframe displacements."""
    T = min(gt.shape[0], pred.shape[0])
    if T < 2:
        return 0.0

    v_gt = gt[1:] - gt[:-1]
    if procrustes_align:
        aligned_pred = np.zeros_like(pred)
        for t in range(T):
            aligned_pred[t] = _procrustes_align(gt[t], pred[t])
        v_pred = aligned_pred[1:] - aligned_pred[:-1]
    else:
        v_pred = pred[1:] - pred[:-1]

    diff = np.linalg.norm(v_pred - v_gt, axis=-1)
    if conf is not None:
        mask = (conf[1:] >= conf_thresh) & (conf[:-1] >= conf_thresh)
        if np.sum(mask) > 0:
            return float(np.mean(diff[mask]))
    return float(np.mean(diff))



def compute_n_mpjpe(
    gt: np.ndarray,
    pred: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_thresh: float = 0.3,
    joint_indices: Optional[List[int]] = None
) -> float:
    T = min(gt.shape[0], pred.shape[0])
    if T == 0:
        return 0.0

    indices = joint_indices or list(range(gt.shape[1]))
    errors = []

    for t in range(T):
        Y = gt[t, indices]
        X = pred[t, indices]

        mask = conf[t, indices] >= conf_thresh if conf is not None else np.ones(len(indices), dtype=bool)
        if np.sum(mask) < 3:
            mask = np.ones(len(indices), dtype=bool)

        Y_m, X_m = Y[mask], X[mask]
        if len(Y_m) < 3:
            continue

        scale_Y = max(float(np.max(np.ptp(Y_m, axis=0))), 1e-4)
        scale_X = max(float(np.max(np.ptp(X_m, axis=0))), 1e-4)

        Y_norm = (Y_m - np.mean(Y_m, axis=0, keepdims=True)) / scale_Y
        X_norm = (X_m - np.mean(X_m, axis=0, keepdims=True)) / scale_X
        errors.extend(np.linalg.norm(X_norm - Y_norm, axis=-1).tolist())

    return float(np.mean(errors)) if errors else 0.0


def compute_pck(
    gt: np.ndarray,
    pred: np.ndarray,
    conf: Optional[np.ndarray] = None,
    bboxes: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    conf_thresh: float = 0.3,
    joint_indices: Optional[List[int]] = None,
    align: bool = True
) -> Tuple[float, int]:
    T = min(gt.shape[0], pred.shape[0])
    if T == 0:
        return 0.0, 0

    indices = joint_indices or list(range(gt.shape[1]))
    correct = 0
    total = 0

    for t in range(T):
        Y = gt[t, indices]
        X = pred[t, indices]

        mask = conf[t, indices] >= conf_thresh if conf is not None else np.ones(len(indices), dtype=bool)
        if np.sum(mask) < 3:
            mask = np.ones(len(indices), dtype=bool)

        Y_m, X_m = Y[mask], X[mask]
        if len(Y_m) < 3:
            continue

        scale = None
        if bboxes is not None and t < len(bboxes) and bboxes[t] is not None and len(bboxes[t]) >= 4:
            scale = max(abs(bboxes[t][2] - bboxes[t][0]), abs(bboxes[t][3] - bboxes[t][1]))

        if scale is None or scale <= 1e-4:
            scale = max(float(np.max(np.ptp(Y_m, axis=0))), 100.0)

        threshold = alpha * scale
        X_eval = _procrustes_align(Y_m, X_m) if align else X_m

        dists = np.linalg.norm(X_eval - Y_m, axis=-1)
        correct += int(np.sum(dists <= threshold))
        total += len(dists)

    score = float((correct / total) * 100.0) if total > 0 else 0.0
    return score, total


class VideoFeatureExtractor:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = None
        self._load_backbone()

    def _load_backbone(self):
        try:
            import torchvision.models.video as vm
            weights = getattr(vm.R3D_18_Weights, "DEFAULT", None)
            net = vm.r3d_18(weights=weights) if weights else vm.r3d_18(pretrained=False)
            net.fc = nn.Identity()
            net.eval().to(self.device)
            self.model = net
        except Exception:
            self.model = None

    def extract_features(
        self,
        video_tensor: torch.Tensor,
        clip_len: int = 16,
        stride: int = 8,
        batch_size: int = 8
    ) -> np.ndarray:
        T, H, W, C = video_tensor.shape
        if T < clip_len:
            pad = clip_len - T
            video_tensor = torch.cat([video_tensor, video_tensor[-1:].repeat(pad, 1, 1, 1)], dim=0)
            T = clip_len

        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

        starts = list(range(0, T - clip_len + 1, stride)) or [0]
        feats = []

        for i in range(0, len(starts), batch_size):
            chunk_starts = starts[i : i + batch_size]
            clips = []
            for s in chunk_starts:
                c = video_tensor[s : s + clip_len].permute(0, 3, 1, 2).float()
                c_res = nn.functional.interpolate(c, size=(112, 112), mode="bilinear", align_corners=False)
                clips.append(c_res.permute(1, 0, 2, 3))

            batch = (torch.stack(clips, dim=0).to(self.device) - mean) / std
            with torch.no_grad():
                if self.model is not None:
                    out = self.model(batch)
                else:
                    diff = batch[:, :, 1:] - batch[:, :, :-1]
                    s_mean = batch.mean(dim=(-2, -1)).view(batch.shape[0], -1)
                    t_diff = diff.abs().mean(dim=(-2, -1)).view(batch.shape[0], -1)
                    out = torch.cat([s_mean, t_diff], dim=1)
                feats.append(out.cpu().numpy())

        return np.concatenate(feats, axis=0)


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-4
) -> float:
    diff = mu1 - mu2
    s1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    s2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    res = scipy.linalg.sqrtm(s1.dot(s2))
    covmean = res[0] if isinstance(res, tuple) else res
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * (eps * 10)
        res = scipy.linalg.sqrtm((s1 + offset).dot(s2 + offset))
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
    return_features: bool = False,
    device: Optional[str] = None
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    if extractor is None:
        extractor = VideoFeatureExtractor(device=device or ("cuda" if torch.cuda.is_available() else "cpu"))

    feats_real = extractor.extract_features(real_video, clip_len=clip_len, stride=stride)
    feats_fake = extractor.extract_features(fake_video, clip_len=clip_len, stride=stride)

    if len(feats_real) < 2 or len(feats_fake) < 2:
        dist = float(np.linalg.norm(np.mean(feats_real, axis=0) - np.mean(feats_fake, axis=0))) * 100.0
        return (dist, feats_real, feats_fake) if return_features else dist

    fvd = calculate_frechet_distance(
        np.mean(feats_real, axis=0),
        np.cov(feats_real, rowvar=False),
        np.mean(feats_fake, axis=0),
        np.cov(feats_fake, rowvar=False)
    )
    return (fvd, feats_real, feats_fake) if return_features else fvd


def compute_dataset_fvd(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    if len(feats_real) < 2 or len(feats_fake) < 2:
        return 0.0
    return calculate_frechet_distance(
        np.mean(feats_real, axis=0),
        np.cov(feats_real, rowvar=False),
        np.mean(feats_fake, axis=0),
        np.cov(feats_fake, rowvar=False)
    )


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0}
    n = len(values)
    mean_val = float(np.mean(values))
    var_val = float(np.var(values, ddof=1)) if n > 1 else 0.0
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return {
        "mean": round(mean_val, 4),
        "variance": round(var_val, 4),
        "std": round(std_val, 4)
    }


def compute_aggregate_stats(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics_list:
        return {}

    metric_keys = ["pa_mpjpe", "pck_005", "pck_010", "n_mpjpe"]
    subsets = list(KEYPOINT_SUBSETS.keys())
    stats = {
        "mean": {k: {} for k in metric_keys},
        "variance": {k: {} for k in metric_keys},
        "std": {k: {} for k in metric_keys}
    }

    for k in metric_keys:
        for s in subsets:
            vals = [m[k][s] for m in metrics_list if k in m and s in m[k]]
            res = compute_stats(vals)
            stats["mean"][k][s] = res["mean"]
            stats["variance"][k][s] = res["variance"]
            stats["std"][k][s] = res["std"]

    fvd_vals = [m["fvd"] for m in metrics_list if m.get("fvd") is not None]
    fvd_res = compute_stats(fvd_vals) if fvd_vals else None
    stats["mean"]["fvd"] = fvd_res["mean"] if fvd_res else None
    stats["variance"]["fvd"] = fvd_res["variance"] if fvd_res else None
    stats["std"]["fvd"] = fvd_res["std"] if fvd_res else None
    return stats


def evaluate_pair(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    gt_confidences: Optional[np.ndarray] = None,
    gt_bboxes: Optional[np.ndarray] = None,
    real_video_tensor: Optional[torch.Tensor] = None,
    gen_video_tensor: Optional[torch.Tensor] = None,
    feature_extractor: Optional[VideoFeatureExtractor] = None,
    conf_thresh: float = 0.3
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    T = min(gt_joints.shape[0], pred_joints.shape[0])
    results = {
        "frames_evaluated": int(T),
        "pa_mpjpe": {},
        "pck_005": {},
        "pck_010": {},
        "n_mpjpe": {},
        "fvd": None,
        "fvd_quality": "N/A"
    }

    for subset, indices in KEYPOINT_SUBSETS.items():
        results["pa_mpjpe"][subset] = round(compute_pa_mpjpe(
            gt_joints, pred_joints, gt_confidences, conf_thresh, indices
        ), 4)

        results["n_mpjpe"][subset] = round(compute_n_mpjpe(
            gt_joints, pred_joints, gt_confidences, conf_thresh, indices
        ), 4)

        pck05, _ = compute_pck(
            gt_joints, pred_joints, gt_confidences, gt_bboxes, 0.05, conf_thresh, indices, align=True
        )
        results["pck_005"][subset] = round(pck05, 4)

        pck10, _ = compute_pck(
            gt_joints, pred_joints, gt_confidences, gt_bboxes, 0.10, conf_thresh, indices, align=True
        )
        results["pck_010"][subset] = round(pck10, 4)

    feats_real, feats_fake = None, None
    if real_video_tensor is not None and gen_video_tensor is not None:
        try:
            fvd, feats_real, feats_fake = compute_fvd(
                real_video_tensor, gen_video_tensor, extractor=feature_extractor, return_features=True
            )
            results["fvd"] = round(fvd, 4)
            results["fvd_quality"] = "Excellent" if fvd < 250 else ("Good" if fvd < 400 else "Moderate")
        except Exception:
            results["fvd"] = None

    return results, feats_real, feats_fake
