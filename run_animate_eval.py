#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
End-to-End Automation & Evaluation Script for Wan2.2 Animate.

Iterates through driving videos in Original_9x16/, animates each video using the reference
female avatar (avatar_nu.png), and evaluates generation fidelity across Computer Vision metrics:
  1. FVD (Fréchet Video Distance - Unterthiner et al., 2018):
     Evaluates video distribution realism and temporal continuity using spatio-temporal 3D-CNN.
     Computes both per-video FVD and Global Dataset-Level FVD.
  2. PA-MPJPE (Procrustes-Aligned MPJPE in pixels)
  3. PCK@0.05 and PCK@0.10 (Percentage of Correct Keypoints)
  4. N-MPJPE (Scale-Normalized Mean Per Joint Position Error)
Reports metrics across 4 keypoint subsets (Overall, Body & Arms, Face, Hands).
Outputs per-video JSON, metrics_summary.json, metrics_summary.csv, and terminal summary tables.
"""

import os
import sys
import glob
import json
import csv
import time
import shutil
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

# Ensure UTF-8 stdout on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Append Wan2.2 root and preprocess directories to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PREPROCESS_DIR = os.path.join(SCRIPT_DIR, "wan", "modules", "animate", "preprocess")
if PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, PREPROCESS_DIR)

# Local imports
from metrics_engine import (
    evaluate_video_pair,
    KEYPOINT_SUBSETS,
    VideoFeatureExtractor,
    compute_fvd,
    compute_dataset_fvd,
    assess_fvd_quality,
    compute_pa_mpjpe,
    compute_n_mpjpe,
    compute_pck
)
from video_io_utils import (
    read_image_unicode,
    write_image_unicode,
    safe_open_video_capture,
    load_video_frames,
    frames_to_video_tensor,
    extract_joints_from_metas
)


# ==============================================================================
# 1. ARGUMENT PARSER & CONFIGURATION
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan2.2 Animate Batch Inference and Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "Original_9x16"),
        help="Directory containing driving sign language videos (.mp4)."
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default=r"D:\git\signbridge-3d-vsl\Model\avatar_nu.png",
        help="Path to the reference avatar image (female avatar)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_animate",
        help="Root directory for preprocessed data, generated videos, and metrics."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./Wan2.2-Animate-14B",
        help="Directory containing Wan2.2-Animate-14B model checkpoints."
    )
    parser.add_argument(
        "--preprocess_ckpt_dir",
        type=str,
        default="./process_checkpoint",
        help="Directory containing YOLO and ViTPose preprocessing models (e.g. ./process_checkpoint or ./preprocess_ckpts)."
    )

    # Runtime & Execution
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Target device for inference and evaluation (e.g. cuda:0 or cpu)."
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=77,
        help="Clip length for animation generation (must be 4n+1, e.g. 77)."
    )
    parser.add_argument(
        "--refert_num",
        type=int,
        default=1,
        help="Number of overlapping reference frames across chunked clips."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for generated video and preprocessing. -1 uses source FPS."
    )
    parser.add_argument(
        "--resolution_area",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="Target resolution area [width, height]."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=20,
        help="Number of diffusion sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=5.0,
        help="Sampling shift parameter for flow solver."
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="Flow matching solver."
    )
    parser.add_argument(
        "--offload_model",
        action="store_true",
        default=True,
        help="Offload text encoder / models to CPU to optimize VRAM."
    )
    parser.add_argument(
        "--no_offload_model",
        action="store_false",
        dest="offload_model",
        help="Keep all models permanently in VRAM."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Enable character replacement mode (requires SAM2)."
    )
    parser.add_argument(
        "--retarget_flag",
        action="store_true",
        default=False,
        help="Enable pose retargeting."
    )

    # Workflow Control Flags
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume execution: skip videos where metrics or outputs already exist."
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        default=False,
        help="Skip animation generation; run evaluation only on existing videos in output_dir."
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="Run video animation generation only; skip metrics calculation."
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit the maximum number of driving videos to process."
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.3,
        help="Confidence threshold for valid keypoints."
    )

    return parser.parse_args()


# ==============================================================================
# 2. RESOLVE REFERENCE AVATAR PATH
# ==============================================================================
def resolve_reference_avatar(ref_arg_path: str) -> str:
    """
    Validates and resolves the reference avatar path with fallback locations.
    """
    candidates = [
        ref_arg_path,
        os.path.join(SCRIPT_DIR, "Model", "avatar_nu.png"),
        os.path.join(SCRIPT_DIR, "avatar_nu.png"),
        r"D:\git\signbridge-3d-vsl\Model\avatar_nu.png",
        r"D:\git\signbridge-3d-vsl\avatar_nu.png",
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)

    return os.path.abspath(ref_arg_path)


# ==============================================================================
# 3. PIPELINE INITIALIZATION & CHECKPOINT RESOLUTION
# ==============================================================================
def resolve_preprocess_checkpoints(
    preprocess_ckpt_dir: str,
    ckpt_dir: str = "",
    replace_flag: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """
    Auto-discovers det_ckpt, pose_ckpt, and sam_ckpt across common directory locations:
      1. preprocess_ckpt_dir
      2. process_checkpoint (official Hugging Face directory name in Wan2.2-Animate-14B)
      3. ckpt_dir / "process_checkpoint"
      4. ckpt_dir / "preprocess_ckpts"
      5. ckpt_dir directly
      6. SCRIPT_DIR / "process_checkpoint"
      7. SCRIPT_DIR / "preprocess_ckpts"
      8. SCRIPT_DIR / "preprocess_checkpoint"
      9. SCRIPT_DIR directly

    Returns:
      (det_ckpt, pose_ckpt, sam_ckpt, searched_directories)
    """
    candidate_dirs = [
        preprocess_ckpt_dir,
        os.path.join(preprocess_ckpt_dir, "process_checkpoint") if preprocess_ckpt_dir else "",
        os.path.join(preprocess_ckpt_dir, "preprocess_ckpts") if preprocess_ckpt_dir else "",
        os.path.join(ckpt_dir, "process_checkpoint") if ckpt_dir else "",
        os.path.join(ckpt_dir, "preprocess_ckpts") if ckpt_dir else "",
        ckpt_dir,
        os.path.join(SCRIPT_DIR, "process_checkpoint"),
        os.path.join(SCRIPT_DIR, "preprocess_ckpts"),
        os.path.join(SCRIPT_DIR, "preprocess_checkpoint"),
        SCRIPT_DIR,
    ]

    valid_dirs = []
    seen = set()
    for d in candidate_dirs:
        if d and os.path.isdir(d):
            ab = os.path.abspath(d)
            if ab not in seen:
                seen.add(ab)
                valid_dirs.append(ab)

    det_ckpt = None
    pose_ckpt = None
    sam_ckpt = None

    # 1. Search for det_ckpt (yolov10m.onnx)
    for d in valid_dirs:
        p1 = os.path.join(d, "det", "yolov10m.onnx")
        p2 = os.path.join(d, "yolov10m.onnx")
        if os.path.isfile(p1):
            det_ckpt = p1
            break
        elif os.path.isfile(p2):
            det_ckpt = p2
            break

    if det_ckpt is None:
        for d in valid_dirs:
            try:
                for match in Path(d).rglob("*yolov10*.onnx"):
                    if match.is_file():
                        det_ckpt = str(match.resolve())
                        break
                if det_ckpt:
                    break
            except Exception:
                pass

    # 2. Search for pose_ckpt (vitpose_h_wholebody.onnx or vitpose directory containing end2end.onnx)
    for d in valid_dirs:
        p1 = os.path.join(d, "pose2d", "vitpose_h_wholebody.onnx")
        p2 = os.path.join(d, "vitpose_h_wholebody.onnx")
        p3 = os.path.join(d, "pose2d", "end2end.onnx")
        p4 = os.path.join(d, "end2end.onnx")
        if os.path.exists(p1):
            pose_ckpt = p1
            break
        elif os.path.exists(p2):
            pose_ckpt = p2
            break
        elif os.path.isfile(p3):
            pose_ckpt = p3
            break
        elif os.path.isfile(p4):
            pose_ckpt = p4
            break

    if pose_ckpt is None:
        for d in valid_dirs:
            try:
                for match in Path(d).rglob("*vitpose_h_wholebody*"):
                    if match.exists():
                        pose_ckpt = str(match.resolve())
                        break
                if pose_ckpt:
                    break
            except Exception:
                pass

    # 3. Search for sam_ckpt (if replace_flag)
    if replace_flag:
        for d in valid_dirs:
            p1 = os.path.join(d, "sam2", "sam2_hiera_large.pt")
            p2 = os.path.join(d, "sam2_hiera_large.pt")
            if os.path.isfile(p1):
                sam_ckpt = p1
                break
            elif os.path.isfile(p2):
                sam_ckpt = p2
                break


    return det_ckpt, pose_ckpt, sam_ckpt, valid_dirs


def initialize_preprocessor(preprocess_ckpt_dir: str, ckpt_dir: str = "", replace_flag: bool = False, device: str = "cuda:0"):
    """
    Instantiates the ProcessPipeline for ViTPose keypoint detection and conditioning.
    """
    det_ckpt, pose_ckpt, sam_ckpt, searched_dirs = resolve_preprocess_checkpoints(
        preprocess_ckpt_dir=preprocess_ckpt_dir,
        ckpt_dir=ckpt_dir,
        replace_flag=replace_flag
    )

    if det_ckpt is None or pose_ckpt is None:
        missing = []
        if det_ckpt is None:
            missing.append("det/yolov10m.onnx (YOLO human detector)")
        if pose_ckpt is None:
            missing.append("pose2d/vitpose_h_wholebody.onnx (ViTPose WholeBody 133-keypoint estimator)")
        if replace_flag and sam_ckpt is None:
            missing.append("sam2/sam2_hiera_large.pt (SAM2 segmentation model)")

        print("\n" + "=" * 80)
        print("[ERROR] PREPROCESSING CHECKPOINTS MISSING!")
        print("=" * 80)
        print("Wan2.2 Animate requires ViTPose & YOLO ONNX models to extract poses from driving videos.")
        print(f"Missing component(s): {', '.join(missing)}")
        print("\nSearched directories:")
        for sd in searched_dirs:
            print(f"  - {sd}")
        print("\n" + "-" * 80)
        print("HOW TO RESOLVE THIS ERROR ON YOUR GPU SERVER:")
        print("-" * 80)
        print("1. If checkpoints are already downloaded in another directory:")
        print("   Pass the directory using --preprocess_ckpt_dir, for example:")
        print("   python run_animate_eval.py --preprocess_ckpt_dir /path/to/process_checkpoint ...")
        print("\n2. If not downloaded yet, download from the official Hugging Face Wan2.2-Animate repo:")
        print("   huggingface-cli download Wan-AI/Wan2.2-Animate-14B --include \"process_checkpoint/*\" --local-dir ./Wan2.2-Animate-14B")
        print("   (Then run with: --preprocess_ckpt_dir ./Wan2.2-Animate-14B/process_checkpoint)")
        print("\n3. If you already have pre-generated videos in the output folder and only want metrics:")
        print("   Run with: --skip_inference")
        print("=" * 80 + "\n")
        return None

    print(f"[Init] Found preprocessing checkpoints:")
    print(f"  - Detection: {det_ckpt}")
    print(f"  - Pose:      {pose_ckpt}")
    if replace_flag and sam_ckpt:
        print(f"  - SAM2:      {sam_ckpt}")

    try:
        import onnxruntime
    except ImportError:
        print("\n" + "=" * 80)
        print("[ERROR] MISSING DEPENDENCY: onnxruntime / onnxruntime-gpu")
        print("=" * 80)
        print("ViTPose (pose estimation) and YOLO (human detection) require ONNX Runtime.")
        print("Please install onnxruntime-gpu in your environment on the GPU server:")
        print("    pip install onnxruntime-gpu")
        print("=" * 80 + "\n")
        return None

    try:
        try:
            from process_pipepline import ProcessPipeline
        except ImportError:
            from wan.modules.animate.preprocess.process_pipepline import ProcessPipeline

        print(f"[Init] Initializing ProcessPipeline with ViTPose & YOLO (device: {device})...")
        pipeline = ProcessPipeline(
            det_checkpoint_path=det_ckpt,
            pose2d_checkpoint_path=pose_ckpt,
            sam_checkpoint_path=sam_ckpt,
            flux_kontext_path=None
        )
        if hasattr(pipeline, "pose2d"):
            try:
                pipeline.pose2d.set_device(device)
            except Exception:
                pass
        return pipeline
    except Exception as e:
        print(f"\n[Warning] Failed initializing ProcessPipeline with {device}: {e}")
        traceback.print_exc()
        try:
            print("[Init] Retrying ProcessPipeline initialization on CPU...")
            pipeline = ProcessPipeline(
                det_checkpoint_path=det_ckpt,
                pose2d_checkpoint_path=pose_ckpt,
                sam_checkpoint_path=sam_ckpt,
                flux_kontext_path=None
            )
            if hasattr(pipeline, "pose2d"):
                pipeline.pose2d.set_device("cpu")
            return pipeline
        except Exception as e2:
            print(f"[Error] Fallback to CPU also failed: {e2}")
            traceback.print_exc()
            return None


def initialize_wan_animate(ckpt_dir: str, offload_model: bool = True, device: str = "cuda:0"):
    """
    Instantiates the WanAnimate pipeline once for all video generations.
    """
    candidates = [
        ckpt_dir,
        os.path.join(SCRIPT_DIR, "Wan2.2-Animate-14B"),
        "./Wan2.2-Animate-14B",
        "../Wan2.2-Animate-14B",
        os.path.join(SCRIPT_DIR, "Wan2.2-T2V-A14B")
    ]
    resolved_dir = None
    for cand in candidates:
        if cand and os.path.isdir(cand):
            resolved_dir = os.path.abspath(cand)
            break

    if resolved_dir is None:
        print(f"[Warning] WanAnimate checkpoint directory not found (searched: {ckpt_dir}, ./Wan2.2-Animate-14B)")
        return None

    ckpt_dir = resolved_dir

    try:
        from easydict import EasyDict
        from wan.configs.shared_config import wan_shared_cfg
        cfg = EasyDict(__name__='Config: Wan animate 14B')
        cfg.update(wan_shared_cfg)
        cfg.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
        cfg.t5_tokenizer = 'google/umt5-xxl'
        cfg.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
        cfg.clip_tokenizer = 'xlm-roberta-large'
        cfg.lora_checkpoint = 'relighting_lora.ckpt'
        cfg.vae_checkpoint = 'Wan2.1_VAE.pth'
        cfg.vae_stride = (4, 8, 8)
        cfg.patch_size = (1, 2, 2)
        cfg.dim = 5120
        cfg.ffn_dim = 13824
        cfg.freq_dim = 256
        cfg.num_heads = 40
        cfg.num_layers = 40
        cfg.window_size = (-1, -1)
        cfg.qk_norm = True
        cfg.cross_attn_norm = True
        cfg.eps = 1e-6
        cfg.use_face_encoder = True
        cfg.motion_encoder_dim = 512
        cfg.sample_shift = 5.0
        cfg.sample_steps = 20
        cfg.sample_guide_scale = 1.0
        cfg.frame_num = 77
        cfg.sample_fps = 30
        cfg.prompt = '视频中的人在做动作'

        from wan.animate import WanAnimate
        device_id = 0
        if ":" in device:
            try:
                device_id = int(device.split(":")[-1])
            except Exception:
                device_id = 0

        print(f"[Init] Initializing WanAnimate 14B pipeline from {ckpt_dir} (device: {device_id})...")
        wan_animate = WanAnimate(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device_id,
            t5_cpu=offload_model,
            offload_model=offload_model
        )
        return wan_animate
    except Exception as e:
        print(f"[Warning] Failed to instantiate WanAnimate model: {e}")
        return None


# ==============================================================================
# 4. PREPROCESSING PER VIDEO
# ==============================================================================
def run_video_preprocessing(
    preprocessor,
    video_path: str,
    ref_image_path: str,
    output_prep_dir: str,
    resolution_area: List[int],
    fps: int,
    replace_flag: bool = False,
    retarget_flag: bool = False
) -> bool:
    """
    Runs ProcessPipeline to create src_pose.mp4, src_face.mp4, and src_ref.png in output_prep_dir.
    """
    os.makedirs(output_prep_dir, exist_ok=True)
    required_files = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
    if all(os.path.exists(os.path.join(output_prep_dir, f)) for f in required_files):
        print(f"[Preprocess] Cached preprocessing data found at {output_prep_dir}")
        return True

    if preprocessor is None:
        raise RuntimeError(
            f"Cannot preprocess '{os.path.basename(video_path)}': Preprocessor (ProcessPipeline) is not initialized.\n"
            f"Please verify that ViTPose and YOLO checkpoints exist (pass --preprocess_ckpt_dir <path>), "
            f"or use --skip_inference if animated videos are already generated."
        )

    print(f"[Preprocess] Extracting poses and face crops: {os.path.basename(video_path)} -> {output_prep_dir}")
    success = preprocessor(
        video_path=video_path,
        refer_image_path=ref_image_path,
        output_path=output_prep_dir,
        resolution_area=resolution_area,
        fps=fps,
        replace_flag=replace_flag,
        retarget_flag=retarget_flag
    )
    return bool(success)


# ==============================================================================
# 5. GENERATION INFERENCE
# ==============================================================================
def run_video_generation(
    wan_model,
    preprocessed_dir: str,
    output_video_path: str,
    clip_len: int = 77,
    refert_num: int = 1,
    sample_shift: float = 5.0,
    sample_solver: str = "unipc",
    sample_steps: int = 20,
    sample_guide_scale: float = 1.0,
    base_seed: int = 42,
    offload_model: bool = True,
    replace_flag: bool = False,
    fps: int = 30
) -> bool:
    """
    Generates animated video from preprocessed directory using WanAnimate.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1000:
        print(f"[Generate] Output video already exists: {output_video_path}")
        return True

    if wan_model is None:
        raise RuntimeError("WanAnimate model is not initialized!")

    print(f"[Generate] Generating animation: {preprocessed_dir} -> {output_video_path}")
    video_tensor = wan_model.generate(
        src_root_path=preprocessed_dir,
        replace_flag=replace_flag,
        refert_num=refert_num,
        clip_len=clip_len,
        shift=sample_shift,
        sample_solver=sample_solver,
        sampling_steps=sample_steps,
        guide_scale=sample_guide_scale,
        seed=base_seed,
        offload_model=offload_model
    )

    if video_tensor is None:
        return False

    from wan.utils.utils import save_video
    save_video(
        tensor=video_tensor[None],
        save_file=output_video_path,
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    return True


# ==============================================================================
# 6. EVALUATION LOGIC
# ==============================================================================
def evaluate_generated_video(
    gt_video_path: str,
    gen_video_path: str,
    pose2d_model,
    feature_extractor: Optional[VideoFeatureExtractor],
    conf_thresh: float = 0.3,
    precomputed_gt_metas: Optional[List[dict]] = None
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Evaluates generated video against ground truth video.
    Extracts keypoints using ViTPose (Pose2d) and computes PA-MPJPE, PCK, N-MPJPE, and FVD.
    Returns (metrics, real_clip_features, fake_clip_features).
    """
    print(f"[Eval] Evaluating generated video: {os.path.basename(gen_video_path)}")

    # 1. Load ground truth frames and keypoints
    gt_frames, gt_fps = load_video_frames(gt_video_path)
    if precomputed_gt_metas is not None:
        gt_metas = precomputed_gt_metas
    elif pose2d_model is not None:
        print(f"[Eval] Running ViTPose on GT video ({len(gt_frames)} frames)...")
        gt_metas = pose2d_model(gt_frames)
    else:
        raise RuntimeError("No pose model or precomputed metas available for GT keypoints!")

    gt_joints, gt_confs, gt_bboxes = extract_joints_from_metas(gt_metas)

    # 2. Load generated video frames and predict keypoints
    gen_frames, gen_fps = load_video_frames(gen_video_path)
    if pose2d_model is not None:
        print(f"[Eval] Running ViTPose on generated video ({len(gen_frames)} frames)...")
        gen_metas = pose2d_model(gen_frames)
        gen_joints, gen_confs, _ = extract_joints_from_metas(gen_metas)
    else:
        raise RuntimeError("ViTPose model required to evaluate generated video keypoints!")

    # 3. Align frame count
    min_len = min(len(gt_frames), len(gen_frames))
    gt_frames = gt_frames[:min_len]
    gen_frames = gen_frames[:min_len]
    gt_joints = gt_joints[:min_len]
    gen_joints = gen_joints[:min_len]
    gt_confs = gt_confs[:min_len]
    gt_bboxes = gt_bboxes[:min_len]

    # Convert to tensors for FVD
    gt_tensor = frames_to_video_tensor(gt_frames)
    gen_tensor = frames_to_video_tensor(gen_frames)

    # 4. Compute all metrics and extract clip features
    metrics, feats_real, feats_fake = evaluate_video_pair(
        gt_joints=gt_joints,
        pred_joints=gen_joints,
        gt_confidences=gt_confs,
        gt_bboxes=gt_bboxes,
        real_video_tensor=gt_tensor,
        gen_video_tensor=gen_tensor,
        feature_extractor=feature_extractor,
        conf_thresh=conf_thresh
    )

    metrics["video_stem"] = Path(gt_video_path).stem
    metrics["gt_video"] = os.path.basename(gt_video_path)
    metrics["gen_video"] = os.path.basename(gen_video_path)
    metrics["fps"] = float(gt_fps)
    return metrics, feats_real, feats_fake


# ==============================================================================
# 7. SUMMARY & REPORTING UTILITIES
# ==============================================================================
def format_summary_table(
    video_metrics_list: List[Dict[str, Any]],
    mean_metrics: Dict[str, Any],
    global_dataset_fvd: Optional[float] = None
) -> str:
    """
    Constructs an ASCII formatted table for console display adhering to Computer Vision standards.
    """
    col_w_name = 18
    header_line = (
        f"{'Video Name':<{col_w_name}} "
        f"{'Frames':>6} "
        f"{'PA-MPJPE':>9} "
        f"{'PCK@0.05':>9} "
        f"{'PCK@0.10':>9} "
        f"{'N-MPJPE':>9} "
        f"{'FVD':>9} "
        f"{'Quality':<10}"
    )
    sep_line = "=" * len(header_line)
    sub_sep = "-" * len(header_line)

    lines = [sep_line, header_line, sub_sep]

    for item in video_metrics_list:
        v_name = item.get("video_stem", "unknown")[:col_w_name]
        frames = item.get("frames_evaluated", 0)
        pa = item.get("pa_mpjpe", {}).get("overall", 0.0)
        pck05 = item.get("pck_005", {}).get("overall", 0.0)
        pck10 = item.get("pck_010", {}).get("overall", 0.0)
        n_mpjpe = item.get("n_mpjpe", {}).get("overall", 0.0)
        fvd = item.get("fvd")
        fvd_str = f"{fvd:>9.2f}" if fvd is not None else f"{'N/A':>9}"
        quality = item.get("fvd_quality", "N/A")

        row = (
            f"{v_name:<{col_w_name}} "
            f"{frames:>6d} "
            f"{pa:>9.2f} "
            f"{pck05:>8.2f}% "
            f"{pck10:>8.2f}% "
            f"{n_mpjpe:>9.4f} "
            f"{fvd_str} "
            f"{quality:<10}"
        )
        lines.append(row)

    lines.append(sep_line)

    # Average per-video row
    mean_pa = mean_metrics.get("pa_mpjpe", {}).get("overall", 0.0)
    mean_pck05 = mean_metrics.get("pck_005", {}).get("overall", 0.0)
    mean_pck10 = mean_metrics.get("pck_010", {}).get("overall", 0.0)
    mean_n_mpjpe = mean_metrics.get("n_mpjpe", {}).get("overall", 0.0)
    mean_fvd = mean_metrics.get("fvd")
    mean_fvd_str = f"{mean_fvd:>9.2f}" if mean_fvd is not None else f"{'N/A':>9}"
    mean_quality = assess_fvd_quality(mean_fvd)

    mean_row = (
        f"{'AVERAGE (video)':<{col_w_name}} "
        f"{'':>6} "
        f"{mean_pa:>9.2f} "
        f"{mean_pck05:>8.2f}% "
        f"{mean_pck10:>8.2f}% "
        f"{mean_n_mpjpe:>9.4f} "
        f"{mean_fvd_str} "
        f"{mean_quality:<10}"
    )
    lines.append(mean_row)

    # Global Dataset FVD row (Computer Vision benchmark)
    if global_dataset_fvd is not None:
        dataset_quality = assess_fvd_quality(global_dataset_fvd)
        dataset_row = (
            f"{'GLOBAL DATASET FVD':<{col_w_name}} "
            f"{'':>6} "
            f"{'':>9} "
            f"{'':>9} "
            f"{'':>9} "
            f"{'':>9} "
            f"{global_dataset_fvd:>9.2f} "
            f"{dataset_quality:<10}"
        )
        lines.append(dataset_row)

    lines.append(sep_line)

    return "\n".join(lines)


def export_csv_summary(
    video_metrics_list: List[Dict[str, Any]],
    mean_metrics: Dict[str, Any],
    csv_path: str,
    global_dataset_fvd: Optional[float] = None
):
    """
    Exports full breakdown of metrics (including subsets and FVD quality) to a CSV file.
    """
    headers = [
        "Video", "Frames",
        "PA-MPJPE (Overall)", "PA-MPJPE (Body)", "PA-MPJPE (Face)", "PA-MPJPE (Hands)",
        "PCK@0.05 (Overall)", "PCK@0.05 (Body)", "PCK@0.05 (Face)", "PCK@0.05 (Hands)",
        "PCK@0.10 (Overall)", "PCK@0.10 (Body)", "PCK@0.10 (Face)", "PCK@0.10 (Hands)",
        "N-MPJPE (Overall)", "N-MPJPE (Body)", "N-MPJPE (Face)", "N-MPJPE (Hands)",
        "FVD", "FVD Assessment"
    ]

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for item in video_metrics_list:
            writer.writerow([
                item.get("video_stem", ""),
                item.get("frames_evaluated", 0),
                item.get("pa_mpjpe", {}).get("overall", ""),
                item.get("pa_mpjpe", {}).get("body", ""),
                item.get("pa_mpjpe", {}).get("face", ""),
                item.get("pa_mpjpe", {}).get("hands", ""),
                item.get("pck_005", {}).get("overall", ""),
                item.get("pck_005", {}).get("body", ""),
                item.get("pck_005", {}).get("face", ""),
                item.get("pck_005", {}).get("hands", ""),
                item.get("pck_010", {}).get("overall", ""),
                item.get("pck_010", {}).get("body", ""),
                item.get("pck_010", {}).get("face", ""),
                item.get("pck_010", {}).get("hands", ""),
                item.get("n_mpjpe", {}).get("overall", ""),
                item.get("n_mpjpe", {}).get("body", ""),
                item.get("n_mpjpe", {}).get("face", ""),
                item.get("n_mpjpe", {}).get("hands", ""),
                item.get("fvd", ""),
                item.get("fvd_quality", "")
            ])

        # Write Mean Row
        writer.writerow([
            "AVERAGE (per-video)",
            "",
            mean_metrics.get("pa_mpjpe", {}).get("overall", ""),
            mean_metrics.get("pa_mpjpe", {}).get("body", ""),
            mean_metrics.get("pa_mpjpe", {}).get("face", ""),
            mean_metrics.get("pa_mpjpe", {}).get("hands", ""),
            mean_metrics.get("pck_005", {}).get("overall", ""),
            mean_metrics.get("pck_005", {}).get("body", ""),
            mean_metrics.get("pck_005", {}).get("face", ""),
            mean_metrics.get("pck_005", {}).get("hands", ""),
            mean_metrics.get("pck_010", {}).get("overall", ""),
            mean_metrics.get("pck_010", {}).get("body", ""),
            mean_metrics.get("pck_010", {}).get("face", ""),
            mean_metrics.get("pck_010", {}).get("hands", ""),
            mean_metrics.get("n_mpjpe", {}).get("overall", ""),
            mean_metrics.get("n_mpjpe", {}).get("body", ""),
            mean_metrics.get("n_mpjpe", {}).get("face", ""),
            mean_metrics.get("n_mpjpe", {}).get("hands", ""),
            mean_metrics.get("fvd", ""),
            assess_fvd_quality(mean_metrics.get("fvd"))
        ])

        # Write Global Dataset FVD Row
        if global_dataset_fvd is not None:
            writer.writerow([
                "GLOBAL DATASET FVD",
                "",
                "", "", "", "",
                "", "", "", "",
                "", "", "", "",
                "", "", "", "",
                round(global_dataset_fvd, 4),
                assess_fvd_quality(global_dataset_fvd)
            ])


def compute_aggregate_means(video_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes average metrics across all evaluated videos.
    """
    if not video_metrics_list:
        return {}

    mean_results = {
        "pa_mpjpe": {},
        "pck_005": {},
        "pck_010": {},
        "n_mpjpe": {},
        "fvd": None
    }

    subsets = list(KEYPOINT_SUBSETS.keys())
    for s in subsets:
        pa_vals = [m["pa_mpjpe"][s] for m in video_metrics_list if "pa_mpjpe" in m and s in m["pa_mpjpe"]]
        mean_results["pa_mpjpe"][s] = round(float(np.mean(pa_vals)), 4) if pa_vals else 0.0

        pck05_vals = [m["pck_005"][s] for m in video_metrics_list if "pck_005" in m and s in m["pck_005"]]
        mean_results["pck_005"][s] = round(float(np.mean(pck05_vals)), 4) if pck05_vals else 0.0

        pck10_vals = [m["pck_010"][s] for m in video_metrics_list if "pck_010" in m and s in m["pck_010"]]
        mean_results["pck_010"][s] = round(float(np.mean(pck10_vals)), 4) if pck10_vals else 0.0

        n_vals = [m["n_mpjpe"][s] for m in video_metrics_list if "n_mpjpe" in m and s in m["n_mpjpe"]]
        mean_results["n_mpjpe"][s] = round(float(np.mean(n_vals)), 4) if n_vals else 0.0

    fvd_vals = [m["fvd"] for m in video_metrics_list if m.get("fvd") is not None]
    mean_results["fvd"] = round(float(np.mean(fvd_vals)), 4) if fvd_vals else None
    return mean_results


# ==============================================================================
# 8. MAIN ORCHESTRATION PIPELINE
# ==============================================================================
def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 80)
    print("  Wan2.2 Animate Batch Inference & Computer Vision Evaluation Pipeline")
    print("=" * 80)
    print(f"  Input Directory:         {args.input_dir}")
    print(f"  Reference Avatar:        {args.ref_image}")
    print(f"  Output Directory:        {args.output_dir}")
    print(f"  Model Checkpoints:       {args.ckpt_dir}")
    print(f"  Preprocess Checkpoints:  {args.preprocess_ckpt_dir}")
    print(f"  Device:                  {args.device}")
    print(f"  Skip Inference:          {args.skip_inference}")
    print(f"  Skip Evaluation:         {args.skip_eval}")
    print(f"  Resume Mode:             {args.resume}")
    print("=" * 80)

    # 1. Resolve Reference Image
    ref_image_path = resolve_reference_avatar(args.ref_image)
    if not os.path.exists(ref_image_path):
        print(f"[Error] Reference avatar not found: {ref_image_path}")
        sys.exit(1)
    print(f"[Setup] Using reference avatar: {ref_image_path}")

    # 2. Gather Driving Videos
    if not os.path.isdir(args.input_dir):
        print(f"[Error] Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    video_paths = sorted(
        glob.glob(os.path.join(args.input_dir, "*.mp4")) +
        glob.glob(os.path.join(args.input_dir, "*.mov")) +
        glob.glob(os.path.join(args.input_dir, "*.avi"))
    )
    if not video_paths:
        print(f"[Error] No video files found in {args.input_dir}")
        sys.exit(1)

    if args.max_videos is not None:
        video_paths = video_paths[:args.max_videos]

    print(f"[Setup] Discovered {len(video_paths)} driving video(s) to process.")

    # 3. Create Output Subdirectories
    output_videos_dir = os.path.join(args.output_dir, "videos")
    output_metrics_dir = os.path.join(args.output_dir, "metrics")
    output_prep_dir = os.path.join(args.output_dir, "preprocessed")
    os.makedirs(output_videos_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)
    os.makedirs(output_prep_dir, exist_ok=True)

    # 4. Initialize Models
    preprocessor = None
    wan_model = None
    pose2d_eval_model = None
    video_extractor = None

    if not args.skip_inference:
        preprocessor = initialize_preprocessor(
            preprocess_ckpt_dir=args.preprocess_ckpt_dir,
            ckpt_dir=args.ckpt_dir,
            replace_flag=args.replace_flag,
            device=args.device
        )
        if preprocessor is None:
            all_cached = True
            missing_cache = []
            for vp in video_paths:
                s = Path(vp).stem
                p_dir = os.path.join(output_prep_dir, s)
                req = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
                if not all(os.path.exists(os.path.join(p_dir, f)) for f in req):
                    all_cached = False
                    missing_cache.append(s)

            if all_cached:
                print("[Notice] Preprocessor is not initialized, but all driving videos already have cached preprocessed data.")
                print("[Notice] Continuing with generation using cached pose/face videos...")
            else:
                print(f"[Fatal Error] Cannot proceed with inference: {len(missing_cache)} video(s) require preprocessing ({', '.join(missing_cache[:3])}...),")
                print("but the preprocessor could not be initialized.")
                print("Please follow the instructions printed above to provide or download the preprocessing checkpoints.")
                sys.exit(1)

        wan_model = initialize_wan_animate(
            ckpt_dir=args.ckpt_dir,
            offload_model=args.offload_model,
            device=args.device
        )

    if not args.skip_eval:
        print("[Init] Initializing FVD Spatio-Temporal Feature Extractor (3D-ResNet)...")
        try:
            device_str = "cuda" if torch.cuda.is_available() and "cuda" in args.device else "cpu"
            video_extractor = VideoFeatureExtractor(device=device_str)
        except Exception as e:
            print(f"[Warning] Could not initialize VideoFeatureExtractor: {e}")
            video_extractor = None

        # ViTPose for generated video keypoints
        if preprocessor is not None and hasattr(preprocessor, "pose2d"):
            pose2d_eval_model = preprocessor.pose2d
        else:
            det_ckpt, pose_ckpt, _, _ = resolve_preprocess_checkpoints(
                preprocess_ckpt_dir=args.preprocess_ckpt_dir,
                ckpt_dir=args.ckpt_dir
            )
            if pose_ckpt and os.path.exists(pose_ckpt):
                try:
                    from pose2d import Pose2d
                    print(f"[Init] Initializing standalone Pose2d for evaluation from {pose_ckpt}...")
                    pose2d_eval_model = Pose2d(
                        checkpoint=pose_ckpt,
                        detector_checkpoint=det_ckpt if det_ckpt and os.path.exists(det_ckpt) else None
                    )
                except Exception as e:
                    print(f"[Warning] Standalone Pose2d failed: {e}")

    # 5. Process Videos Sequentially
    all_evaluated_metrics = []
    dataset_real_clips = []
    dataset_gen_clips = []

    for idx, video_path in enumerate(video_paths, 1):
        stem = Path(video_path).stem
        print(f"\n--- [{idx}/{len(video_paths)}] Processing: {stem} ---")

        prep_sub_dir = os.path.join(output_prep_dir, stem)
        gen_video_path = os.path.join(output_videos_dir, f"{stem}_animate.mp4")
        metric_json_path = os.path.join(output_metrics_dir, f"{stem}_metrics.json")

        # Resume check
        if args.resume and os.path.exists(metric_json_path):
            print(f"[Resume] Metrics already exist for {stem}. Loading from {metric_json_path}...")
            try:
                with open(metric_json_path, "r", encoding="utf-8") as f:
                    all_evaluated_metrics.append(json.load(f))
                continue
            except Exception as e:
                print(f"[Resume] Failed reading existing metrics ({e}), re-evaluating...")

        try:
            # Stage A: Preprocessing
            if not args.skip_inference:
                run_video_preprocessing(
                    preprocessor=preprocessor,
                    video_path=video_path,
                    ref_image_path=ref_image_path,
                    output_prep_dir=prep_sub_dir,
                    resolution_area=args.resolution_area,
                    fps=args.fps,
                    replace_flag=args.replace_flag,
                    retarget_flag=args.retarget_flag
                )

            # Stage B: Video Generation
            if not args.skip_inference:
                run_video_generation(
                    wan_model=wan_model,
                    preprocessed_dir=prep_sub_dir,
                    output_video_path=gen_video_path,
                    clip_len=args.clip_len,
                    refert_num=args.refert_num,
                    sample_shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sample_steps=args.sample_steps,
                    sample_guide_scale=args.sample_guide_scale,
                    base_seed=args.base_seed,
                    offload_model=args.offload_model,
                    replace_flag=args.replace_flag,
                    fps=args.fps
                )

            # Stage C: Evaluation
            if not args.skip_eval:
                if not os.path.exists(gen_video_path):
                    print(f"[Warning] Generated video missing: {gen_video_path}. Skipping evaluation.")
                    continue

                metrics, feats_real, feats_fake = evaluate_generated_video(
                    gt_video_path=video_path,
                    gen_video_path=gen_video_path,
                    pose2d_model=pose2d_eval_model,
                    feature_extractor=video_extractor,
                    conf_thresh=args.conf_thresh
                )

                if feats_real is not None and feats_fake is not None:
                    dataset_real_clips.append(feats_real)
                    dataset_gen_clips.append(feats_fake)

                # Save per-video JSON report
                with open(metric_json_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                print(f"[Metric] Saved per-video report to {metric_json_path}")
                all_evaluated_metrics.append(metrics)

        except Exception as e:
            print(f"[Error] Failed processing video {stem}: {e}")
            traceback.print_exc()
            continue

    # 6. Aggregation and Export
    if not args.skip_eval and all_evaluated_metrics:
        print("\n" + "=" * 80)
        print("  AGGREGATING METRICS ACROSS VIDEOS")
        print("=" * 80)

        # Compute Global Dataset FVD (Gold-standard Computer Vision metric)
        global_dataset_fvd = None
        if dataset_real_clips and dataset_gen_clips:
            try:
                all_real_feats = np.concatenate(dataset_real_clips, axis=0)
                all_fake_feats = np.concatenate(dataset_gen_clips, axis=0)
                global_dataset_fvd = compute_dataset_fvd(all_real_feats, all_fake_feats)
                print(f"[FVD] Computed Global Dataset FVD over {len(all_real_feats)} video clips: {global_dataset_fvd:.2f}")
            except Exception as e:
                print(f"[Warning] Failed computing Global Dataset FVD: {e}")

        mean_summary = compute_aggregate_means(all_evaluated_metrics)

        # Print console table
        table_str = format_summary_table(
            all_evaluated_metrics,
            mean_summary,
            global_dataset_fvd=global_dataset_fvd
        )
        print("\n" + table_str + "\n")

        # Save metrics_summary.json
        summary_json_path = os.path.join(args.output_dir, "metrics_summary.json")
        summary_payload = {
            "timestamp": datetime.now().isoformat(),
            "total_videos_evaluated": len(all_evaluated_metrics),
            "global_dataset_fvd": round(global_dataset_fvd, 4) if global_dataset_fvd is not None else None,
            "global_fvd_quality": assess_fvd_quality(global_dataset_fvd) if global_dataset_fvd is not None else "N/A",
            "mean_metrics": mean_summary,
            "individual_videos": all_evaluated_metrics
        }
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
        print(f"[Export] Saved comprehensive JSON summary to: {summary_json_path}")

        # Save metrics_summary.csv
        summary_csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
        export_csv_summary(
            all_evaluated_metrics,
            mean_summary,
            summary_csv_path,
            global_dataset_fvd=global_dataset_fvd
        )
        print(f"[Export] Saved CSV summary report to: {summary_csv_path}")

    elapsed = time.time() - start_time
    print(f"\n[Done] Pipeline finished in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
