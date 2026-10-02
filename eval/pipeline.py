import os
import sys
import glob
import json
import csv
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
PREPROCESS_DIR = os.path.join(ROOT_DIR, "wan", "modules", "animate", "preprocess")
if PREPROCESS_DIR not in sys.path:
    sys.path.insert(0, PREPROCESS_DIR)

from eval.metrics import evaluate_pair, compute_aggregate_stats, compute_dataset_fvd, VideoFeatureExtractor
from eval.video_io import read_image, load_frames, frames_to_tensor, extract_joints


def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 Animate Evaluation Pipeline")
    parser.add_argument("--input_dir", type=str, default=os.path.join(ROOT_DIR, "Original_9x16"))
    parser.add_argument("--ref_image", type=str, default=os.path.join(ROOT_DIR, "Model", "avatar_nu.png"))
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--ckpt_dir", type=str, default="./Wan2.2-Animate-14B")
    parser.add_argument("--preprocess_ckpt_dir", type=str, default="./process_checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--clip_len", type=int, default=77)
    parser.add_argument("--refert_num", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution_area", type=int, nargs=2, default=[1280, 720])
    parser.add_argument("--sample_steps", type=int, default=20)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=1.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--offload_model", action="store_true", default=True)
    parser.add_argument("--no_offload_model", action="store_false", dest="offload_model")
    parser.add_argument("--replace_flag", action="store_true", default=False)
    parser.add_argument("--retarget_flag", action="store_true", default=False)

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--skip_inference", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    return parser.parse_args()


def resolve_checkpoints(preprocess_dir: str, ckpt_dir: str) -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        preprocess_dir,
        os.path.join(preprocess_dir, "process_checkpoint") if preprocess_dir else "",
        os.path.join(ckpt_dir, "process_checkpoint") if ckpt_dir else "",
        ckpt_dir,
        os.path.join(ROOT_DIR, "process_checkpoint"),
        ROOT_DIR,
    ]
    valid_dirs = [os.path.abspath(d) for d in candidates if d and os.path.isdir(d)]

    det_ckpt, pose_ckpt = None, None
    for d in valid_dirs:
        for p in [os.path.join(d, "det", "yolov10m.onnx"), os.path.join(d, "yolov10m.onnx")]:
            if os.path.isfile(p):
                det_ckpt = p
                break
        if det_ckpt:
            break

    for d in valid_dirs:
        for p in [os.path.join(d, "pose2d", "vitpose_h_wholebody.onnx"), os.path.join(d, "vitpose_h_wholebody.onnx")]:
            if os.path.exists(p):
                pose_ckpt = p
                break
        if pose_ckpt:
            break

    return det_ckpt, pose_ckpt


def init_preprocessor(preprocess_dir: str, ckpt_dir: str, device: str = "cuda:0"):
    det_ckpt, pose_ckpt = resolve_checkpoints(preprocess_dir, ckpt_dir)
    if not det_ckpt or not pose_ckpt:
        print(f"[Error] ViTPose/YOLO checkpoints missing in {preprocess_dir} or {ckpt_dir}.")
        return None

    try:
        from wan.modules.animate.preprocess.process_pipepline import ProcessPipeline
    except ImportError:
        from process_pipepline import ProcessPipeline  # type: ignore

    try:
        pipeline = ProcessPipeline(
            det_checkpoint_path=det_ckpt,
            pose2d_checkpoint_path=pose_ckpt,
            sam_checkpoint_path=None,
            flux_kontext_path=None
        )
        if hasattr(pipeline, "pose2d"):
            try:
                pipeline.pose2d.set_device(device)
            except Exception:
                pass
        return pipeline
    except Exception as e:
        print(f"[Warning] Failed initializing preprocessor on {device}: {e}, trying CPU fallback...")
        try:
            pipeline = ProcessPipeline(
                det_checkpoint_path=det_ckpt,
                pose2d_checkpoint_path=pose_ckpt,
                sam_checkpoint_path=None,
                flux_kontext_path=None
            )
            if hasattr(pipeline, "pose2d"):
                pipeline.pose2d.set_device("cpu")
            return pipeline
        except Exception as e2:
            print(f"[Error] Fallback to CPU also failed: {e2}")
            return None


def init_wan(ckpt_dir: str, offload_model: bool = True, device: str = "cuda:0"):
    candidates = [ckpt_dir, os.path.join(ROOT_DIR, "Wan2.2-Animate-14B"), "./Wan2.2-Animate-14B"]
    resolved = next((os.path.abspath(c) for c in candidates if c and os.path.isdir(c)), None)
    if not resolved:
        print(f"[Error] WanAnimate checkpoint directory not found: {ckpt_dir}")
        return None

    from easydict import EasyDict
    from wan.configs.shared_config import wan_shared_cfg
    cfg = EasyDict(__name__="Config: Wan animate 14B")
    cfg.update(wan_shared_cfg)
    cfg.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
    cfg.t5_tokenizer = "google/umt5-xxl"
    cfg.clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    cfg.clip_tokenizer = "xlm-roberta-large"
    cfg.lora_checkpoint = "relighting_lora.ckpt"
    cfg.vae_checkpoint = "Wan2.1_VAE.pth"
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
    cfg.prompt = "视频中的人在做动作"

    from wan.animate import WanAnimate
    device_id = int(device.split(":")[-1]) if ":" in device else 0
    print(f"[Init] Initializing WanAnimate 14B from {resolved} (device: {device_id})...")
    return WanAnimate(
        config=cfg,
        checkpoint_dir=resolved,
        device_id=device_id,
        t5_cpu=offload_model
    )


def run_preprocess(preprocessor, video_path: str, ref_image_path: str, out_dir: str, resolution_area: List[int], fps: int) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    required = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
    if all(os.path.exists(os.path.join(out_dir, f)) for f in required):
        return True
    if preprocessor is None:
        raise RuntimeError("Preprocessor not initialized.")
    return bool(preprocessor(
        video_path=video_path,
        refer_image_path=ref_image_path,
        output_path=out_dir,
        resolution_area=resolution_area,
        fps=fps
    ))


def run_generation(wan_model, prep_dir: str, out_video: str, args) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
    if os.path.exists(out_video) and os.path.getsize(out_video) > 1000:
        return True
    if wan_model is None:
        raise RuntimeError("WanAnimate model not initialized.")

    tensor = wan_model.generate(
        src_root_path=prep_dir,
        replace_flag=args.replace_flag,
        refert_num=args.refert_num,
        clip_len=args.clip_len,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model
    )
    if tensor is None:
        return False

    from wan.utils.utils import save_video
    save_video(tensor=tensor[None], save_file=out_video, fps=args.fps, nrow=1, normalize=True, value_range=(-1, 1))
    return True


def run_eval(gt_video: str, gen_video: str, pose_model, extractor, conf_thresh: float = 0.3):
    gt_frames, fps = load_frames(gt_video)
    gen_frames, _ = load_frames(gen_video)
    min_len = min(len(gt_frames), len(gen_frames))

    gt_frames = gt_frames[:min_len]
    gen_frames = gen_frames[:min_len]

    gt_metas = pose_model(gt_frames)
    gen_metas = pose_model(gen_frames)

    gt_joints, gt_confs, gt_bboxes = extract_joints(gt_metas)
    gen_joints, gen_confs, _ = extract_joints(gen_metas)

    metrics, feats_real, feats_fake = evaluate_pair(
        gt_joints=gt_joints[:min_len],
        pred_joints=gen_joints[:min_len],
        gt_confidences=gt_confs[:min_len],
        gt_bboxes=gt_bboxes[:min_len],
        real_video_tensor=frames_to_tensor(gt_frames),
        gen_video_tensor=frames_to_tensor(gen_frames),
        feature_extractor=extractor,
        conf_thresh=conf_thresh
    )
    metrics.update({
        "video_stem": Path(gt_video).stem,
        "gt_video": os.path.basename(gt_video),
        "gen_video": os.path.basename(gen_video),
        "fps": float(fps)
    })
    return metrics, feats_real, feats_fake


def format_table(metrics_list: List[Dict[str, Any]], stats: Dict[str, Any], global_fvd: Optional[float] = None) -> str:
    name_w = 18
    hdr = f"{'Video Name':<{name_w}} {'Frames':>6} {'PA-MPJPE':>9} {'PCK@0.05':>9} {'PCK@0.10':>9} {'N-MPJPE':>9} {'FVD':>9} {'Quality':<10}"
    sep = "=" * len(hdr)
    lines = [sep, hdr, "-" * len(hdr)]

    for m in metrics_list:
        fvd_val = m.get("fvd")
        fvd_s = f"{fvd_val:>9.2f}" if fvd_val is not None else f"{'N/A':>9}"
        lines.append(
            f"{m.get('video_stem', '')[:name_w]:<{name_w}} "
            f"{m.get('frames_evaluated', 0):>6d} "
            f"{m.get('pa_mpjpe', {}).get('overall', 0):>9.2f} "
            f"{m.get('pck_005', {}).get('overall', 0):>8.2f}% "
            f"{m.get('pck_010', {}).get('overall', 0):>8.2f}% "
            f"{m.get('n_mpjpe', {}).get('overall', 0):>9.4f} "
            f"{fvd_s} "
            f"{m.get('fvd_quality', 'N/A'):<10}"
        )

    lines.append(sep)
    mean_s = stats.get("mean", {})
    std_s = stats.get("std", {})
    var_s = stats.get("variance", {})

    if mean_s:
        fvd_m = mean_s.get("fvd")
        fvd_str = f"{fvd_m:>9.2f}" if fvd_m is not None else f"{'N/A':>9}"
        lines.append(
            f"{'MEAN (video)':<{name_w}} {'':>6} "
            f"{mean_s.get('pa_mpjpe', {}).get('overall', 0):>9.2f} "
            f"{mean_s.get('pck_005', {}).get('overall', 0):>8.2f}% "
            f"{mean_s.get('pck_010', {}).get('overall', 0):>8.2f}% "
            f"{mean_s.get('n_mpjpe', {}).get('overall', 0):>9.4f} "
            f"{fvd_str} {'':<10}"
        )

    if std_s:
        fvd_std = std_s.get("fvd")
        fvd_std_s = f"±{fvd_std:>8.2f}" if fvd_std is not None else f"{'':>9}"
        lines.append(
            f"{'STD DEV (±)':<{name_w}} {'':>6} "
            f"±{std_s.get('pa_mpjpe', {}).get('overall', 0):>8.2f} "
            f"±{std_s.get('pck_005', {}).get('overall', 0):>7.2f}% "
            f"±{std_s.get('pck_010', {}).get('overall', 0):>7.2f}% "
            f"±{std_s.get('n_mpjpe', {}).get('overall', 0):>8.4f} "
            f"{fvd_std_s} {'':<10}"
        )

    if var_s:
        fvd_var = var_s.get("fvd")
        fvd_var_s = f"{fvd_var:>9.2f}" if fvd_var is not None else f"{'':>9}"
        lines.append(
            f"{'VARIANCE (s²)':<{name_w}} {'':>6} "
            f"{var_s.get('pa_mpjpe', {}).get('overall', 0):>9.2f} "
            f"{var_s.get('pck_005', {}).get('overall', 0):>9.2f} "
            f"{var_s.get('pck_010', {}).get('overall', 0):>9.2f} "
            f"{var_s.get('n_mpjpe', {}).get('overall', 0):>9.6f} "
            f"{fvd_var_s} {'':<10}"
        )

    if global_fvd is not None:
        lines.append(f"{'GLOBAL DATASET FVD':<{name_w}} {'':>6} {'':>9} {'':>9} {'':>9} {'':>9} {global_fvd:>9.2f} {'':<10}")

    lines.append(sep)
    return "\n".join(lines)


def export_csv(metrics_list: List[Dict[str, Any]], stats: Dict[str, Any], path: str, global_fvd: Optional[float] = None):
    cols = [
        "Statistic / Video", "Frames",
        "PA-MPJPE (Overall)", "PA-MPJPE (Body)", "PA-MPJPE (Face)", "PA-MPJPE (Hands)",
        "PCK@0.05 (Overall)", "PCK@0.05 (Body)", "PCK@0.05 (Face)", "PCK@0.05 (Hands)",
        "PCK@0.10 (Overall)", "PCK@0.10 (Body)", "PCK@0.10 (Face)", "PCK@0.10 (Hands)",
        "N-MPJPE (Overall)", "N-MPJPE (Body)", "N-MPJPE (Face)", "N-MPJPE (Hands)",
        "FVD", "FVD Assessment"
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in metrics_list:
            w.writerow([
                m.get("video_stem", ""), m.get("frames_evaluated", 0),
                m.get("pa_mpjpe", {}).get("overall", ""), m.get("pa_mpjpe", {}).get("body", ""),
                m.get("pa_mpjpe", {}).get("face", ""), m.get("pa_mpjpe", {}).get("hands", ""),
                m.get("pck_005", {}).get("overall", ""), m.get("pck_005", {}).get("body", ""),
                m.get("pck_005", {}).get("face", ""), m.get("pck_005", {}).get("hands", ""),
                m.get("pck_010", {}).get("overall", ""), m.get("pck_010", {}).get("body", ""),
                m.get("pck_010", {}).get("face", ""), m.get("pck_010", {}).get("hands", ""),
                m.get("n_mpjpe", {}).get("overall", ""), m.get("n_mpjpe", {}).get("body", ""),
                m.get("n_mpjpe", {}).get("face", ""), m.get("n_mpjpe", {}).get("hands", ""),
                m.get("fvd", ""), m.get("fvd_quality", "")
            ])

        for label, tag in [("MEAN", "mean"), ("STD DEV", "std"), ("VARIANCE", "variance")]:
            s_dict = stats.get(tag, {})
            if s_dict:
                w.writerow([
                    label, "",
                    s_dict.get("pa_mpjpe", {}).get("overall", ""), s_dict.get("pa_mpjpe", {}).get("body", ""),
                    s_dict.get("pa_mpjpe", {}).get("face", ""), s_dict.get("pa_mpjpe", {}).get("hands", ""),
                    s_dict.get("pck_005", {}).get("overall", ""), s_dict.get("pck_005", {}).get("body", ""),
                    s_dict.get("pck_005", {}).get("face", ""), s_dict.get("pck_005", {}).get("hands", ""),
                    s_dict.get("pck_010", {}).get("overall", ""), s_dict.get("pck_010", {}).get("body", ""),
                    s_dict.get("pck_010", {}).get("face", ""), s_dict.get("pck_010", {}).get("hands", ""),
                    s_dict.get("n_mpjpe", {}).get("overall", ""), s_dict.get("n_mpjpe", {}).get("body", ""),
                    s_dict.get("n_mpjpe", {}).get("face", ""), s_dict.get("n_mpjpe", {}).get("hands", ""),
                    s_dict.get("fvd", ""), ""
                ])

        if global_fvd is not None:
            w.writerow(["GLOBAL DATASET FVD", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", round(global_fvd, 4), ""])


def main():
    args = parse_args()
    start_time = time.time()

    if not os.path.exists(args.ref_image):
        print(f"[Error] Reference avatar not found: {args.ref_image}")
        sys.exit(1)

    video_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")) + glob.glob(os.path.join(args.input_dir, "*.mov")))
    if not video_paths:
        print(f"[Error] No videos found in {args.input_dir}")
        sys.exit(1)

    if args.max_videos:
        video_paths = video_paths[:args.max_videos]

    out_vids = os.path.join(args.output_dir, "videos")
    out_metrics = os.path.join(args.output_dir, "metrics")
    out_prep = os.path.join(args.output_dir, "preprocessed")
    for d in [out_vids, out_metrics, out_prep]:
        os.makedirs(d, exist_ok=True)

    preprocessor = None
    wan_model = None
    if not args.skip_inference:
        preprocessor = init_preprocessor(args.preprocess_ckpt_dir, args.ckpt_dir, args.device)
        wan_model = init_wan(args.ckpt_dir, args.offload_model, args.device)

    pose_model, extractor = None, None
    if not args.skip_eval:
        extractor = VideoFeatureExtractor(device="cuda" if "cuda" in args.device and torch.cuda.is_available() else "cpu")
        if preprocessor and hasattr(preprocessor, "pose2d"):
            pose_model = preprocessor.pose2d
        else:
            det_ckpt, pose_ckpt = resolve_checkpoints(args.preprocess_ckpt_dir, args.ckpt_dir)
            if pose_ckpt:
                try:
                    try:
                        from wan.modules.animate.preprocess.pose2d import Pose2d
                    except ImportError:
                        from pose2d import Pose2d  # type: ignore
                    pose_model = Pose2d(checkpoint=pose_ckpt, detector_checkpoint=det_ckpt)
                except Exception:
                    pass

    all_metrics = []
    real_clips, gen_clips = [], []

    for idx, v_path in enumerate(video_paths, 1):
        stem = Path(v_path).stem
        print(f"[{idx}/{len(video_paths)}] Processing: {stem}")
        prep_dir = os.path.join(out_prep, stem)
        gen_path = os.path.join(out_vids, f"{stem}_animate.mp4")
        metric_path = os.path.join(out_metrics, f"{stem}_metrics.json")

        if args.resume and os.path.exists(metric_path):
            with open(metric_path, "r", encoding="utf-8") as f:
                all_metrics.append(json.load(f))
            continue

        try:
            if not args.skip_inference:
                run_preprocess(preprocessor, v_path, args.ref_image, prep_dir, args.resolution_area, args.fps)
                run_generation(wan_model, prep_dir, gen_path, args)

            if not args.skip_eval and os.path.exists(gen_path):
                m, f_r, f_g = run_eval(v_path, gen_path, pose_model, extractor, args.conf_thresh)
                if f_r is not None and f_g is not None:
                    real_clips.append(f_r)
                    gen_clips.append(f_g)
                with open(metric_path, "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2, ensure_ascii=False)
                all_metrics.append(m)
        except Exception as e:
            print(f"[Error] {stem}: {e}")
            traceback.print_exc()

    if not args.skip_eval and all_metrics:
        global_fvd = None
        if real_clips and gen_clips:
            try:
                global_fvd = compute_dataset_fvd(np.concatenate(real_clips, axis=0), np.concatenate(gen_clips, axis=0))
            except Exception:
                pass

        stats = compute_aggregate_stats(all_metrics)
        print("\n" + format_table(all_metrics, stats, global_fvd) + "\n")

        with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_videos": len(all_metrics),
                "global_dataset_fvd": round(global_fvd, 4) if global_fvd is not None else None,
                "aggregate_statistics": stats,
                "individual_videos": all_metrics
            }, f, indent=2, ensure_ascii=False)

        export_csv(all_metrics, stats, os.path.join(args.output_dir, "metrics_summary.csv"), global_fvd)

    print(f"[Done] Finished in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
