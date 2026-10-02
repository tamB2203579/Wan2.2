# -*- coding: utf-8 -*-
"""
Wan2.2 Animation Evaluation Runner (Modal Cloud Execution)

Utilizes Modal (H100 GPU) to run the full Wan2.2-Animate evaluation pipeline:
- Kaggle sign language dataset downloading (or local sample evaluation)
- Video preprocessing (ViTPose whole-body & YOLOv10 detection)
- Wan2.2 Animate inference (14B model on H100 GPU)
- Metrics calculation (PA-MPJPE, N-MPJPE, PCK@0.05, PCK@0.10, FVD)
- Automatic artifact sync back to local `./outputs/`

Usage:
    # Run single video evaluation on Modal H100
    modal run run_evaluation.py --video-name D0001B.mp4

    # Run batch evaluation on 25 random samples from Kaggle
    modal run run_evaluation.py --mode kaggle_batch --num-samples 25

    # Run without inference (evaluation only)
    modal run run_evaluation.py --skip-inference

    # Run with Python CLI directly (convenience wrapper that invokes modal)
    python run_evaluation.py --mode kaggle_sample
"""

import os
import sys
import csv
import io
import random
import urllib.request
import urllib.parse
import subprocess
from typing import List, Dict, Any, Optional
import modal

app = modal.App("wan22-animate-eval")
volume = modal.Volume.from_name("wan22-models-data", create_if_missing=True)

DATA_DIR = "/data"
CKPT_DIR = f"{DATA_DIR}/Wan2.2-Animate-14B"
PREPROCESS_CKPT_DIR = f"{CKPT_DIR}/process_checkpoint"
OUTPUT_DIR = f"{DATA_DIR}/outputs"
KAGGLE_DATASET = "aresusayhi/vsl-vietnamese-sign-languages"

cuda_tag = "12.4.1-cudnn-devel-ubuntu22.04"

wan_h100_image = (
    modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("ninja", "psutil", "packaging", "wheel")
    .run_commands(
        "pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl || echo 'FlashAttention wheel skipped'"
    )
    .pip_install(
        "diffusers>=0.31.0",
        "transformers>=4.49.0,<=4.51.3",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "easydict",
        "imageio",
        "imageio-ffmpeg",
        "opencv-python-headless",
        "onnxruntime-gpu==1.20.2",
        "scipy",
        "numpy<2",
        "tqdm",
        "huggingface-hub",
        "moviepy",
        "decord",
        "einops",
        "peft",
        "loguru",
        "ftfy",
        "matplotlib",
        "pandas",
        "sentencepiece"
    )
    .add_local_dir(
        local_path=".",
        remote_path="/app",
        ignore=[".git", ".kilo", ".codegraph*", "results_*", "__pycache__", "*.pyc", "test_*", "outputs"]
    )
)


def fetch_kaggle_catalog() -> List[Dict[str, str]]:
    """Fetch video manifest from the VSL Kaggle dataset."""
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}/Dataset%2FLabels%2Flabel.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return [{"video": r["VIDEO"].strip(), "label": r.get("LABEL", "").strip()} for r in reader if r.get("VIDEO", "").endswith(".mp4")]


def download_kaggle_video(video_filename: str, dest_dir: str) -> str:
    """Download individual video from Kaggle dataset with caching."""
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, video_filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        return out_path

    encoded = urllib.parse.quote(f"Dataset/Videos/{video_filename}", safe="")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    return out_path


@app.function(
    image=wan_h100_image,
    gpu="H100",
    volumes={DATA_DIR: volume},
    timeout=14400
)
def run_eval_remote(
    mode: str = "kaggle_sample",
    video_list: Optional[List[Dict[str, str]]] = None,
    single_video_name: str = "D0001B.mp4",
    sample_steps: int = 20,
    skip_inference: bool = False,
    skip_eval: bool = False,
    max_videos: Optional[int] = None
) -> Dict[str, Any]:
    """Execute evaluation pipeline on remote H100 instance."""
    import os
    import subprocess
    import json
    import torch

    print(f"[Remote] Hardware: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)")

    # 1. Prepare input videos based on mode
    if mode == "local":
        input_dir = "/app/Original_9x16"
    else:
        input_dir = "/tmp/modal_eval_inputs"
        os.makedirs(input_dir, exist_ok=True)
        targets = [single_video_name] if mode == "kaggle_sample" else [v["video"] for v in (video_list or [])]
        for idx, v_name in enumerate(targets, 1):
            p = download_kaggle_video(v_name, input_dir)
            print(f"[{idx}/{len(targets)}] Cached: {v_name} ({os.path.getsize(p)/1024:.1f} KB)")

    # 2. Ensure model checkpoints exist in persistent volume
    det_onnx = f"{PREPROCESS_CKPT_DIR}/det/yolov10m.onnx"
    vitpose_onnx = f"{PREPROCESS_CKPT_DIR}/pose2d/vitpose_h_wholebody.onnx"
    vae_file = f"{CKPT_DIR}/Wan2.1_VAE.pth"
    if not (os.path.exists(det_onnx) and os.path.exists(vitpose_onnx) and os.path.exists(vae_file)):
        from huggingface_hub import snapshot_download
        print(f"[Remote] Downloading Wan2.2-Animate-14B models to {CKPT_DIR}...")
        os.makedirs(CKPT_DIR, exist_ok=True)
        snapshot_download(repo_id="Wan-AI/Wan2.2-Animate-14B", local_dir=CKPT_DIR)
        volume.commit()

    # 3. Run evaluation pipeline via modular eval.pipeline
    cmd = [
        "python", "-m", "eval.pipeline",
        "--input_dir", input_dir,
        "--ref_image", "/app/Model/avatar_nu.png",
        "--ckpt_dir", CKPT_DIR,
        "--preprocess_ckpt_dir", PREPROCESS_CKPT_DIR,
        "--output_dir", OUTPUT_DIR,
        "--device", "cuda:0",
        "--sample_steps", str(sample_steps),
        "--no_offload_model",
        "--resume"
    ]
    if skip_inference:
        cmd.append("--skip_inference")
    if skip_eval:
        cmd.append("--skip_eval")
    if max_videos:
        cmd.extend(["--max_videos", str(max_videos)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, cwd="/app", env=env)
    volume.commit()

    summary_file = f"{OUTPUT_DIR}/metrics_summary.json"
    summary_data = {}
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    return {"status": "success", "summary": summary_data, "output_dir": OUTPUT_DIR}


@app.local_entrypoint()
def main(
    mode: str = "kaggle_sample",
    num_samples: int = 25,
    seed: int = 42,
    video_name: str = "D0001B.mp4",
    sample_steps: int = 20,
    skip_inference: bool = False,
    skip_eval: bool = False,
    max_videos: Optional[int] = None,
    local_output_dir: str = "./outputs",
    download_results: bool = True
):
    """Local entrypoint for modal run run_evaluation.py."""
    print(f"=== Modal H100 Evaluation Runner (mode: {mode}) ===")
    video_list = None
    if mode == "kaggle_batch":
        catalog = fetch_kaggle_catalog()
        rng = random.Random(seed)
        video_list = rng.sample(catalog, min(num_samples, len(catalog)))
        print(f"Sampled {len(video_list)} videos with seed {seed}: {[v['video'] for v in video_list[:5]]}...")

    res = run_eval_remote.remote(
        mode=mode,
        video_list=video_list,
        single_video_name=video_name,
        sample_steps=sample_steps,
        skip_inference=skip_inference,
        skip_eval=skip_eval,
        max_videos=max_videos
    )

    if download_results:
        try:
            subprocess.run(["modal", "volume", "get", "--force", "wan22-models-data", "outputs", "."], check=True)
            print(f"[Downloaded] Results updated in {local_output_dir}")
        except Exception:
            try:
                os.makedirs(local_output_dir, exist_ok=True)
                subprocess.run(["modal", "volume", "get", "--force", "wan22-models-data", "outputs", local_output_dir], check=True)
                print(f"[Downloaded] Results updated in {local_output_dir}")
            except Exception as e:
                print(f"[Notice] Failed to auto-download volume: {e}")
                print(f"You can manually download outputs anytime with:")
                print(f"  modal volume get --force wan22-models-data outputs .")

    summary = res.get("summary", {})
    stats = summary.get("aggregate_statistics", {})
    if stats:
        mean_s = stats.get("mean", {})
        std_s = stats.get("std", {})
        var_s = stats.get("variance", {})
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Evaluated: {summary.get('total_videos')} videos | Global FVD: {summary.get('global_dataset_fvd')}")
        print(f"{'Metric':<16} {'Subset':<10} {'Mean':>10} {'Std Dev (±)':>14} {'Variance (s²)':>16}")
        print("-" * 70)
        for label, key, dec in [("PA-MPJPE", "pa_mpjpe", 2), ("PCK@0.05", "pck_005", 2), ("PCK@0.10", "pck_010", 2), ("N-MPJPE", "n_mpjpe", 4)]:
            for i, sub in enumerate(["overall", "body", "face", "hands"]):
                m_v = mean_s.get(key, {}).get(sub, 0.0)
                s_v = std_s.get(key, {}).get(sub, 0.0)
                v_v = var_s.get(key, {}).get(sub, 0.0)
                lead = label if i == 0 else ""
                fmt = f"{m_v:>10.4f} {f'±{s_v:.4f}':>14} {v_v:>16.6f}" if dec == 4 else f"{m_v:>10.2f} {f'±{s_v:.2f}':>14} {v_v:>16.2f}"
                print(f"{lead:<16} {sub.capitalize():<10} {fmt}")
            print("-" * 70)


if __name__ == "__main__":
    print("[INFO] Launching evaluation via Modal CLI...")
    cmd = [sys.executable, "-m", "modal", "run", __file__] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
