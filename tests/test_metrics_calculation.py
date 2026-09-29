#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification tests for Metrics Calculation Engine:
Tests PA-MPJPE, N-MPJPE, Raw MPJPE, PCK@0.05, PCK@0.10, FVE, and FVD.
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from run_animate_eval import (
    compute_pa_mpjpe,
    compute_n_mpjpe,
    compute_raw_mpjpe,
    compute_pck,
    compute_fve,
    compute_fvd,
    VideoFeatureExtractor,
    EVAL_SUBSETS
)

class TestMetricsCalculation(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.T = 16
        self.J = 133
        # Generate synthetic realistic human keypoints in 1280x720 frame
        self.gt_joints = np.random.uniform(100.0, 600.0, size=(self.T, self.J, 2)).astype(np.float32)
        self.confs = np.random.uniform(0.5, 1.0, size=(self.T, self.J)).astype(np.float32)
        self.bboxes = np.array([[50.0, 50.0, 650.0, 650.0]] * self.T, dtype=np.float32)

    def test_pa_mpjpe_identity(self):
        """Identical poses should yield 0.0 PA-MPJPE."""
        err, count = compute_pa_mpjpe(self.gt_joints, self.gt_joints, self.confs)
        self.assertAlmostEqual(err, 0.0, places=4)
        self.assertGreater(count, 0)

    def test_pa_mpjpe_rigid_invariance(self):
        """PA-MPJPE should be invariant to rotation, scaling, and translation."""
        theta = np.pi / 6 # 30 degrees rotation
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        scale = 1.35
        translation = np.array([45.0, -80.0])

        transformed_joints = np.zeros_like(self.gt_joints)
        for t in range(self.T):
            transformed_joints[t] = scale * (self.gt_joints[t] @ R) + translation

        err, count = compute_pa_mpjpe(self.gt_joints, transformed_joints, self.confs)
        self.assertAlmostEqual(err, 0.0, places=3, msg="PA-MPJPE must be invariant to similarity transformation")

    def test_raw_mpjpe_with_shift(self):
        """Raw MPJPE should reflect direct translation offset."""
        shift = np.array([10.0, 0.0]) # 10 px shift
        shifted_joints = self.gt_joints + shift
        raw_err = compute_raw_mpjpe(self.gt_joints, shifted_joints, self.confs)
        self.assertAlmostEqual(raw_err, 10.0, places=2)

    def test_pck_perfect_match(self):
        """Identical poses should yield 100% PCK."""
        pck_05, count = compute_pck(self.gt_joints, self.gt_joints, self.confs, self.bboxes, alpha=0.05)
        pck_10, _ = compute_pck(self.gt_joints, self.gt_joints, self.confs, self.bboxes, alpha=0.10)
        self.assertAlmostEqual(pck_05, 100.0, places=2)
        self.assertAlmostEqual(pck_10, 100.0, places=2)

    def test_pck_threshold_behavior(self):
        """Shifting joints beyond threshold should drop PCK score."""
        scale = 600.0 # BBox size is 600 px
        # Shift by 0.08 * scale (48 px) -> Should fail PCK@0.05 (threshold 30px), pass PCK@0.10 (threshold 60px)
        offset = 48.0
        shifted_joints = self.gt_joints.copy()
        shifted_joints[:, :, 0] += offset

        pck_05, _ = compute_pck(self.gt_joints, shifted_joints, self.confs, self.bboxes, alpha=0.05)
        pck_10, _ = compute_pck(self.gt_joints, shifted_joints, self.confs, self.bboxes, alpha=0.10)
        self.assertEqual(pck_05, 0.0)
        self.assertEqual(pck_10, 100.0)

    def test_fve_velocity_error(self):
        """Test First-order Velocity Error computation."""
        # Identical poses -> 0 velocity error
        fve_zero = compute_fve(self.gt_joints, self.gt_joints, self.confs, procrustes_align=True)
        self.assertAlmostEqual(fve_zero, 0.0, places=4)

        # Constant velocity drift: add drift * t
        drift_joints = self.gt_joints.copy()
        for t in range(self.T):
            drift_joints[t] += t * 5.0 # 5 px per frame drift
        fve_val = compute_fve(self.gt_joints, drift_joints, self.confs, procrustes_align=False)
        self.assertAlmostEqual(fve_val, 5.0, places=1)

    def test_fvd_execution(self):
        """Test FVD feature extraction and distance computation."""
        # Synthetic video tensors [T, H, W, C] in range [0, 1]
        video_a = torch.rand(16, 128, 128, 3)
        video_b = torch.rand(16, 128, 128, 3)
        fvd_identical = compute_fvd(video_a, video_a, clip_len=16, stride=8, device="cpu")
        self.assertAlmostEqual(fvd_identical, 0.0, places=1)

        fvd_diff = compute_fvd(video_a, video_b, clip_len=16, stride=8, device="cpu")
        self.assertGreater(fvd_diff, 0.0)

    def test_eval_subsets_coverage(self):
        """Ensure all 133 WholeBody subsets are valid indices."""
        for name, indices in EVAL_SUBSETS.items():
            self.assertGreater(len(indices), 0)
            self.assertTrue(all(0 <= idx < 133 for idx in indices))

if __name__ == "__main__":
    unittest.main()
