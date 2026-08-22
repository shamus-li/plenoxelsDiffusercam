#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class _HardwareScene:
    def __init__(
        self,
        *,
        model_dir: Path,
        gaussians: Any,
        cameras: list[Any],
        point_cloud: Any,
        radius: float,
        initialize: bool,
    ) -> None:
        self.model_path = str(model_dir)
        self.gaussians = gaussians
        self.cameras_extent = radius
        self.avg_angle = 21.21
        self.group_pruning_scales = {0: 1.3}
        self.pruning_extent_scale = 1.3
        self.multiplexed_gt: dict[int, Any] = {}
        self._train_cameras = {0: cameras}
        self._full_test_cameras = cameras
        if initialize:
            gaussians.create_from_pcd(point_cloud, radius)

    def getTrainCameras(self, scale: float = 1.0) -> dict[int, list[Any]]:
        del scale
        return self._train_cameras

    def getTestCameras(self, scale: float = 1.0) -> list[Any]:
        del scale
        return []

    def getFullTestCameras(self, scale: float = 1.0) -> list[Any]:
        del scale
        return self._full_test_cameras

    def save(self, iteration: int, path: str = "point_cloud.ply") -> None:
        point_cloud = Path(self.model_path) / "point_cloud" / f"iteration_{iteration}" / path
        print(f"Saving point cloud at {point_cloud.parent}")
        self.gaussians.save_ply(str(point_cloud))


def _load_scene(
    root: Path,
    *,
    gaussians: Any,
    training: bool,
) -> _HardwareScene:
    import numpy as np
    from PIL import Image

    from scene.cameras import Camera  # ty: ignore[unresolved-import]
    from scene.colmap_loader import (  # ty: ignore[unresolved-import]
        qvec2rotmat,
        read_extrinsics_text,
        read_intrinsics_text,
    )
    from scene.gaussian_model import BasicPointCloud  # ty: ignore[unresolved-import]
    from utils.general_utils import PILtoTorch  # ty: ignore[unresolved-import]
    from utils.graphics_utils import focal2fov, getWorld2View2  # ty: ignore[unresolved-import]

    scene_dir = root / "dataset" / ("train_scene" if training else "eval_scene")
    sparse = scene_dir / "sparse"
    extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
    intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))
    cameras = []
    camera_centers = []
    for camera_id, key in enumerate(sorted(extrinsics)):
        extrinsic = extrinsics[key]
        intrinsic = intrinsics[extrinsic.camera_id]
        if intrinsic.model != "PINHOLE":
            raise ValueError(f"Expected PINHOLE intrinsics, got {intrinsic.model}.")
        image_path = scene_dir / "images" / Path(extrinsic.name).name
        mask_path = scene_dir / "calibration_masks" / f"{image_path.stem}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing hardware image or mask: {image_path}, {mask_path}")
        image = Image.open(image_path).convert("RGB")
        width = round(image.width / 8)
        height = round(image.height / 8)
        image_tensor = PILtoTorch(image, (width, height))[:3]
        mask_image = Image.fromarray(
            (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8) * 255
        )
        mask = (PILtoTorch(mask_image, (width, height))[:1] > 0.5).float()
        rotation = np.transpose(qvec2rotmat(extrinsic.qvec))
        translation = np.asarray(extrinsic.tvec, dtype=np.float64)
        fx, fy = float(intrinsic.params[0]), float(intrinsic.params[1])
        camera = Camera(
            colmap_id=camera_id,
            R=rotation,
            T=translation,
            FoVx=focal2fov(fx, intrinsic.width),
            FoVy=focal2fov(fy, intrinsic.height),
            image=image_tensor,
            gt_alpha_mask=mask,
            mask=mask,
            image_name=image_path.name,
            uid=camera_id,
            group_id=0,
            data_device="cuda",
        )
        cameras.append(camera)
        camera_centers.append(np.linalg.inv(getWorld2View2(rotation, translation))[:3, 3])
    cameras.sort(key=lambda camera: camera.image_name)
    centers = np.stack(camera_centers)
    radius = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max() * 1.1)
    point_cloud = None
    if training:
        from plyfile import PlyData  # ty: ignore[unresolved-import]

        point_cloud_path = root / "dataset" / "initial_points.ply"
        if not point_cloud_path.is_file():
            raise FileNotFoundError(f"Missing hardware initialization: {point_cloud_path}")
        vertices = PlyData.read(point_cloud_path)["vertex"].data
        names = set(vertices.dtype.names or ())
        xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float32)
        rgb = np.column_stack([vertices[channel] for channel in ("red", "green", "blue")])
        rgb = rgb.astype(np.float32) / 255.0
        if {"nx", "ny", "nz"} <= names:
            normals = np.column_stack([vertices[axis] for axis in ("nx", "ny", "nz")]).astype(
                np.float32
            )
        else:
            normals = np.zeros_like(xyz)
        point_cloud = BasicPointCloud(points=xyz, colors=rgb, normals=normals)
    return _HardwareScene(
        model_dir=root / "run",
        gaussians=gaussians,
        cameras=cameras,
        point_cloud=point_cloud,
        radius=radius,
        initialize=training,
    )


def _dataset(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        sh_degree=3,
        model_path=str((root / "run").resolve()),
        white_background=False,
        use_multiplexing=False,
    )


def _train(root: Path) -> None:
    import torch

    import train_sim_multiviews as trainer  # ty: ignore[unresolved-import]
    from scene import GaussianModel  # ty: ignore[unresolved-import]
    from utils.general_utils import safe_state  # ty: ignore[unresolved-import]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for hardware training.")
    def masked_l1_loss(output: Any, target: Any, camera: Any) -> Any:
        mask = camera.mask.to(output.device)
        return (torch.abs(output - target) * mask).sum() / (
            mask.sum().clamp_min(1.0) * output.shape[0]
        )

    safe_state(False)
    dataset = _dataset(root)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = _load_scene(
        root,
        gaussians=gaussians,
        training=True,
    )
    optimization = SimpleNamespace(
        iterations=8_000,
        position_lr_init=0.000022,
        position_lr_final=0.000002,
        position_lr_delay_mult=0.02,
        position_lr_max_steps=3_000,
        feature_lr=0.0018,
        opacity_lr=0.045,
        scaling_lr=0.0035,
        rotation_lr=0.0007,
        exposure_lr_init=0.01,
        exposure_lr_final=0.001,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        percent_dense=0.015,
        lambda_dssim=0.0,
        densification_interval=50,
        opacity_reset_interval=3_000,
        densify_from_iter=120,
        densify_until_iter=7_000,
        densify_grad_threshold=0.000005,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        random_background=False,
        tv_weight=0.0,
        tv_unseen_weight=0.0,
        optimizer_type="default",
        lambda_read=0.0,
        lambda_shot=0.0,
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    gaussians.training_setup(optimization)
    Path(dataset.model_path).mkdir(parents=True, exist_ok=True)
    trainer.training(
        scene=scene,
        gaussians=gaussians,
        dataset=dataset,
        opt=optimization,
        pipe=pipeline,
        testing_iterations=[],
        saving_iterations=[8_000],
        debug_from=-1,
        resolution=8,
        dls=20,
        size_threshold=150,
        extent_multiplier=1.0,
        max_eval_images=0,
        profile_gpu=False,
        tb_writer=None,
        loss_fn=masked_l1_loss,
    )


def _evaluate(root: Path) -> None:
    import imageio.v3 as iio
    import numpy as np
    import torch
    from lpipsPyTorch.modules.lpips import LPIPS  # ty: ignore[unresolved-import]

    from gaussian_renderer import render  # ty: ignore[unresolved-import]
    from scene import GaussianModel  # ty: ignore[unresolved-import]
    from utils.loss_utils import ssim  # ty: ignore[unresolved-import]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for hardware evaluation.")
    dataset = _dataset(root)
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = _load_scene(
        root,
        gaussians=gaussians,
        training=False,
    )
    point_cloud = root / "run" / "point_cloud" / "iteration_8000" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(f"Missing trained point cloud: {point_cloud}")
    gaussians.load_ply(str(point_cloud))
    training_images = {path.name for path in (root / "dataset/train_scene/images").iterdir()}
    cameras = [
        camera for camera in scene.getFullTestCameras() if camera.image_name not in training_images
    ]

    out_dir = root / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    lpips_metric = LPIPS(net_type="vgg").to("cuda").eval()
    for parameter in lpips_metric.parameters():
        parameter.requires_grad_(False)
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    with torch.no_grad():
        for camera in cameras:
            prediction = render(camera, gaussians, pipeline, background)["render"].clamp(0.0, 1.0)
            ground_truth = camera.original_image.clamp(0.0, 1.0)
            mask = camera.mask.to("cuda")
            masked_prediction = prediction * mask
            denominator = mask.sum().clamp_min(1.0) * prediction.shape[0]
            squared_error = ((masked_prediction - ground_truth) ** 2 * mask).sum()
            metrics = {
                "psnr": float(20 * torch.log10(1.0 / torch.sqrt(squared_error / denominator))),
                "ssim": float(ssim(masked_prediction, ground_truth)),
                "lpips": float(lpips_metric(masked_prediction[None], ground_truth[None]).mean()),
            }
            for key in totals:
                totals[key] += metrics[key]
            image_stem = Path(camera.image_name).stem
            iio.imwrite(
                out_dir / f"{image_stem}.png",
                (prediction.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            )
    metrics = {key: value / len(cameras) for key, value in totals.items()}
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "eval"))
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    root = args.data.expanduser().resolve()
    if args.stage == "train":
        _train(root)
    else:
        _evaluate(root)


if __name__ == "__main__":
    main()
