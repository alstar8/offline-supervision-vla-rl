"""360 photo spheres for OpenReal2Sim backgrounds.

Insta360 X3 ``.insp`` files in ``assets/360_photos`` are dual-fisheye JPEGs.
They are converted to equirectangular textures and mapped onto an inverted
UV sphere so cameras inside the workspace see the room instead of a black
clear color.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import sapien
import sapien.render
from PIL import Image

from .scene_loader import SIM2REAL_REPO_ROOT

DEFAULT_360_PHOTOS_DIR = SIM2REAL_REPO_ROOT / "assets" / "360_photos"
DEFAULT_PANO_SPHERE_RADIUS = 20.0
DEFAULT_EQUIRECT_WIDTH = 2048
DEFAULT_DFISHEYE_FOV_DEG = 190.0
PANO_PHOTO_EXTENSIONS = (".insp", ".jpg", ".jpeg", ".png", ".webp")


def list_360_photo_paths(photos_dir: str | Path | None = None) -> list[Path]:
    """Return source 360 captures, skipping generated cache files."""
    root = Path(photos_dir) if photos_dir is not None else DEFAULT_360_PHOTOS_DIR
    if not root.is_dir():
        return []
    photos: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("_") or path.name.startswith("."):
            continue
        if path.suffix.lower() not in PANO_PHOTO_EXTENSIONS:
            continue
        photos.append(path)
    return photos


def inverted_uv_sphere_mesh(
    radius: float,
    n_lat: int = 48,
    n_lon: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit-style UV sphere with inward normals and equirectangular UVs.

    ``v=0`` is the +Z pole (sky in Z-up equirect). ``u`` is flipped so the
    photo is not mirrored when viewed from inside.
    """
    if n_lat < 3 or n_lon < 3:
        raise ValueError("n_lat and n_lon must be >= 3")
    verts = []
    normals = []
    uvs = []
    for i in range(n_lat + 1):
        v = i / n_lat
        phi = v * np.pi
        sin_phi = float(np.sin(phi))
        cos_phi = float(np.cos(phi))
        for j in range(n_lon + 1):
            u = j / n_lon
            theta = u * 2.0 * np.pi
            x = radius * sin_phi * float(np.cos(theta))
            y = radius * sin_phi * float(np.sin(theta))
            z = radius * cos_phi
            verts.append((x, y, z))
            inv = 1.0 / max(radius, 1e-8)
            normals.append((-x * inv, -y * inv, -z * inv))
            uvs.append((1.0 - u, v))
    triangles = []
    stride = n_lon + 1
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * stride + j
            b = a + stride
            triangles.append((a, b, a + 1))
            triangles.append((a + 1, b, b + 1))
    inward = np.asarray(triangles, dtype=np.uint32)
    # Both windings so inside faces survive GPU backface culling.
    triangles_np = np.concatenate([inward, inward[:, ::-1]], axis=0)
    return (
        np.asarray(verts, dtype=np.float32),
        triangles_np,
        np.asarray(normals, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
    )


def dual_fisheye_to_equirect(
    image: np.ndarray,
    fov_deg: float = DEFAULT_DFISHEYE_FOV_DEG,
    width: int = DEFAULT_EQUIRECT_WIDTH,
) -> np.ndarray:
    """Equidistant dual-fisheye (left=front, right=back) to equirectangular.

    Used when ffmpeg's ``v360`` filter is unavailable. Insta360 back cameras
    are rotated 180 degrees in the image plane.
    """
    src = np.asarray(image)
    if src.ndim != 3 or src.shape[2] < 3:
        raise ValueError(f"expected HxWx3 image, got {src.shape}")
    in_h, in_w = src.shape[:2]
    out_w = int(width)
    out_h = out_w // 2
    fov = np.deg2rad(float(fov_deg))
    half_fov = 0.5 * fov
    u = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    v = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    uu, vv = np.meshgrid(u, v)
    lon = (uu - 0.5) * (2.0 * np.pi)
    lat = (0.5 - vv) * np.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    front = z >= 0.0
    rho_xy = np.hypot(x, y)
    theta_f = np.arccos(np.clip(z, -1.0, 1.0))
    theta_b = np.arccos(np.clip(-z, -1.0, 1.0))
    scale_f = (theta_f / half_fov) / np.maximum(rho_xy, 1e-8)
    scale_b = (theta_b / half_fov) / np.maximum(rho_xy, 1e-8)
    fx = x * scale_f
    fy = y * scale_f
    bx = -x * scale_b
    by = -y * scale_b
    radius_px = 0.5 * float(in_h)
    cx_l = 0.25 * float(in_w)
    cx_r = 0.75 * float(in_w)
    cy = 0.5 * float(in_h)
    map_x = np.where(front, cx_l + fx * radius_px, cx_r + bx * radius_px).astype(np.float32)
    map_y = np.where(front, cy - fy * radius_px, cy - by * radius_px).astype(np.float32)
    return cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _run_ffmpeg_dfisheye_to_equirect(
    src_jpeg: Path,
    dst_jpeg: Path,
    width: int,
    fov_deg: float,
) -> None:
    ffmpeg = _ffmpeg_bin()
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg")
    vf = (
        f"v360=dfisheye:equirect:ih_fov={fov_deg:g}:iv_fov={fov_deg:g}"
        f":w={int(width)}:h={int(width) // 2}:interp=linear"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_jpeg),
            "-vf",
            vf,
            "-q:v",
            "3",
            str(dst_jpeg),
        ],
        check=True,
    )


def resolve_equirect_texture(
    photo_path: str | Path,
    *,
    width: int = DEFAULT_EQUIRECT_WIDTH,
    fov_deg: float = DEFAULT_DFISHEYE_FOV_DEG,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a cached equirectangular JPEG for ``photo_path``."""
    src = Path(photo_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    cache_root = Path(cache_dir) if cache_dir is not None else src.parent / "_equirect_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / f"{src.stem}_w{int(width)}_fov{int(fov_deg)}.jpg"
    if cached.is_file() and cached.stat().st_mtime >= src.stat().st_mtime:
        return cached

    with Image.open(src) as image:
        rgb = np.asarray(image.convert("RGB"))

    is_insp = src.suffix.lower() == ".insp"
    looks_dual = is_insp or _looks_like_dual_fisheye(rgb)
    if looks_dual:
        converted = None
        ffmpeg = _ffmpeg_bin()
        if ffmpeg is not None:
            with tempfile.TemporaryDirectory(prefix="pano_dfisheye_") as tmp:
                tmp_src = Path(tmp) / "src.jpg"
                Image.fromarray(rgb).save(tmp_src, quality=95)
                tmp_dst = Path(tmp) / "dst.jpg"
                try:
                    _run_ffmpeg_dfisheye_to_equirect(tmp_src, tmp_dst, width, fov_deg)
                    converted = np.asarray(Image.open(tmp_dst).convert("RGB"))
                except (subprocess.CalledProcessError, FileNotFoundError):
                    converted = None
        if converted is None:
            converted = dual_fisheye_to_equirect(rgb, fov_deg=fov_deg, width=width)
        rgb = converted
    elif rgb.shape[1] != width:
        h = max(1, int(round(width * rgb.shape[0] / rgb.shape[1])))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        rgb = np.asarray(Image.fromarray(rgb).resize((int(width), h), resample))

    Image.fromarray(rgb).save(cached, quality=90)
    return cached


def _looks_like_dual_fisheye(image: np.ndarray) -> bool:
    """Heuristic: 2:1 frame with a dark column between two circular views."""
    if image.ndim != 3:
        return False
    h, w = image.shape[:2]
    if w < 2 * h * 0.8 or w > 2 * h * 1.2:
        return False
    mid = image[:, w // 2 - 2 : w // 2 + 2].astype(np.float32).mean(axis=2)
    left_q = image[:, w // 4 - 2 : w // 4 + 2].astype(np.float32).mean(axis=2)
    return bool(mid.mean() + 25.0 < left_q.mean() and (mid < 16.0).mean() > 0.25)


def make_unlit_pano_material(texture_path: str | Path) -> sapien.render.RenderMaterial:
    texture = sapien.render.RenderTexture2D(
        filename=str(texture_path),
        mipmap_levels=4,
        srgb=True,
    )
    material = sapien.render.RenderMaterial()
    # Scene lights sit inside the sphere; inward normals receive that lighting.
    # Emission is kept off so the photo does not clip to white.
    material.set_base_color([1.0, 1.0, 1.0, 1.0])
    material.set_base_color_texture(texture)
    material.set_emission([0.0, 0.0, 0.0, 1.0])
    material.set_metallic(0.0)
    material.set_roughness(1.0)
    material.set_specular(0.0)
    return material


def build_pano_sphere_actor(
    scene,
    texture_path: str | Path,
    name: str,
    radius: float = DEFAULT_PANO_SPHERE_RADIUS,
    initial_pose: sapien.Pose | None = None,
):
    """Build a kinematic inverted sphere tracked by GPU PhysX.

    A tiny collision is placed far above the actor origin so the body is in the
    GPU pose buffer (needed for yaw updates) without touching the table.
    Returns ``(actor, material)`` so the texture can be swapped per episode.
    """
    vertices, triangles, normals, uvs = inverted_uv_sphere_mesh(radius)
    material = make_unlit_pano_material(texture_path)
    shape = sapien.render.RenderShapeTriangleMesh(
        vertices=vertices,
        triangles=triangles,
        normals=normals,
        uvs=uvs,
        material=material,
    )
    builder = scene.create_actor_builder()
    builder._procedural_shapes.append(shape)
    builder.add_box_collision(
        pose=sapien.Pose(p=[0.0, 0.0, 40.0]),
        half_size=[0.02, 0.02, 0.02],
    )
    pose = initial_pose if initial_pose is not None else sapien.Pose()
    builder.set_initial_pose(pose)
    actor = builder.build_kinematic(name=name)
    return actor, material


def load_pano_texture(texture_path: str | Path) -> sapien.render.RenderTexture2D:
    return sapien.render.RenderTexture2D(
        filename=str(texture_path),
        mipmap_levels=4,
        srgb=True,
    )

