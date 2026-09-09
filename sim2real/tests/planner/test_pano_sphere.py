from __future__ import annotations

import numpy as np

from openreal2sim.simulation.maniskill.utils.pano_sphere import (
    DEFAULT_360_PHOTOS_DIR,
    dual_fisheye_to_equirect,
    inverted_uv_sphere_mesh,
    list_360_photo_paths,
)
from openreal2sim.simulation.maniskill.utils.scene_loader import SIM2REAL_REPO_ROOT


def test_list_360_photos_finds_insp_captures() -> None:
    photos = list_360_photo_paths()
    assert DEFAULT_360_PHOTOS_DIR == SIM2REAL_REPO_ROOT / "assets" / "360_photos"
    assert len(photos) >= 1
    assert all(path.suffix.lower() == ".insp" for path in photos)
    assert all(path.parent == DEFAULT_360_PHOTOS_DIR for path in photos)


def test_inverted_uv_sphere_has_inward_normals_and_equirect_uvs() -> None:
    verts, tris, normals, uvs = inverted_uv_sphere_mesh(radius=2.0, n_lat=8, n_lon=16)
    assert verts.shape[1] == 3
    assert tris.shape[1] == 3
    assert verts.shape[0] == normals.shape[0] == uvs.shape[0]
    radial = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    inward = np.sum(normals * radial, axis=1)
    assert float(inward.max()) < 0.0
    assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0)
    # u is flipped for inside viewing: first longitude sample is u=1
    assert float(uvs[0, 0]) == 1.0
    assert int(tris.max()) < len(verts)


def test_dual_fisheye_to_equirect_maps_front_circle_to_center() -> None:
    h, w = 64, 128
    image = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    left = (xx - w * 0.25) ** 2 + (yy - h * 0.5) ** 2 <= (0.45 * h) ** 2
    right = (xx - w * 0.75) ** 2 + (yy - h * 0.5) ** 2 <= (0.45 * h) ** 2
    image[left] = (255, 0, 0)
    image[right] = (0, 255, 0)
    equirect = dual_fisheye_to_equirect(image, fov_deg=190.0, width=128)
    assert equirect.shape == (64, 128, 3)
    center = equirect[32, 64]
    # Front (+Z) lands at the image center and should sample the left (red) circle.
    assert int(center[0]) > int(center[1]) + 50


def test_rl_gym_kwargs_enable_360_background() -> None:
    from openreal2sim.simulation.maniskill.rl_gym import build_openreal2sim_rl_gym_kwargs

    kwargs = build_openreal2sim_rl_gym_kwargs(initialize_renderer=False, use_wrist_camera=True)
    assert kwargs["use_360_background"] is True
