"""Run explicitly on the simulation container's NVIDIA GPU (no CPU renderer)."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import sapien
import yaml

from openreal2sim.simulation.maniskill.utils.hand_visual_materials import apply_hand_visual_profile


def test_real_sapien_read_only_material_api_preserves_mesh_and_other_components():
    original = sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1])
    entities = []
    for _ in range(2):
        entity = sapien.Entity()
        body = sapien.render.RenderBodyComponent()
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=np.eye(3, dtype=np.float32), triangles=np.array([[0, 1, 2]], dtype=np.uint32),
            normals=np.ones((3, 3), dtype=np.float32), uvs=np.zeros((3, 2), dtype=np.float32),
            material=original)
        shape.local_pose = sapien.Pose([1, 2, 3])
        shape.scale = [0.001] * 3
        body.attach(shape)
        entity.add_component(body)
        entity.add_component(sapien.physx.PhysxRigidDynamicComponent())
        entities.append(entity)
    robot = SimpleNamespace(links_map={'hand': SimpleNamespace(_objs=[
        SimpleNamespace(entity=entities[0])])})
    physics = entities[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    old_body = entities[0].find_component_by_type(sapien.render.RenderBodyComponent)
    profile = {'materials': {'plastic': {'base_color': [0.45, 0.46, 0.41, 1], 'roughness': 0.8}},
               'links': {'hand': 'plastic'}}
    apply_hand_visual_profile(robot, profile)
    body = entities[0].find_component_by_type(sapien.render.RenderBodyComponent)
    assert body is not old_body
    assert entities[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) is physics
    shape = body.render_shapes[0]
    np.testing.assert_array_equal(shape.parts[0].vertices, np.eye(3))
    np.testing.assert_array_equal(shape.parts[0].triangles, [[0, 1, 2]])
    np.testing.assert_array_equal(shape.parts[0].get_vertex_normal(), np.ones((3, 3)))
    np.testing.assert_array_equal(shape.parts[0].get_vertex_uv(), np.zeros((3, 2)))
    np.testing.assert_allclose(shape.scale, [0.001] * 3)
    np.testing.assert_allclose(shape.local_pose.p, [1, 2, 3])
    np.testing.assert_allclose(shape.parts[0].material.base_color, [0.45, 0.46, 0.41, 1])
    other = entities[1].find_component_by_type(sapien.render.RenderBodyComponent)
    np.testing.assert_allclose(other.render_shapes[0].parts[0].material.base_color, [0.8, 0.8, 0.8, 1])


def test_camera_actual_mesh_split_preserves_geometry_and_physics():
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load((root / 'config/config_debug.yaml').read_text())['hand_visual_profiles']['rc5_real_matte_v1']
    mesh = root / 'openreal2sim/simulation/maniskill/robot_assets/rc5_aero_hand/urdf_rc5_right_hand/prehand_meshes/prehand_cam.dae'
    source = sapien.render.RenderShapeTriangleMesh(str(mesh), scale=[0.001] * 3)
    assert len(source.parts) == 4
    colors = [p.material.base_color.copy() for p in source.parts]
    entity = sapien.Entity()
    original_body = sapien.render.RenderBodyComponent()
    original_body.attach(source)
    entity.add_component(original_body)
    physics = sapien.physx.PhysxRigidDynamicComponent()
    entity.add_component(physics)
    robot = SimpleNamespace(links_map={'prehand': SimpleNamespace(_objs=[SimpleNamespace(entity=entity)])})
    apply_hand_visual_profile(robot, {'materials': profile['materials'], 'parts': profile['parts']})
    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    assert entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) is physics
    assert body is not original_body
    parts = [p for s in body.render_shapes for p in s.parts]
    assert len(parts) == 5
    for i, (shape, part) in enumerate(zip(body.render_shapes, parts)):
        src = source.parts[min(i, 3)]
        np.testing.assert_array_equal(part.vertices, src.vertices)
        np.testing.assert_array_equal(part.get_vertex_normal(), src.get_vertex_normal())
        np.testing.assert_array_equal(part.get_vertex_uv(), src.get_vertex_uv())
        np.testing.assert_allclose(shape.scale, source.scale)
        np.testing.assert_allclose(shape.local_pose.to_transformation_matrix(), source.local_pose.to_transformation_matrix())
        assert shape.front_face == source.front_face
        if i < 3:
            np.testing.assert_array_equal(part.triangles, src.triangles)
    assert len(parts[3].triangles) == 4326
    assert len(parts[4].triangles) == 1710
    combined = np.concatenate([parts[3].triangles, parts[4].triangles])
    assert sorted(map(tuple, combined)) == sorted(map(tuple, source.parts[3].triangles))
    assert parts[4].vertices[parts[4].triangles, 2].max() <= 73.00001
    for part, material_name in zip(parts[3:], ['camera_metal', 'camera_panel']):
        for field, expected in profile['materials'][material_name].items():
            np.testing.assert_allclose(getattr(part.material, field), expected)
    for part, color in zip(source.parts, colors):
        np.testing.assert_array_equal(part.material.base_color, color)
