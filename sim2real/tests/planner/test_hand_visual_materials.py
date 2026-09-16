import copy
from types import SimpleNamespace

import numpy as np
import pytest
import sapien

from openreal2sim.simulation.maniskill.envs import openr2s_ms_env as env_module
from openreal2sim.simulation.maniskill.utils import hand_visual_materials as materials_module


class Material:
    def __init__(self):
        self.base_color = [0.8, 0.8, 0.8, 1]
        self.emission = [0, 0, 0, 0]
        self.metallic = 0
        self.roughness = 0.3
        self.specular = 0.9
        self.ior = 1.5
        self.transmission = self.transmission_roughness = 0
        for field in ('base_color_texture', 'emission_texture', 'normal_texture',
                      'roughness_texture', 'metallic_texture', 'transmission_texture'):
            setattr(self, field, None)


class Part:
    def __init__(self, material):
        self._material = material
        self.vertices = np.eye(3)
        self.triangles = np.array([[0, 1, 2]])

    @property
    def material(self):
        return self._material

    def get_vertex_normal(self):
        return np.ones((3, 3))

    def get_vertex_uv(self):
        return np.zeros((3, 2))


class Shape:
    def __init__(self, vertices, triangles, normals, uvs, material):
        self.parts = [Part(material)]
        self.parts[0].vertices = vertices.copy()
        self.parts[0].triangles = triangles.copy()
        self.local_pose = 'original-pose'
        self.scale = [0.001] * 3
        self.front_face = 'counterclockwise'
        self.name = 'original-shape'


class Body:
    def __init__(self):
        self.render_shapes = []
        self.name = 'original-body'
        self.visibility = 0.75
        self.shading_mode = 0

    def attach(self, shape):
        self.render_shapes.append(shape)

    @property
    def is_enabled(self):
        return True

    @property
    def is_render_id_disabled(self):
        return False


class Entity:
    def __init__(self, body):
        self.body = body

    def find_component_by_type(self, _):
        return self.body

    def remove_component(self, body):
        assert body is self.body
        self.body = None

    def add_component(self, body):
        self.body = body


@pytest.fixture(autouse=True)
def render_api(monkeypatch):
    monkeypatch.setattr(sapien.render, 'RenderMaterial', Material)
    monkeypatch.setattr(sapien.render, 'RenderShapeTriangleMesh', Shape)
    monkeypatch.setattr(sapien.render, 'RenderBodyComponent', Body)


def fixture_robot():
    original = Material()
    parts = {}
    links = {}
    for name, count in [('body0', 1), ('right_base_link', 1), ('right_index_tip_link', 1), ('prehand', 4)]:
        parts[name] = [Part(original) for _ in range(count)]
        shape = Shape(np.eye(3), np.array([[0, 1, 2]]), None, None, original)
        shape.parts = parts[name]
        body = Body()
        body.attach(shape)
        # Two sub-scenes share source materials, as can happen with cached meshes.
        links[name] = SimpleNamespace(_objs=[SimpleNamespace(entity=Entity(body)) for _ in range(2)])
    return SimpleNamespace(links_map=links), parts, original


def profile():
    return {'name': 'test', 'materials': {
        'plastic': {'base_color': [0.45, 0.46, 0.41, 1], 'roughness': 0.8, 'specular': 0.2},
        'pad': {'base_color': [0.035, 0.035, 0.035, 1]},
        'green': {'base_color': [0.035, 0.22, 0.025, 1]},
    }, 'links': {'right_base_link': 'plastic', 'right_index_tip_link': 'pad'},
        'parts': {'prehand': ['green', 'green', 'green', 'plastic']}}


def test_exact_link_and_part_materials_do_not_mutate_shared_arm_material(monkeypatch):
    monkeypatch.setattr(sapien.render, 'RenderMaterial', Material)
    robot, parts, original = fixture_robot()
    original.base_color_texture = object()
    p = profile()
    saved = copy.deepcopy(p)
    env_module.apply_hand_visual_profile(robot, p)
    assert parts['body0'][0].material is original
    assert original.base_color == [0.8, 0.8, 0.8, 1]
    for name, link in robot.links_map.items():
        for obj in link._objs:
            body = obj.entity.body
            current = [part for shape in body.render_shapes for part in shape.parts]
            if name == 'body0':
                assert current[0].material is original
                continue
            selected = [p['links'][name]] if name in p['links'] else p['parts'][name]
            assert body.visibility == 0.75
            assert body.name == 'original-body'
            for shape in body.render_shapes:
                assert shape.local_pose == 'original-pose'
                assert shape.scale == [0.001] * 3
                assert shape.front_face == 'counterclockwise'
            for part, material in zip(current, selected):
                assert part.material.base_color == p['materials'][material]['base_color']
                assert part.material.base_color_texture is original.base_color_texture
                np.testing.assert_array_equal(part.vertices, np.eye(3))
                np.testing.assert_array_equal(part.triangles, [[0, 1, 2]])
    assert p == saved


@pytest.mark.parametrize('problem', ['missing_link', 'unknown_material', 'part_count', 'nan', 'unknown_field'])
def test_invalid_profile_fails_before_any_material_changes(monkeypatch, problem):
    monkeypatch.setattr(sapien.render, 'RenderMaterial', Material)
    robot, parts, original = fixture_robot()
    p = profile()
    if problem == 'missing_link': p['links']['nonexistent'] = 'plastic'
    if problem == 'unknown_material': p['links']['right_base_link'] = 'unknown'
    if problem == 'part_count': p['parts']['prehand'].pop()
    if problem == 'nan': p['materials']['plastic']['specular'] = float('nan')
    if problem == 'unknown_field': p['materials']['plastic']['roughnes'] = 0.8
    with pytest.raises((ValueError, RuntimeError), match='hand_visual_profile'):
        env_module.apply_hand_visual_profile(robot, p)
    assert all(part.material is original for group in parts.values() for part in group)


def camera_selection():
    return {'material': 'plastic', 'expected_triangles': 2, 'regions': [
        {'material': 'pad', 'bounds_min': [-1, -1, -0.01], 'bounds_max': [2, 2, 0.01],
         'expected_triangles': 1}]}


def camera_fixture():
    robot, parts, original = fixture_robot()
    part = parts['prehand'][3]
    part.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                              [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float)
    part.triangles = np.array([[0, 1, 2], [3, 4, 5]])
    p = profile()
    p['parts']['prehand'][3] = camera_selection()
    return robot, parts, original, p


def test_camera_front_split_preserves_every_triangle_and_other_parts():
    robot, parts, original, p = camera_fixture()
    materials_module.apply_hand_visual_profile(robot, p)
    for obj in robot.links_map['prehand']._objs:
        shapes = obj.entity.body.render_shapes
        assert len(shapes) == 5
        assert all(s.parts[0].material.base_color == p['materials']['green']['base_color'] for s in shapes[:3])
        base, front = [s.parts[0] for s in shapes[3:]]
        assert base.material.base_color == p['materials']['plastic']['base_color']
        assert front.material.base_color == p['materials']['pad']['base_color']
        np.testing.assert_array_equal(base.triangles, [[3, 4, 5]])
        np.testing.assert_array_equal(front.triangles, [[0, 1, 2]])
        for part in (base, front):
            np.testing.assert_array_equal(part.vertices, parts['prehand'][3].vertices)
        assert all(s.local_pose == 'original-pose' and s.scale == [0.001] * 3 for s in shapes)
    assert parts['prehand'][3].material is original


@pytest.mark.parametrize('problem', ['total_count', 'region_count', 'overlap', 'missing_material', 'bounds', 'typo'])
def test_invalid_camera_region_fails_before_replacing_any_body(problem):
    robot, parts, original, p = camera_fixture()
    selection = p['parts']['prehand'][3]
    if problem == 'total_count': selection['expected_triangles'] = 3
    if problem == 'region_count': selection['regions'][0]['expected_triangles'] = 2
    if problem == 'overlap': selection['regions'] *= 2
    if problem == 'missing_material': selection['regions'][0]['material'] = 'missing'
    if problem == 'bounds': selection['regions'][0]['bounds_min'] = [float('nan')] * 3
    if problem == 'typo': selection['regions'][0]['bound_max'] = [2, 2, 2]
    bodies = [obj.entity.body for link in robot.links_map.values() for obj in link._objs]
    with pytest.raises((ValueError, RuntimeError), match='hand_visual_profile'):
        materials_module.apply_hand_visual_profile(robot, p)
    assert [obj.entity.body for link in robot.links_map.values() for obj in link._objs] == bodies
