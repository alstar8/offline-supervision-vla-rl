"""Strict, isolated visual material overrides for robot links and scene objects."""

import numpy as np
import sapien


def apply_hand_visual_profile(robot, profile):
    entities = {name: [link.entity for link in links._objs] for name, links in robot.links_map.items()}
    _apply_visual_material_profile(entities, profile, 'hand_visual_profile', 'HandVisual')


def apply_object_visual_material(actor, config, object_id):
    scope = f'object_placements.{object_id}.render_material'
    profile = {'name': str(object_id), 'materials': {'object': config}, 'links': {str(object_id): 'object'}}
    _apply_visual_material_profile({str(object_id): actor._objs}, profile, scope, 'ObjectVisual')


def _material_triangle_groups(part, selection, materials, scope):
    def material(name):
        if not isinstance(name, str) or name not in materials:
            raise ValueError(f'{scope}: unknown material {name!r}')
        return materials[name]

    if isinstance(selection, str):
        return [(material(selection), part.triangles)]
    if not isinstance(selection, dict) or set(selection) != {'material', 'regions', 'expected_triangles'}:
        raise ValueError(f'{scope}: expected material/regions/expected_triangles')
    base = material(selection['material'])
    triangles = np.asarray(part.triangles)

    def check_count(expected, actual):
        if type(expected) is not int or expected <= 0 or expected != actual:
            raise ValueError(f'{scope}: expected_triangles={expected!r}, actual={actual}')

    check_count(selection['expected_triangles'], len(triangles))
    regions = selection['regions']
    if not isinstance(regions, list) or not regions:
        raise ValueError(f'{scope}: regions must be a nonempty list')
    points = np.asarray(part.vertices)[triangles]
    used = np.zeros(len(triangles), dtype=bool)
    groups = []
    for region in regions:
        if not isinstance(region, dict) or set(region) != {'material', 'bounds_min', 'bounds_max', 'expected_triangles'}:
            raise ValueError(f'{scope}: invalid face region fields')
        selected_material = material(region['material'])
        try:
            bounds = np.asarray([region['bounds_min'], region['bounds_max']], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{scope}: invalid region bounds') from exc
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[0] > bounds[1]).any():
            raise ValueError(f'{scope}: expected finite ordered 3D region bounds')
        # Bounds use the original mesh coordinates, before the URDF visual scale.
        mask = ((points >= bounds[0] - 1e-5) & (points <= bounds[1] + 1e-5)).all(axis=(1, 2))
        check_count(region['expected_triangles'], int(mask.sum()))
        if (used & mask).any():
            raise ValueError(f'{scope}: overlapping face regions')
        used |= mask
        groups.append((selected_material, triangles[mask]))
    if used.all():
        raise ValueError(f'{scope}: no triangles remain for the base material')
    return [(base, triangles[~used])] + groups


def _apply_visual_material_profile(entities, profile, scope, log_label):
    if not isinstance(profile, dict) or set(profile) - {'name', 'materials', 'links', 'parts'}:
        raise ValueError(f'{scope}: expected name/materials/links/parts mapping')
    materials = profile.get('materials')
    links, parts = profile.get('links', {}), profile.get('parts', {})
    if not isinstance(materials, dict) or not materials or not isinstance(links, dict) or not isinstance(parts, dict):
        raise ValueError(f'{scope}: nonempty materials and link mappings are required')
    if not (links or parts) or set(links) & set(parts):
        raise ValueError(f'{scope}: links/parts must be nonempty and disjoint')
    allowed = {'base_color', 'metallic', 'roughness', 'specular', 'use_base_color_texture'}
    values = {}
    for name, config in materials.items():
        if not isinstance(config, dict) or not config or set(config) - allowed:
            raise ValueError(f'{scope}.{name}: unknown or empty material parameters')
        values[name] = {}
        for key, value in config.items():
            if key == 'use_base_color_texture':
                if not isinstance(value, bool):
                    raise ValueError(f'{scope}.{name}.{key}: expected a boolean')
                values[name][key] = value
                continue
            try:
                a = np.asarray(value, dtype=float)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{scope}.{name}.{key}: expected numeric values') from exc
            shape = (4,) if key == 'base_color' else ()
            if a.shape != shape or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
                raise ValueError(f'{scope}.{name}.{key}: expected shape {shape}, finite values in [0,1]')
            values[name][key] = a.tolist() if shape else float(a)

    assignments = []
    for name in list(links) + list(parts):
        if name not in entities or not entities[name]:
            raise RuntimeError(f'{scope}: no rendered entities for {name!r}')
        selected = [links[name]] if name in links else parts[name]
        if not isinstance(selected, list) or not selected:
            raise ValueError(f'{scope}: invalid material selection for {name!r}')
        for entity in entities[name]:
            body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            mesh_parts = [] if body is None else [p for s in body.render_shapes for p in s.parts]
            if not mesh_parts or (name in parts and len(mesh_parts) != len(selected)):
                raise RuntimeError(f'{scope}: unexpected mesh part count for {name!r}: {len(mesh_parts)}')
            if any(not isinstance(s, sapien.render.RenderShapeTriangleMesh) for s in body.render_shapes):
                raise RuntimeError(f'{scope}: expected triangle meshes for {name!r}')
            configs = [_material_triangle_groups(part, selected[0] if name in links else selected[i],
                                                values, f'{scope}.{name}.part[{i}]')
                       for i, part in enumerate(mesh_parts)]
            assignments.append((entity, body, configs))
    if not assignments:
        raise RuntimeError(f'{scope}: no rendered links matched')

    # SAPIEN mesh-part materials are read-only and cached/shared. Recreate only
    # visual components with private materials, preserving mesh data/transforms.
    # Prepare everything before replacing anything; physics components stay put.
    copies = []
    fields = ('base_color', 'emission', 'metallic', 'roughness', 'specular', 'ior',
              'transmission', 'transmission_roughness', 'base_color_texture',
              'emission_texture', 'normal_texture', 'roughness_texture',
              'metallic_texture', 'transmission_texture')
    part_count = 0
    for entity, old_body, configs in assignments:
        body = sapien.render.RenderBodyComponent()
        body.name = old_body.name
        body.visibility = old_body.visibility
        body.shading_mode = old_body.shading_mode
        if not old_body.is_enabled:
            body.disable()
        if old_body.is_render_id_disabled:
            body.disable_render_id()
        config_iter = iter(configs)
        for old_shape in old_body.render_shapes:
            for part in old_shape.parts:
                for config, triangles in next(config_iter):
                    material = sapien.render.RenderMaterial()
                    for field in fields:
                        setattr(material, field, getattr(part.material, field))
                    for field, value in config.items():
                        if field == 'use_base_color_texture':
                            if not value:
                                material.base_color_texture = None
                        else:
                            setattr(material, field, value)
                    shape = sapien.render.RenderShapeTriangleMesh(
                        vertices=part.vertices, triangles=triangles,
                        normals=part.get_vertex_normal(), uvs=part.get_vertex_uv(), material=material)
                    shape.local_pose = old_shape.local_pose
                    shape.scale = old_shape.scale
                    shape.front_face = old_shape.front_face
                    shape.name = old_shape.name
                    body.attach(shape)
                    part_count += 1
        copies.append((entity, old_body, body))
    for entity, old_body, body in copies:
        entity.remove_component(old_body)
        entity.add_component(body)
    print(f"[{log_label}] Applied profile={profile.get('name', '<unnamed>')} "
          f"links={len(links) + len(parts)} parts={part_count}")
