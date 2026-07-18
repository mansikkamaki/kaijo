# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit-primitive meshes (generated once, instanced by the renderer)."""

import numpy as np


def uv_sphere(stacks=14, slices=20):
    """Unit sphere; returns (verts, normals, faces)."""
    verts = []
    for i in range(stacks + 1):
        theta = np.pi * i / stacks
        st, ct = np.sin(theta), np.cos(theta)
        for j in range(slices):
            phi = 2 * np.pi * j / slices
            verts.append((st * np.cos(phi), st * np.sin(phi), ct))
    verts = np.array(verts, dtype=np.float32)
    faces = []
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = i * slices + (j + 1) % slices
            c = a + slices
            d = b + slices
            # counter-clockwise seen from outside (front faces survive
            # back-face culling)
            if i > 0:
                faces.append((a, c, b))
            if i < stacks - 1:
                faces.append((b, c, d))
    return verts, verts.copy(), np.array(faces, dtype=np.uint32)


def cylinder(slices=18, caps=True):
    """Unit cylinder along +z from z=0 to z=1, radius 1."""
    ang = 2 * np.pi * np.arange(slices) / slices
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    verts, normals, faces = [], [], []
    for z in (0.0, 1.0):
        for cx, cy in ring:
            verts.append((cx, cy, z))
            normals.append((cx, cy, 0.0))
    for j in range(slices):
        a, b = j, (j + 1) % slices
        c, d = a + slices, b + slices
        faces.append((a, b, c))
        faces.append((b, d, c))
    if caps:
        base = len(verts)
        for z, nz in ((0.0, -1.0), (1.0, 1.0)):
            centre = len(verts)
            verts.append((0.0, 0.0, z))
            normals.append((0.0, 0.0, nz))
            for cx, cy in ring:
                verts.append((cx, cy, z))
                normals.append((0.0, 0.0, nz))
            for j in range(slices):
                a = centre + 1 + j
                b = centre + 1 + (j + 1) % slices
                if nz > 0:
                    faces.append((centre, a, b))
                else:
                    faces.append((centre, b, a))
        del base
    return (np.array(verts, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(faces, dtype=np.uint32))


def cone(slices=20):
    """Cone: base radius 1 at z=0, apex at z=1, with base cap."""
    ang = 2 * np.pi * np.arange(slices) / slices
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
    verts, normals, faces = [], [], []
    # side: normal of cone with half-angle: n = normalize((cos, sin, 1))
    inv = 1.0 / np.sqrt(2.0)
    for cx, cy in ring:
        verts.append((cx, cy, 0.0))
        normals.append((cx * inv, cy * inv, inv))
    apex_start = len(verts)
    for cx, cy in ring:  # duplicated apex per segment for decent shading
        verts.append((0.0, 0.0, 1.0))
        normals.append((cx * inv, cy * inv, inv))
    for j in range(slices):
        a, b = j, (j + 1) % slices
        faces.append((a, b, apex_start + j))
    centre = len(verts)
    verts.append((0.0, 0.0, 0.0))
    normals.append((0.0, 0.0, -1.0))
    for cx, cy in ring:
        verts.append((cx, cy, 0.0))
        normals.append((0.0, 0.0, -1.0))
    for j in range(slices):
        a = centre + 1 + j
        b = centre + 1 + (j + 1) % slices
        faces.append((centre, b, a))
    return (np.array(verts, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(faces, dtype=np.uint32))


def disc(slices=48):
    """Unit disc in the z=0 plane (lit two-sided by the surface shader)."""
    ang = 2 * np.pi * np.arange(slices) / slices
    verts = [(0.0, 0.0, 0.0)]
    verts += [(np.cos(a), np.sin(a), 0.0) for a in ang]
    normals = [(0.0, 0.0, 1.0)] * (slices + 1)
    faces = [(0, 1 + j, 1 + (j + 1) % slices) for j in range(slices)]
    return (np.array(verts, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(faces, dtype=np.uint32))


def instance_matrix(position, radius):
    """Uniform-scale sphere instance: rows [R*S | t] flattened."""
    m = np.eye(4, dtype=np.float32) * radius
    m[3, 3] = 1.0
    m[:3, 3] = position
    return m


def bond_matrix(p0, p1, radius):
    """Transform mapping the unit cylinder (z 0..1) onto segment p0->p1."""
    p0 = np.asarray(p0, dtype=np.float64)
    d = np.asarray(p1, dtype=np.float64) - p0
    length = np.linalg.norm(d)
    if length < 1e-9:
        return None
    z = d / length
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = x * radius
    m[:3, 1] = y * radius
    m[:3, 2] = z * length
    m[:3, 3] = p0
    return m


def perpendicular(v):
    """Any unit vector perpendicular to v (stable choice)."""
    v = np.asarray(v, dtype=np.float64)
    ref = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    p = np.cross(v, ref)
    return p / np.linalg.norm(p)
