# SPDX-License-Identifier: GPL-3.0-or-later
"""Isosurface extraction via marching cubes (skimage, C implementation)."""

import numpy as np
from skimage import measure


class SurfaceMesh:
    __slots__ = ("vertices", "normals", "faces")

    def __init__(self, vertices, normals, faces):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.uint32)

    @property
    def empty(self):
        return len(self.faces) == 0

    def nbytes(self):
        return (self.vertices.nbytes + self.normals.nbytes
                + self.faces.nbytes)


_EMPTY = None


def empty_mesh():
    global _EMPTY
    if _EMPTY is None:
        _EMPTY = SurfaceMesh(np.zeros((0, 3)), np.zeros((0, 3)),
                             np.zeros((0, 3)))
    return _EMPTY


def extract(volume, level, grid):
    """Extract the isosurface volume == level.  Vertices in grid (bohr)
    coordinates.  Returns an empty mesh when the level is not crossed."""
    vmin, vmax = float(volume.min()), float(volume.max())
    if not vmin < level < vmax:
        return empty_mesh()
    verts, faces, normals, _ = measure.marching_cubes(
        volume, level=level, spacing=tuple(grid.spacing),
        gradient_direction="descent" if level >= 0 else "ascent")
    verts = verts + grid.origin
    if level < 0:
        # keep outward orientation for the negative lobe
        normals = -normals
        faces = faces[:, ::-1]
    return SurfaceMesh(verts, normals, faces)


def extract_pair(volume, isovalue, grid):
    """(positive lobe, negative lobe) meshes for +/-isovalue."""
    return (extract(volume, abs(isovalue), grid),
            extract(volume, -abs(isovalue), grid))
