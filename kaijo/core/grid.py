# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluation grids.

Only the rectangular (rectilinear, axis-aligned) grid is implemented, but
grids are behind a small registry so that other grid types (adaptive,
molecular-shaped, ...) can be added without touching the rest of the code.
A grid must provide `axis(i)`, `shape`, `spacing`, `origin`, and
`describe()`; the isosurface code additionally relies on the grid being
usable by `skimage.measure.marching_cubes` via uniform spacing.
"""

import numpy as np


class RectilinearGrid:
    """Axis-aligned rectangular grid with uniform spacing per axis."""

    kind = "rectangular"

    def __init__(self, origin, spacing, shape):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.spacing = np.asarray(spacing, dtype=np.float64)
        self.shape = tuple(int(n) for n in shape)
        self._axes = [self.origin[i] + self.spacing[i] * np.arange(self.shape[i])
                      for i in range(3)]

    def axis(self, i):
        return self._axes[i]

    @property
    def npoints(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def describe(self):
        return (f"{self.shape[0]}x{self.shape[1]}x{self.shape[2]} points, "
                f"spacing {self.spacing[0]:.3f} bohr")

    def nbytes(self, dtype_size=4):
        return self.npoints * dtype_size

    @classmethod
    def for_molecule(cls, molecule, spacing=0.35, margin=4.0):
        """Bounding-box grid around the molecule (units: bohr)."""
        pos = molecule.coords
        lo = pos.min(axis=0) - margin
        hi = pos.max(axis=0) + margin
        shape = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 8)
        return cls(lo, (spacing, spacing, spacing), shape)


GRID_TYPES = {
    RectilinearGrid.kind: RectilinearGrid,
}


def make_grid(kind, molecule, **options):
    try:
        gcls = GRID_TYPES[kind]
    except KeyError:
        raise ValueError(f"Unknown grid type '{kind}'") from None
    return gcls.for_molecule(molecule, **options)
