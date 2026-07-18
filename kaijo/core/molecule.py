# SPDX-License-Identifier: GPL-3.0-or-later
"""Molecule model: atoms, automatic bond perception, manual bond overrides.

Internally all coordinates are in bohr (atomic units), matching the
quantum-chemistry data.  Conversion to Å happens only for display of
numbers to the user.
"""

import numpy as np

from . import elements
from .elements import BOHR


class Bond:
    __slots__ = ("i", "j", "order")

    def __init__(self, i, j, order=1):
        self.i = i
        self.j = j
        self.order = order  # 0 = suppressed, 1, 2, 3

    def key(self):
        return (min(self.i, self.j), max(self.i, self.j))


class Molecule:
    """A set of atoms with perceived bonds.

    Atom deletion is non-destructive: deleted atoms are only masked out so
    the operation is undoable and never touches the underlying files.
    """

    def __init__(self, numbers, coords_bohr, name=""):
        self.numbers = np.asarray(numbers, dtype=np.int32)
        self.coords = np.asarray(coords_bohr, dtype=np.float64)
        self.name = name
        self.visible = np.ones(len(self.numbers), dtype=bool)
        # manual overrides: {(i, j): order or None}; None = revert to auto
        self.bond_overrides = {}
        self._auto_bonds = None

    @property
    def natoms(self):
        return len(self.numbers)

    def center(self):
        vis = self.visible
        if not vis.any():
            return np.zeros(3)
        return self.coords[vis].mean(axis=0)

    def extent(self):
        """Radius (bohr) of the bounding sphere around the centroid."""
        vis = self.visible
        if not vis.any():
            return 5.0
        c = self.center()
        d = np.linalg.norm(self.coords[vis] - c, axis=1)
        rad = np.array([elements.display_radius(z) / BOHR
                        for z in self.numbers[vis]])
        return float((d + rad).max())

    # ------------------------------------------------------------------ bonds

    def _perceive_bonds(self):
        """Automatic bond perception from the Pyykkö additive radii."""
        bonds = []
        n = self.natoms
        if n < 2:
            return bonds
        pos = self.coords * BOHR  # work in Å
        zs = self.numbers
        r1 = np.array([elements.covalent_r1(z) for z in zs])
        # Neighbour search: simple O(n^2) distance matrix is fine up to a few
        # thousand atoms; use blocking to keep memory bounded.
        maxcut = (r1.max() * 2 + 0.45)
        for i0 in range(0, n, 512):
            i1 = min(i0 + 512, n)
            d = np.linalg.norm(pos[i0:i1, None, :] - pos[None, :, :], axis=2)
            for ii in range(i0, i1):
                row = d[ii - i0]
                cand = np.nonzero((row < maxcut) & (np.arange(n) > ii))[0]
                for jj in cand:
                    zi, zj = zs[ii], zs[jj]
                    dij = row[jj]
                    if dij < 1e-3:
                        continue
                    s1 = elements.covalent_r1(zi) + elements.covalent_r1(zj)
                    if dij > 1.15 * s1 + 0.05:
                        continue
                    order = 1
                    r3i, r3j = elements.covalent_r3(zi), elements.covalent_r3(zj)
                    r2i, r2j = elements.covalent_r2(zi), elements.covalent_r2(zj)
                    if r3i and r3j and dij <= (r3i + r3j) * 1.04:
                        order = 3
                    elif r2i and r2j and dij <= (r2i + r2j) * 1.04:
                        order = 2
                    bonds.append(Bond(ii, int(jj), order))
        return bonds

    def bonds(self):
        """Current bond list with overrides and visibility applied."""
        if self._auto_bonds is None:
            self._auto_bonds = self._perceive_bonds()
        out = []
        seen = set()
        for b in self._auto_bonds:
            key = b.key()
            seen.add(key)
            order = self.bond_overrides.get(key, b.order)
            if order and self.visible[b.i] and self.visible[b.j]:
                out.append(Bond(b.i, b.j, order))
        # overrides may create bonds that auto-perception did not find
        for key, order in self.bond_overrides.items():
            if key in seen or not order:
                continue
            i, j = key
            if self.visible[i] and self.visible[j]:
                out.append(Bond(i, j, order))
        return out

    def set_bond_override(self, i, j, order):
        """order: 0 = none, 1..3, or None to revert to automatic."""
        key = (min(i, j), max(i, j))
        prev = self.bond_overrides.get(key, "auto")
        if order is None:
            self.bond_overrides.pop(key, None)
        else:
            self.bond_overrides[key] = int(order)
        return prev  # for undo

    def restore_bond_override(self, i, j, prev):
        key = (min(i, j), max(i, j))
        if prev == "auto":
            self.bond_overrides.pop(key, None)
        else:
            self.bond_overrides[key] = prev

    # ------------------------------------------------------------ atom masking

    def hide_atoms(self, indices):
        self.visible[list(indices)] = False

    def show_atoms(self, indices):
        self.visible[list(indices)] = True
