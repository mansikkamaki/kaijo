# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene model: the molecular structure plus displayable items.

Each preview-grid cell shows the structure plus (at most) one item -- an
orbital isosurface pair or a geometric object.  The scene owns the item
list; the renderer consumes instance arrays and meshes built here.
All world coordinates are bohr.
"""

import itertools

import numpy as np

from ..core import elements
from ..core.elements import BOHR
from ..core.isosurface import SurfaceMesh
from . import meshes

_HIGHLIGHT = np.array([1.0, 0.75, 0.1])

_id_counter = itertools.count(1)


class SceneItem:
    kind = "item"

    def __init__(self, label=""):
        self.id = next(_id_counter)
        self.label = label
        self.export_selected = False

    def describe(self):
        return self.label


class StructureItem(SceneItem):
    """The bare molecular structure (always the first preview cell)."""
    kind = "structure"

    def __init__(self):
        super().__init__("Structure")


class OrbitalItem(SceneItem):
    kind = "orbital"

    def __init__(self, orb_index, label):
        super().__init__(label)
        self.orb_index = orb_index
        self.mesh_pos = None   # SurfaceMesh or None
        self.mesh_neg = None
        self.dirty = True      # GPU buffers need re-upload


class FieldItem(SceneItem):
    """Computed scalar-field isosurface: the electron density, the spin
    density, or the ESP mapped onto a density isosurface.  `kind` is the
    field name itself, so at most one item of each field exists."""

    def __init__(self, field, label):
        super().__init__(label)
        self.kind = field           # "density" | "spin" | "esp"
        self.mesh_pos = None        # SurfaceMesh or None
        self.mesh_neg = None
        self.esp_colors = None      # (nvert, 4) float32, kind == "esp"
        self.esp_range = 0.0        # symmetric color-scale limit (Eh/e)
        self.dirty = True

    def describe(self):
        if self.kind == "esp" and self.esp_range:
            return f"{self.label}  (±{self.esp_range:.3f} Eh)"
        return self.label


class VectorItem(SceneItem):
    kind = "vector"

    #: default extension past the atoms for a two-atom (piercing) vector
    PIERCE_EXTEND = 1.6  # bohr, each side

    def __init__(self, p0, p1, label, length=None,
                 color=(0.15, 0.55, 0.15), alpha=1.0):
        """p0, p1 define the direction and midpoint; `length` is the total
        drawn length in bohr (default: the p0-p1 distance extended by
        PIERCE_EXTEND on each side)."""
        super().__init__(label)
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.color = list(color)
        self.alpha = alpha
        self.radius = 0.16          # bohr
        if length is None:
            length = self.default_length(self.p0, self.p1)
        self.length = float(length)

    @staticmethod
    def default_length(p0, p1):
        return float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))
                     + 2.0 * VectorItem.PIERCE_EXTEND)

    def endpoints(self):
        d = self.p1 - self.p0
        n = np.linalg.norm(d)
        if n < 1e-9 or self.length < 1e-9:
            return None
        d /= n
        mid = 0.5 * (self.p0 + self.p1)
        half = 0.5 * self.length
        return mid - d * half, mid + d * half


class PlaneItem(SceneItem):
    kind = "plane"

    def __init__(self, center, normal, radius, label,
                 color=(0.5, 0.35, 0.8), alpha=0.55):
        super().__init__(label)
        self.center = np.asarray(center, dtype=np.float64)
        self.normal = np.asarray(normal, dtype=np.float64)
        self.radius = float(radius)
        self.color = list(color)
        self.alpha = alpha
        self.size_scale = 1.0

    @classmethod
    def fit(cls, points, label, **kw):
        """Least-squares plane through >= 3 points."""
        pts = np.asarray(points, dtype=np.float64)
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c)
        normal = vt[2]
        inplane = pts - c
        d = np.linalg.norm(inplane - np.outer(inplane @ normal, normal),
                           axis=1)
        radius = float(d.max()) + 1.5
        return cls(c, normal, radius, label, **kw)


class PolyhedronItem(SceneItem):
    kind = "polyhedron"

    def __init__(self, points, label, color=(0.2, 0.55, 0.85), alpha=0.55):
        super().__init__(label)
        from scipy.spatial import ConvexHull
        pts = np.asarray(points, dtype=np.float64)
        hull = ConvexHull(pts)
        # flat shading: duplicate vertices per face with the face normal
        verts, norms, faces = [], [], []
        centre = pts.mean(axis=0)
        for simplex, eq in zip(hull.simplices, hull.equations):
            n = eq[:3]
            tri = pts[simplex]
            if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), n) < 0:
                tri = tri[::-1]
            base = len(verts)
            verts.extend(tri)
            norms.extend([n] * 3)
            faces.append((base, base + 1, base + 2))
        self.mesh = SurfaceMesh(np.array(verts), np.array(norms),
                                np.array(faces))
        edges = set()
        for simplex in hull.simplices:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edges.add((min(simplex[a], simplex[b]),
                           max(simplex[a], simplex[b])))
        self.edges = np.array([[pts[a], pts[b]] for a, b in edges],
                              dtype=np.float32)
        self.color = list(color)
        self.alpha = alpha
        self.dirty = True


class Scene:
    """Molecule + items + selection; builds renderer instance arrays."""

    def __init__(self, settings):
        self.settings = settings
        self.molecule = None
        self.items = []           # SceneItem list (structure first)
        self.selected_atoms = []  # ordered selection
        self.selected_item = None
        self._structure_cache = None

    def set_molecule(self, molecule):
        self.molecule = molecule
        self.items = [StructureItem()]
        self.selected_atoms = []
        self.selected_item = None
        self.invalidate_structure()

    def invalidate_structure(self):
        self._structure_cache = None

    def add_item(self, item):
        self.items.append(item)
        return item

    def remove_item(self, item):
        if item in self.items:
            self.items.remove(item)
        if self.selected_item is item:
            self.selected_item = None

    def orbital_items(self):
        return [i for i in self.items if i.kind == "orbital"]

    def field_item(self, field):
        """The density/spin/esp item, or None (at most one per field)."""
        return next((i for i in self.items if i.kind == field), None)

    # ------------------------------------------------------- appearance data

    def atom_color(self, z):
        override = self.settings["atom_colors"].get(str(z))
        return np.array(override if override else elements.element_color(z))

    def atom_radius(self, z):
        override = self.settings["atom_radii"].get(str(z))
        r = override if override else elements.display_radius(z)
        return r * self.settings["atom_scale"] / BOHR

    # ------------------------------------------------- structure instancing

    def structure_arrays(self):
        """(sphere_instances, cylinder_instances, line_vertices,
        halo_instances) as float32.

        Instance record: 16 model + 4 color + 4 scale = 24 floats.
        Line vertex record: 3 position + 3 color = 6 floats.
        Halos are enlarged translucent spheres marking selected atoms;
        the renderer draws them with a rim-glow shader.
        """
        if self._structure_cache is not None:
            return self._structure_cache
        mol = self.molecule
        rep = self.settings["representation"]
        show_h = self.settings["show_hydrogens"]
        multi = self.settings["multiple_bonds"]
        bond_r = self.settings["bond_radius"] / BOHR
        sel = set(self.selected_atoms)

        spheres, cyls, lines, halos = [], [], [], []
        if mol is None:
            self._structure_cache = (np.zeros((0, 24), np.float32),) * 2 \
                + (np.zeros((0, 6), np.float32),
                   np.zeros((0, 24), np.float32))
            return self._structure_cache

        shown = [i for i in range(mol.natoms)
                 if mol.visible[i] and (show_h or mol.numbers[i] != 1)]
        shownset = set(shown)

        def add_sphere(pos, radius, color):
            m = meshes.instance_matrix(pos, radius)
            spheres.append(np.concatenate([
                m.T.flatten(), [*color, 1.0],
                [radius, radius, radius, 0.0]]))

        def add_cyl(p0, p1, radius, color):
            m = meshes.bond_matrix(p0, p1, radius)
            if m is None:
                return
            length = np.linalg.norm(np.asarray(p1) - np.asarray(p0))
            cyls.append(np.concatenate([
                m.T.flatten(), [*color, 1.0],
                [radius, radius, length, 0.0]]))

        def add_halo(pos, radius):
            m = meshes.instance_matrix(pos, radius)
            halos.append(np.concatenate([
                m.T.flatten(), [*_HIGHLIGHT, 0.85],
                [radius, radius, radius, 0.0]]))

        for i in shown:
            z = mol.numbers[i]
            color = self.atom_color(z)
            if i in sel:
                color = 0.35 * color + 0.65 * _HIGHLIGHT
            if rep == "ball-and-stick":
                r = self.atom_radius(z)
                add_sphere(mol.coords[i], r, color)
                if i in sel:
                    add_halo(mol.coords[i], r * 1.5)
            elif rep == "sticks":
                add_sphere(mol.coords[i], bond_r * 1.7, color)
                if i in sel:
                    add_halo(mol.coords[i], bond_r * 3.2)
            else:  # wireframe: small marker sphere at selected atoms only
                if i in sel:
                    add_sphere(mol.coords[i], bond_r, color)
                    add_halo(mol.coords[i], bond_r * 3.2)

        for b in mol.bonds():
            if b.i not in shownset or b.j not in shownset:
                continue
            pi, pj = mol.coords[b.i], mol.coords[b.j]
            ci = self.atom_color(mol.numbers[b.i])
            cj = self.atom_color(mol.numbers[b.j])
            if b.i in sel:
                ci = 0.35 * ci + 0.65 * _HIGHLIGHT
            if b.j in sel:
                cj = 0.35 * cj + 0.65 * _HIGHLIGHT
            mid = 0.5 * (pi + pj)
            norder = b.order if multi else 1
            if rep == "wireframe":
                lines.append(np.concatenate([pi, ci]))
                lines.append(np.concatenate([mid, ci]))
                lines.append(np.concatenate([mid, cj]))
                lines.append(np.concatenate([pj, cj]))
                continue
            if norder <= 1:
                offsets = [np.zeros(3)]
                r = bond_r
            else:
                perp = meshes.perpendicular(pj - pi)
                r = bond_r * (0.62 if norder == 2 else 0.5)
                spread = bond_r * (1.15 if norder == 2 else 1.3)
                if norder == 2:
                    offsets = [perp * spread, -perp * spread]
                else:
                    offsets = [np.zeros(3), perp * 2 * spread,
                               -perp * 2 * spread]
            for off in offsets:
                add_cyl(pi + off, mid + off, r, ci)
                add_cyl(mid + off, pj + off, r, cj)

        self._structure_cache = (
            np.array(spheres, np.float32).reshape(-1, 24),
            np.array(cyls, np.float32).reshape(-1, 24),
            np.array(lines, np.float32).reshape(-1, 6),
            np.array(halos, np.float32).reshape(-1, 24))
        return self._structure_cache

    # --------------------------------------------------- geometry instancing

    def item_instances(self, item):
        """(cylinder_instances, cone_instances) for a vector item."""
        cyls, cones = [], []
        if item.kind == "vector":
            ends = item.endpoints()
            if ends is not None:
                p0, p1 = ends
                d = p1 - p0
                length = np.linalg.norm(d)
                dhat = d / length
                tip_len = min(item.radius * 4.5, length * 0.3)
                pmid = p1 - dhat * tip_len
                color = [*item.color, item.alpha]
                m = meshes.bond_matrix(p0, pmid, item.radius)
                if m is not None:
                    cyls.append(np.concatenate([
                        m.T.flatten(), color,
                        [item.radius, item.radius,
                         np.linalg.norm(pmid - p0), 0.0]]))
                mc = meshes.bond_matrix(pmid, p1, item.radius * 2.4)
                if mc is not None:
                    cones.append(np.concatenate([
                        mc.T.flatten(), color,
                        [item.radius * 2.4, item.radius * 2.4,
                         tip_len, 0.0]]))
        elif item.kind == "plane":
            n = item.normal / np.linalg.norm(item.normal)
            x = meshes.perpendicular(n)
            y = np.cross(n, x)
            r = item.radius * item.size_scale
            m = np.eye(4, dtype=np.float32)
            m[:3, 0] = x * r
            m[:3, 1] = y * r
            m[:3, 2] = n
            m[:3, 3] = item.center
            cyls = []  # planes use the disc mesh; renderer handles this
            return [np.concatenate([m.T.flatten(),
                                    [*item.color, item.alpha],
                                    [r, r, 1.0, 0.0]])]
        return cyls, cones

    # -------------------------------------------------------------- picking

    def pick_atom(self, origin, direction):
        """CPU ray-sphere picking; returns atom index or None."""
        mol = self.molecule
        if mol is None:
            return None
        show_h = self.settings["show_hydrogens"]
        rep = self.settings["representation"]
        bond_r = self.settings["bond_radius"] / BOHR
        best, best_t = None, np.inf
        for i in range(mol.natoms):
            if not mol.visible[i]:
                continue
            z = mol.numbers[i]
            if not show_h and z == 1:
                continue
            if rep == "ball-and-stick":
                r = self.atom_radius(z)
            else:
                r = max(bond_r * 2.0, 0.6)
            oc = origin - mol.coords[i]
            b = np.dot(oc, direction)
            c = np.dot(oc, oc) - r * r
            disc = b * b - c
            if disc < 0:
                continue
            t = -b - np.sqrt(disc)
            if 0 < t < best_t:
                best, best_t = i, t
        return best
