# SPDX-License-Identifier: GPL-3.0-or-later
"""Offscreen render test: structure + HOMO isosurface -> PNG."""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gtk  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

Gtk.init([])

win = Gdk.get_default_root_window()
ctx = win.create_gl_context()
ctx.set_required_version(3, 3)
ctx.realize()
ctx.make_current()

from kaijo.core.settings import Settings  # noqa: E402
from kaijo.core.grid import RectilinearGrid  # noqa: E402
from kaijo.core.basis import evaluate_orbitals  # noqa: E402
from kaijo.core import isosurface  # noqa: E402
from kaijo.formats import load_file  # noqa: E402
from kaijo.render.camera import Camera  # noqa: E402
from kaijo.render.renderer import Renderer, OffscreenTarget  # noqa: E402
from kaijo.render.scene import (Scene, OrbitalItem, VectorItem,  # noqa: E402
                                PlaneItem, PolyhedronItem)

import testdata  # noqa: E402

data = load_file(sys.argv[1] if len(sys.argv) > 1
                 else testdata.path(testdata.GEOMETRY_FILE))
mol, mos = data.molecule, data.orbitals

settings = Settings()
scene = Scene(settings)
scene.set_molecule(mol)

homo = mos.homo_index()
grid = RectilinearGrid.for_molecule(mol, spacing=0.35, margin=4.0)
vol = evaluate_orbitals(data.basis, mos.coeffs[[homo]], grid)[0]
pos, neg = isosurface.extract_pair(vol, 0.03, grid)
item = OrbitalItem(homo, mos.name(homo))
item.mesh_pos, item.mesh_neg = pos, neg
scene.add_item(item)

cam = Camera()
cam.fit(mol.center(), mol.extent())

r = Renderer()
tgt = OffscreenTarget(900, 700)

out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/kaijo_test.png"

tgt.bind()
r.render(scene, item, cam, 900, 700)
img = tgt.read_pixels()
Image.fromarray(img).save(out)
print("wrote", out)

# also: rotated view + vector + polyhedron to exercise geometry paths
cam.trackball(0.0, 0.0, 0.4, 0.3)
vec = VectorItem(mol.coords[0], mol.coords[1], "vec")
scene.add_item(vec)
tgt.bind()
r.render(scene, vec, cam, 900, 700)
Image.fromarray(tgt.read_pixels()).save(out.replace(".png", "_vec.png"))

sel = np.argsort(np.linalg.norm(mol.coords - mol.coords[0], axis=1))[1:9]
poly = PolyhedronItem(mol.coords[sel], "poly")
scene.add_item(poly)
tgt.bind()
r.render(scene, poly, cam, 900, 700)
Image.fromarray(tgt.read_pixels()).save(out.replace(".png", "_poly.png"))
print("wrote geometry test images")
