# SPDX-License-Identifier: GPL-3.0-or-later
"""Scripted end-to-end GUI test: load -> visualize orbitals -> geometry ->
export.  Drives the real window through its own methods on GLib timeouts
and takes screenshots along the way."""

import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

import numpy as np  # noqa: E402

from kaijo.ui import mainwindow  # noqa: E402
from kaijo.ui import dialogs  # noqa: E402
from kaijo.core.pipeline import GridParams  # noqa: E402
from kaijo.render.scene import (FieldItem, OrbitalItem,  # noqa: E402
                                VectorItem)

import testdata  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
# A restricted, three-dimensional system: the geometry steps build a
# polyhedron (which degenerates on a planar molecule) and a restricted
# file keeps the spin-density button disabled (asserted in calc_fields).
FILE = testdata.path(testdata.GEOMETRY_FILE)

failures = []


def shot(win, name):
    win.queue_draw()
    while Gtk.events_pending():
        Gtk.main_iteration_do(False)
    xid = win.get_window().get_xid()
    subprocess.run(["import", "-window", str(xid),
                    os.path.join(OUT, name)], check=False)


class Driver:
    def __init__(self, app):
        self.app = app
        self.win = None
        self.step = 0

    def start(self):
        self.win = self.app.win
        GLib.timeout_add(500, self.poll_loaded)

    def poll_loaded(self):
        if self.win.data is None:
            return True
        GLib.timeout_add(400, self.visualize)
        return False

    def visualize(self):
        win = self.win
        mos = win.data.orbitals
        homo = mos.homo_index()
        # emulate the visualize flow without the modal dialog
        orbs = [homo - 1, homo, homo + 1, homo + 2]
        win._grid_params = GridParams("rectangular", 0.35, 4.0)
        win._current_isovalue = 0.03
        for orb in orbs:
            win.scene.add_item(OrbitalItem(
                orb, f"MO {mos.name(orb)}  E={mos.energies[orb]:.4f}"))
        win._resort_items()
        win.preview.sync_items()
        win._compute_surfaces(orbs)
        GLib.timeout_add(500, self.poll_surfaces)
        return False

    def poll_surfaces(self):
        items = self.win.scene.orbital_items()
        if not items or any(i.mesh_pos is None for i in items):
            return True
        # activate an orbital in the preview grid -> main view
        self.win._item_activated(items[1])
        self.win.preview.refresh_thumbnails(immediate=True)
        GLib.timeout_add(600, self.calc_fields)
        return False

    def calc_fields(self):
        # density + ESP through the Calculate flow (sans modal dialog);
        # both share one grid so the ESP re-uses the cached density
        win = self.win
        if win.panel.spin_btn.get_sensitive():
            failures.append("spin button sensitive for a restricted file")
        gp = GridParams("rectangular", 0.55, 4.0)
        win._field_grid_params["dens"] = gp
        win._field_grid_params["esp"] = gp
        win._current_field_iso["dens"] = 0.002
        win._current_field_iso["esp"] = 0.001
        win.scene.add_item(FieldItem("density", "Electron density"))
        win.scene.add_item(FieldItem("esp", "ESP"))
        win._resort_items()
        win.preview.sync_items()
        win._compute_fields([("density", gp, 0.002), ("esp", gp, 0.001)])
        GLib.timeout_add(500, self.poll_fields)
        return False

    def poll_fields(self):
        win = self.win
        dens = win.scene.field_item("density")
        esp = win.scene.field_item("esp")
        if dens.mesh_pos is None or esp.mesh_pos is None:
            return True
        if dens.mesh_pos.empty:
            failures.append("density surface is empty")
        if esp.esp_colors is None or \
                len(esp.esp_colors) != len(esp.mesh_pos.vertices):
            failures.append("ESP surface has no per-vertex colors")
        # properties panel: density shows editable colors, ESP does not
        win._item_activated(dens)
        if win.props_panel.item.kind != "density" or \
                win.props_panel._pos is None:
            failures.append("density properties panel wrong")
        win._item_activated(esp)
        if win.props_panel.item.kind != "esp" or \
                win.props_panel._pos is not None:
            failures.append("ESP properties panel should hide colors")
        GLib.timeout_add(500, self.shot_esp)
        return False

    def shot_esp(self):
        shot(self.win, "flow1c_esp.png")
        self.win._item_activated(self.win.scene.field_item("density"))
        GLib.timeout_add(500, self.shot_density)
        return False

    def shot_density(self):
        shot(self.win, "flow1d_density.png")
        GLib.timeout_add(400, self.after_surfaces)
        return False

    def after_surfaces(self):
        shot(self.win, "flow1_orbitals.png")
        # geometry: select some atoms around Dy, make polyhedron
        win = self.win
        mol = win.scene.molecule
        near = np.argsort(np.linalg.norm(
            mol.coords - mol.coords[0], axis=1))[1:9]
        win.scene.selected_atoms = [int(i) for i in near]
        win.panel.update_selection(win.scene.selected_atoms, mol)
        win._redraw_all()
        win._item_activated(win.scene.items[0])  # structure + selection
        shot(win, "flow1b_selection_glow.png")
        win._make_polyhedron()
        if win.scene.selected_atoms:
            failures.append("selection not cleared after polyhedron")
        # vector creation now prompts for length; build the item directly
        vec = VectorItem(mol.coords[0], mol.coords[1], "Vector test")
        win._add_item_command(vec, "Add vector")
        win._clear_selection()
        win.preview.refresh_thumbnails(immediate=True)
        GLib.timeout_add(500, self.after_geometry)
        return False

    def after_geometry(self):
        shot(self.win, "flow2_geometry.png")
        win = self.win
        # close a preview cell via its [x] handler, then undo the close
        orb_items = win.scene.orbital_items()
        n_before = len(win.scene.items)
        win._item_closed(orb_items[0])
        if len(win.scene.items) != n_before - 1:
            failures.append("preview close did not remove the item")
        win._undo()
        if len(win.scene.items) != n_before:
            failures.append("undo did not restore the closed item")
        # undo twice (vector, polyhedron), redo once
        n0 = len(win.scene.items)
        win._undo()
        win._undo()
        n1 = len(win.scene.items)
        win._redo()
        n2 = len(win.scene.items)
        if not (n1 == n0 - 2 and n2 == n0 - 1):
            failures.append(f"undo/redo counts wrong: {n0} {n1} {n2}")
        # export: orbitals + structure + polyhedron + density (the
        # density is recomputed at the export grid quality)
        for it in win.scene.items:
            if it.kind in ("orbital", "structure", "polyhedron",
                           "density"):
                it.export_selected = True
        opts = dict(win.settings["export"])
        opts.update(folder=OUT, basename="kaijo_export", scale=1.0,
                    transparent=True, compression=9, crop=True,
                    grid_spacing=0.35, bg_color=[1, 1, 1])
        items = [i for i in win.scene.items if i.export_selected]
        filenames = mainwindow.export_mod.make_filenames(
            items, win.data.orbitals, opts)
        job = mainwindow.export_mod.ExportJob(win, items, filenames, opts)
        win.export_job = job
        job.start()
        GLib.timeout_add(700, self.poll_export)
        return False

    def poll_export(self):
        if self.win.export_job is not None:
            return True
        files = sorted(f for f in os.listdir(OUT)
                       if f.startswith("kaijo_export"))
        print("exported files:", files)
        if len(files) < 5:
            failures.append(f"expected >=5 exported files, got {files}")
        shot(self.win, "flow3_after_export.png")
        # isovalue change through the surface-properties panel ->
        # re-extract from cached volumes (fast path)
        win = self.win
        win._item_activated(win.scene.orbital_items()[0])
        if win.props_panel.item is None or \
                win.props_panel.item.kind != "orbital":
            failures.append("properties panel not in surface mode")
        else:
            win.props_panel._iso.set_value(0.06)
            if abs(win.settings["isovalue"] - 0.06) > 1e-9:
                failures.append("panel isovalue did not reach settings")
        GLib.timeout_add(600, self.poll_iso)
        return False

    def poll_iso(self):
        items = self.win.scene.orbital_items()
        if any(i.mesh_pos is None for i in items):
            return True
        shot(self.win, "flow4_isovalue.png")
        win = self.win
        # visible items + camera angle sections
        win.axes_check.set_active(True)
        for label, direction in (("z", (0, 0, 1)),):
            win.camera.set_view_direction(direction)
        win._item_activated(win.scene.items[0])
        GLib.timeout_add(500, self.after_axes)
        return False

    def after_axes(self):
        shot(self.win, "flow5_axes_camz.png")
        win = self.win
        win.camera.set_view_direction((1, 0, 0))
        win.h_check.set_active(False)
        win.sym_check.set_active(True)   # labels: symbols + indices
        win.idx_check.set_active(True)
        spheres, _c, _l, _h = win.scene.structure_arrays()
        mol = win.scene.molecule
        n_heavy = int((mol.numbers[mol.visible] != 1).sum())
        if len(spheres) != n_heavy:
            failures.append(
                f"H hiding wrong: {len(spheres)} spheres, "
                f"{n_heavy} heavy atoms")
        GLib.timeout_add(500, self.after_h_hidden)
        return False

    def after_h_hidden(self):
        win = self.win
        shot(win, "flow6_noH_camx_labels.png")
        mol = win.scene.molecule
        n_heavy = int((mol.numbers[mol.visible] != 1).sum())
        texts = win.glview.renderer._labels_key[0]
        if len(texts) != n_heavy:
            failures.append(f"labels: {len(texts)} for {n_heavy} "
                            "heavy atoms")
        if texts and "(" not in texts[0]:
            failures.append(f"combined label format wrong: {texts[0]}")
        win.h_check.set_active(True)
        win.idx_check.set_active(False)  # symbols only, H back on
        win.camera.zoom(0.55)
        GLib.timeout_add(500, self.after_labels)
        return False

    def after_labels(self):
        win = self.win
        shot(win, "flow7_symbol_labels.png")
        texts = win.glview.renderer._labels_key[0]
        if len(texts) != int(win.scene.molecule.visible.sum()):
            failures.append("labels missing after re-enabling hydrogens")
        if any("(" in t or t.isdigit() for t in texts):
            failures.append("symbols-only labels contain indices")
        print("FAILURES:" if failures else "GUI FLOW OK",
              "; ".join(failures))
        self.app.quit()
        return False


def main():
    app = mainwindow.KaijoApp()
    app.set_start_file(FILE)
    driver = Driver(app)

    def on_activate(app_):
        GLib.timeout_add(300, driver.start)

    app.connect("activate", on_activate)
    app.run(None)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
