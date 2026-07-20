# SPDX-License-Identifier: GPL-3.0-or-later
"""Kaijo main window: assembles the 3D view, preview grid and options bar."""

import os
import threading

import numpy as np

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GLib, Gtk

from .. import formats
from ..formats.axis import AxisError, parse_axis_file
from ..core.pipeline import GridParams, OrbitalPipeline
from ..core.scratch import ScratchManager
from ..core.settings import Settings
from ..core.undo import FuncCommand, UndoStack
from ..core.pipeline import FIELD_LABELS
from ..render.camera import Camera
from ..render.scene import (FieldItem, OrbitalItem, PlaneItem,
                            PolyhedronItem, Scene, VectorItem)
from . import export as export_mod
from .cas import CasDialog
from .dialogs import (GridDialog, LengthDialog, OptionsDialog,
                      VectorDialog, show_about)
from .gl_view import GLView
from .orbital_panel import OrbitalPanel
from .preview import PreviewGrid
from .properties import PropertiesPanel


def _orbital_label(mos, orb):
    """Multi-line preview-cell caption: the per-spin file index and the
    symmetry species/index, then the energy and occupation."""
    return (f"MO {mos.name(orb)}   sym {mos.sym_name(orb)}\n"
            f"E {mos.energies[orb]:.3f}   occ {mos.occupations[orb]:.2f}")


class KaijoWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Kaijo")
        self.set_default_size(1500, 950)

        self.settings = Settings()
        self.scratch = ScratchManager()
        self.pipeline = OrbitalPipeline(self.scratch)
        self.undo_stack = UndoStack()
        self.undo_stack.on_change = self._update_undo_buttons

        self.data = None
        self.scene = Scene(self.settings)
        self.camera = Camera()
        self.active_item = None
        self.export_job = None
        self._compute_cancelled = False
        self._options_dialog = None
        self._grid_params = GridParams(
            self.settings["grid_type"], self.settings["grid_spacing"],
            self.settings["grid_margin"])
        self._current_isovalue = self.settings["isovalue"]
        # densities and the ESP each keep their own grid + isovalue
        self._field_grid_params = {
            "dens": GridParams(self.settings["dens_grid_type"],
                               self.settings["dens_grid_spacing"],
                               self.settings["dens_grid_margin"]),
            "esp": GridParams(self.settings["esp_grid_type"],
                              self.settings["esp_grid_spacing"],
                              self.settings["esp_grid_margin"]),
        }
        self._current_field_iso = {
            "dens": self.settings["dens_isovalue"],
            "esp": self.settings["esp_isovalue"],
        }
        # field jobs deferred until a running orbital re-extraction ends
        self._pending_field_jobs = []
        # axis file given on the command line, applied once loading is done
        self._axis_path = None

        self._build_ui()
        self.camera.on_change = self._camera_changed
        GLib.timeout_add_seconds(3, self._tick_scratch)
        self.connect("destroy", self._on_destroy)
        self.connect("key-press-event", self._on_key)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        header = Gtk.HeaderBar(title="Kaijo", show_close_button=True)
        self.set_titlebar(header)

        open_btn = Gtk.Button(label="Open")
        open_btn.set_tooltip_text("Open a molden, fchk or xyz file")
        open_btn.connect("clicked", self._open_clicked)
        header.pack_start(open_btn)

        self.undo_btn = Gtk.Button.new_from_icon_name(
            "edit-undo-symbolic", Gtk.IconSize.BUTTON)
        self.undo_btn.set_tooltip_text("Undo (Ctrl+Z)")
        self.undo_btn.connect("clicked", lambda b: self._undo())
        header.pack_start(self.undo_btn)
        self.redo_btn = Gtk.Button.new_from_icon_name(
            "edit-redo-symbolic", Gtk.IconSize.BUTTON)
        self.redo_btn.set_tooltip_text("Redo (Ctrl+Y)")
        self.redo_btn.connect("clicked", lambda b: self._redo())
        header.pack_start(self.redo_btn)

        fit_btn = Gtk.Button(label="Fit view")
        fit_btn.connect("clicked", lambda b: self._fit_view())
        header.pack_start(fit_btn)

        export_btn = Gtk.Button(label="Export images...")
        export_btn.get_style_context().add_class("suggested-action")
        export_btn.set_tooltip_text(
            "Export the preview-grid images ticked for selection")
        export_btn.connect("clicked", self._export_clicked)
        header.pack_end(export_btn)

        cas_btn = Gtk.Button(label="Orca CAS")
        cas_btn.set_tooltip_text(
            "Build the ORCA CASSCF active-space rotations from the orbitals "
            "selected in the preview grid")
        cas_btn.connect("clicked", self._cas_clicked)
        header.pack_end(cas_btn)

        opts_btn = Gtk.Button(label="Options")
        opts_btn.set_tooltip_text("Visualization options")
        opts_btn.connect("clicked", self._options_clicked)
        header.pack_end(opts_btn)

        about_btn = Gtk.Button(label="About")
        about_btn.set_tooltip_text("About Kaijo")
        about_btn.connect("clicked", self._about_clicked)
        header.pack_end(about_btn)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(root)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        root.pack_start(paned, True, True, 0)

        # left: options bar
        self.panel = OrbitalPanel()
        self.panel.set_size_request(290, -1)
        self.panel.on_visualize = self._visualize
        self.panel.on_calculate = self._calculate
        self.panel.on_make_vector = self._make_vector
        self.panel.on_make_plane = self._make_plane
        self.panel.on_make_polyhedron = self._make_polyhedron
        self.panel.on_set_bond = self._set_bond
        self.panel.on_clear_selection = self._clear_selection
        self.props_panel = PropertiesPanel(self.settings)
        self.props_panel.on_change = self._item_edited
        self.props_panel.on_surface_change = self.apply_options
        self.panel.pack_start(self.props_panel, False, False, 0)
        paned.pack1(self.panel, False, False)

        # centre + right
        paned2 = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.pack2(paned2, True, False)

        self.glview = GLView(self.scene, self.camera)
        self.glview.set_size_request(500, 400)
        self.glview.on_atom_clicked = self._atom_clicked
        self.glview.on_interaction_end = \
            lambda: self.preview.refresh_thumbnails()
        paned2.pack1(self.glview, True, False)

        vis_frame = Gtk.Frame(label="Visible items")
        vis_box = Gtk.Grid(column_spacing=10, row_spacing=2,
                           border_width=4)
        self._syncing_h = False
        self._syncing_multi = False
        self.h_check = Gtk.CheckButton(label="Hydrogen atoms")
        self.h_check.set_active(self.settings["show_hydrogens"])
        self.h_check.set_tooltip_text(
            "Show hydrogen atoms and their bonds in all views")
        self.h_check.connect("toggled", self._h_visibility_toggled)
        vis_box.attach(self.h_check, 0, 0, 1, 1)
        self.axes_check = Gtk.CheckButton(label="Cartesian axes")
        self.axes_check.set_tooltip_text(
            "Show the x/y/z axes at the origin (main view only)")
        self.axes_check.connect("toggled", self._axes_toggled)
        vis_box.attach(self.axes_check, 1, 0, 1, 1)
        self.sym_check = Gtk.CheckButton(label="Element symbols")
        self.sym_check.set_tooltip_text(
            "Label atoms with their element symbol (main view only)")
        self.sym_check.connect("toggled", self._labels_toggled)
        vis_box.attach(self.sym_check, 0, 1, 1, 1)
        self.idx_check = Gtk.CheckButton(label="Atom indices")
        self.idx_check.set_tooltip_text(
            "Label atoms with their number in the coordinate listing "
            "(main view only)")
        self.idx_check.connect("toggled", self._labels_toggled)
        vis_box.attach(self.idx_check, 1, 1, 1, 1)
        self.multi_check = Gtk.CheckButton(label="Draw double/triple bonds")
        self.multi_check.set_active(self.settings["multiple_bonds"])
        self.multi_check.set_tooltip_text(
            "Draw double and triple bonds as separate lines in all views; "
            "off draws every bond as a single line")
        self.multi_check.connect("toggled", self._multi_bonds_toggled)
        vis_box.attach(self.multi_check, 0, 2, 2, 1)
        vis_frame.add(vis_box)

        cam_frame = Gtk.Frame(label="Camera angle")
        cam_grid = Gtk.Grid(column_spacing=4, row_spacing=4,
                            border_width=4)
        cam_grid.set_column_homogeneous(True)
        views = (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1)),
                 ("x*", (-1, 0, 0)), ("y*", (0, -1, 0)),
                 ("z*", (0, 0, -1)))
        for n, (label, direction) in enumerate(views):
            btn = Gtk.Button(label=label)
            btn.set_tooltip_text(
                f"View along the {label[0]} axis in the "
                f"{'negative' if '*' in label else 'positive'} direction")
            btn.connect("clicked",
                        lambda b, d=direction:
                        self.camera.set_view_direction(d))
            cam_grid.attach(btn, n % 3, n // 3, 1, 1)
        cam_frame.add(cam_grid)

        # both sections live in the left bar, right below the orbital list
        self.panel.pack_start(vis_frame, False, False, 0)
        self.panel.pack_start(cam_frame, False, False, 0)
        self.panel.reorder_child(vis_frame, 2)
        self.panel.reorder_child(cam_frame, 3)

        self.preview = PreviewGrid(self.scene, self.glview, self.settings)
        self.preview.set_size_request(280, -1)
        self.preview.on_item_activated = self._item_activated
        self.preview.on_item_closed = self._item_closed
        paned2.pack2(self.preview, False, False)
        paned2.set_position(980)

        # bottom: status bar with progress + scratch usage
        status = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        status.set_border_width(3)
        self.status_label = Gtk.Label(label="Open a molden, fchk or xyz file "
                                            "to begin")
        self.status_label.set_xalign(0.0)
        status.pack_start(self.status_label, True, True, 4)
        self.progress = Gtk.ProgressBar()
        self.progress.set_size_request(220, -1)
        self.progress.set_no_show_all(True)
        status.pack_start(self.progress, False, False, 0)
        self.cancel_btn = Gtk.Button(label="Cancel")
        self.cancel_btn.set_no_show_all(True)
        self.cancel_btn.set_tooltip_text("Abort the running calculation")
        self.cancel_btn.connect("clicked", self._cancel_clicked)
        status.pack_start(self.cancel_btn, False, False, 0)
        self.scratch_label = Gtk.Label(label="scratch: 0 B")
        self.scratch_label.set_tooltip_text(
            "Scratch disk space used (deleted when Kaijo exits)")
        status.pack_end(self.scratch_label, False, False, 4)
        root.pack_start(status, False, False, 0)

        self.panel.set_orbitals(None)
        self._update_undo_buttons()

    # -------------------------------------------------------------- status

    def show_progress(self, fraction, message):
        self.progress.show()
        self.progress.set_fraction(min(max(fraction, 0.0), 1.0))
        self.status_label.set_text(message)
        return False

    def hide_progress(self, message=""):
        self.progress.hide()
        self.cancel_btn.hide()
        self.status_label.set_text(message)
        return False

    def _show_cancel(self):
        """Reveal the status-bar Cancel button for a cancelable task."""
        self.cancel_btn.set_sensitive(True)
        self.cancel_btn.show()

    def _cancel_clicked(self, *_a):
        """Abort whichever calculation is currently running."""
        self.cancel_btn.set_sensitive(False)
        if self.export_job is not None:
            self.export_job.cancelled = True
            self.status_label.set_text("Cancelling export...")
        elif self.pipeline.is_running():
            self._compute_cancelled = True
            self.status_label.set_text("Cancelling...")
            self.pipeline.request_cancel()
        else:
            self.cancel_btn.hide()

    def show_error(self, title, message):
        dlg = Gtk.MessageDialog(transient_for=self, modal=True,
                                message_type=Gtk.MessageType.ERROR,
                                buttons=Gtk.ButtonsType.OK, text=title)
        dlg.format_secondary_text(message)
        dlg.run()
        dlg.destroy()
        return False

    def show_warning(self, title, message):
        dlg = Gtk.MessageDialog(transient_for=self, modal=True,
                                message_type=Gtk.MessageType.WARNING,
                                buttons=Gtk.ButtonsType.OK, text=title)
        dlg.format_secondary_text(message)
        dlg.run()
        dlg.destroy()
        return False

    def show_info(self, title, message):
        dlg = Gtk.MessageDialog(transient_for=self, modal=True,
                                message_type=Gtk.MessageType.INFO,
                                buttons=Gtk.ButtonsType.OK, text=title)
        dlg.format_secondary_text(message)
        dlg.run()
        dlg.destroy()
        return False

    def update_scratch_label(self):
        n = self.scratch.usage_bytes()
        for unit in ("B", "kB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                break
            n /= 1024
        self.scratch_label.set_text(f"scratch: {n:.1f} {unit}"
                                    if unit != "B"
                                    else f"scratch: {int(n)} B")
        return False

    def _tick_scratch(self):
        self.update_scratch_label()
        return True

    # ------------------------------------------------------------- loading

    def open_path(self, path, axis_path=None):
        self._axis_path = axis_path
        self.show_progress(0.0, f"Loading {os.path.basename(path)}...")

        def worker():
            try:
                data = formats.load_file(
                    path, progress=lambda f, m="Loading...":
                    GLib.idle_add(self.show_progress, f, m))
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self.hide_progress, "Load failed")
                GLib.idle_add(self.show_error,
                              f"Could not load {os.path.basename(path)}",
                              str(exc))
                return
            GLib.idle_add(self._loaded, data)

        threading.Thread(target=worker, daemon=True).start()

    def _loaded(self, data):
        self.data = data
        self._pending_field_jobs = []
        self.pipeline.set_system(data)
        self.scene.set_molecule(data.molecule)
        self.active_item = self.scene.items[0]
        self.glview.current_item = None
        self.glview.invalidate_labels()
        if self.glview.get_realized() and self.glview.renderer:
            self.glview.make_current()
            self.glview.renderer.drop_all_surfaces()
        self.camera.fit(data.molecule.center(), data.molecule.extent())
        self.panel.set_orbitals(data.orbitals)
        self.panel.update_selection([], data.molecule)
        self.undo_stack = UndoStack()
        self.undo_stack.on_change = self._update_undo_buttons
        self._update_undo_buttons()
        self.preview.sync_items()
        self.preview.set_active_item(self.active_item)
        self.set_title(f"Kaijo – {data.molecule.name}")
        n = data.orbitals.nmo if data.has_orbitals else 0
        self.hide_progress(
            f"{data.molecule.natoms} atoms"
            + (f", {n} molecular orbitals" if n else " (no orbitals)"))
        self.preview.refresh_thumbnails()
        self.update_scratch_label()
        self._apply_axis_file()
        return False

    def _apply_axis_file(self):
        """Add the vector described by the axis file given on the command
        line.  Failures are non-critical: they are reported and the program
        carries on with the structure alone."""
        path, self._axis_path = self._axis_path, None
        if not path:
            return
        name = os.path.basename(path)
        try:
            spec = parse_axis_file(path)
            index, warning = spec.resolve_atom(self.scene.molecule)
        except AxisError as exc:
            self.show_error(f"Could not read {name}", str(exc))
            return
        vec = spec.vector / np.linalg.norm(spec.vector)
        center = self.scene.molecule.coords[index]
        item = VectorItem(center - vec / 2, center + vec / 2,
                          f"Vector {len(self.scene.items)}",
                          length=spec.length_bohr())
        # the structure stays selected; the vector only joins the grid
        self._add_item_command(item, "Add vector", activate=False)
        if warning:
            self.show_warning(f"{name}: ambiguous atom", warning)

    def _open_clicked(self, *_a):
        dlg = Gtk.FileChooserDialog(title="Open structure/orbital file",
                                    transient_for=self,
                                    action=Gtk.FileChooserAction.OPEN)
        dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL,
                        "Open", Gtk.ResponseType.OK)
        for name, patterns in formats.file_filters():
            filt = Gtk.FileFilter()
            filt.set_name(name)
            for p in patterns:
                filt.add_pattern(p)
            dlg.add_filter(filt)
        if dlg.run() == Gtk.ResponseType.OK:
            path = dlg.get_filename()
            dlg.destroy()
            self.open_path(path)
        else:
            dlg.destroy()

    # -------------------------------------------------------- camera/render

    def _h_visibility_toggled(self, btn):
        if self._syncing_h:
            return
        self.settings["show_hydrogens"] = btn.get_active()
        self._redraw_all()

    def _multi_bonds_toggled(self, btn):
        if self._syncing_multi:
            return
        self.settings["multiple_bonds"] = btn.get_active()
        self._redraw_all()

    def _axes_toggled(self, btn):
        self.glview.show_axes = btn.get_active()
        self.glview.queue_draw()

    def _labels_toggled(self, btn):
        self.glview.show_symbols = self.sym_check.get_active()
        self.glview.show_indices = self.idx_check.get_active()
        self.glview.invalidate_labels()
        self.glview.queue_draw()

    def _camera_changed(self):
        self.glview.queue_draw()
        self.preview.refresh_thumbnails()

    def _fit_view(self):
        if self.scene.molecule is not None:
            self.camera.fit(self.scene.molecule.center(),
                            self.scene.molecule.extent())

    def _redraw_all(self):
        self.scene.invalidate_structure()
        self.glview.invalidate_labels()
        self.glview.queue_draw()
        self.preview.refresh_thumbnails()

    # ------------------------------------------------------- preview/items

    def _item_activated(self, item):
        self.active_item = item
        self.scene.selected_item = item if item.kind != "structure" else None
        self.glview.current_item = item if item.kind != "structure" else None
        self.preview.set_active_item(item)
        self.props_panel.set_item(item)
        self.glview.queue_draw()
        self.status_label.set_text(item.describe())

    def _item_edited(self, item):
        """Live edit from the properties panel."""
        self.glview.queue_draw()
        self.preview.refresh_one(item)

    def _item_closed(self, item):
        """[x] pressed on a preview cell: remove the item (undoable)."""
        if item.kind == "structure":
            return
        if item.kind == "orbital":
            # keep the orbital list in sync so the next 'Visualize' does
            # not immediately re-add the closed orbital
            self.panel.unselect_orbital(item.orb_index)
        self._delete_item_command(item)

    def _resort_items(self):
        """Structure first, orbitals in display order, then geometry."""
        if self.data is None or not self.data.has_orbitals:
            return
        order = {int(o): r for r, o in
                 enumerate(self.data.orbitals.display_order())}
        structure = [i for i in self.scene.items if i.kind == "structure"]
        orbs = sorted((i for i in self.scene.items if i.kind == "orbital"),
                      key=lambda i: order.get(i.orb_index, 1 << 30))
        rest = [i for i in self.scene.items
                if i.kind not in ("structure", "orbital")]
        self.scene.items = structure + orbs + rest

    # ------------------------------------------------------- visualization

    def _visualize(self, orb_indices):
        if self.data is None or not self.data.has_orbitals:
            return
        dlg = GridDialog(self, self.settings)
        resp = dlg.run()
        vals = dlg.values()
        dlg.destroy()
        if resp != Gtk.ResponseType.OK:
            return
        grid_changed = (
            vals["grid_type"] != self._grid_params.kind
            or abs(vals["grid_spacing"] - self._grid_params.spacing) > 1e-9
            or abs(vals["grid_margin"] - self._grid_params.margin) > 1e-9)
        iso_changed = abs(vals["isovalue"] - self._current_isovalue) > 1e-12
        for k, v in vals.items():
            if k != "isovalue":
                self.settings[k] = v
        self.settings["isovalue"] = vals["isovalue"]
        self._current_isovalue = vals["isovalue"]
        self.props_panel.sync_surface()
        self._grid_params = GridParams(vals["grid_type"],
                                       vals["grid_spacing"],
                                       vals["grid_margin"])
        if grid_changed:
            self.scratch.clear_volumes()

        # diff the orbital items against the new selection
        current = {i.orb_index: i for i in self.scene.orbital_items()}
        wanted = set(orb_indices)
        for orb, item in list(current.items()):
            if orb not in wanted:
                self.scene.remove_item(item)
                self.glview.drop_item_surfaces(item.id)
                if self.active_item is item:
                    self._item_activated(self.scene.items[0])
        mos = self.data.orbitals
        for orb in wanted:
            if orb not in current:
                label = _orbital_label(mos, orb)
                self.scene.add_item(OrbitalItem(orb, label))
        self._resort_items()
        self.preview.sync_items()
        self.preview.refresh_thumbnails(immediate=True)

        todo = [i.orb_index for i in self.scene.orbital_items()
                if grid_changed or iso_changed or i.mesh_pos is None]
        if todo:
            self._compute_surfaces(todo)

    def _compute_surfaces(self, orb_indices):
        items = {i.orb_index: i for i in self.scene.orbital_items()}

        def on_progress(frac, msg):
            GLib.idle_add(self.show_progress, frac, msg)

        def on_surface(orb, pos, neg):
            GLib.idle_add(self._surface_ready, items.get(orb), pos, neg)

        def on_done(err):
            GLib.idle_add(self._surfaces_done, err)

        self.show_progress(0.0, "Starting isosurface computation...")
        self._compute_cancelled = False
        self._show_cancel()
        self.pipeline.compute_surfaces(
            orb_indices, self._grid_params, self._current_isovalue,
            on_progress, on_surface, on_done)

    def _surface_ready(self, item, pos, neg):
        if item is None or item not in self.scene.items:
            return False
        item.mesh_pos, item.mesh_neg = pos, neg
        self.preview.refresh_one(item)
        if self.active_item is item:
            self.glview.queue_draw()
        self.update_scratch_label()
        return False

    def _surfaces_done(self, err):
        cancelled = self._compute_cancelled
        if cancelled:
            self._compute_cancelled = False
            self._pending_field_jobs = []
            self.hide_progress("Isosurface computation cancelled")
        elif err is not None:
            self.hide_progress("Isosurface computation failed")
            self.show_error("Isosurface computation failed", str(err))
        else:
            self.hide_progress("Isosurfaces ready")
        self.update_scratch_label()
        if not cancelled and err is None and self._pending_field_jobs:
            jobs, self._pending_field_jobs = self._pending_field_jobs, []
            self._compute_fields(jobs)
        return False

    def _reextract_isosurfaces(self):
        """Isovalue changed: re-run marching cubes from cached volumes."""
        orbs = [i.orb_index for i in self.scene.orbital_items()]
        if orbs:
            self._compute_surfaces(orbs)

    # ------------------------------------------------- density / spin / ESP

    def _calculate(self, kind):
        """'Calculate' button pressed: kind is density | spin | esp."""
        if self.data is None or not self.data.has_orbitals:
            self.show_error("No orbitals loaded",
                            "Calculating a density or the ESP needs a "
                            "file with molecular orbitals (molden or fchk).")
            return
        if kind == "spin" and not self.data.orbitals.unrestricted:
            self.show_error("Spin density unavailable",
                            "The spin density (alpha minus beta) needs "
                            "unrestricted orbitals; this file is "
                            "restricted.")
            return
        pre = "esp" if kind == "esp" else "dens"
        titles = {"density": "Electron density grid",
                  "spin": "Spin density grid", "esp": "ESP grid"}
        dlg = GridDialog(
            self, self.settings, prefix=pre + "_", title=titles[kind],
            iso_lo=0.0001,
            iso_tip=("Electron-density isovalue of the surface the "
                     "potential is mapped onto" if kind == "esp" else
                     "Density isovalue (e/bohr^3)"))
        resp = dlg.run()
        vals = dlg.values()
        dlg.destroy()
        if resp != Gtk.ResponseType.OK:
            return
        for k in ("grid_type", "grid_spacing", "grid_margin", "isovalue"):
            self.settings[f"{pre}_{k}"] = vals[k]
        self._current_field_iso[pre] = vals["isovalue"]
        self.props_panel.sync_surface()
        gp = GridParams(vals["grid_type"], vals["grid_spacing"],
                        vals["grid_margin"])
        self._field_grid_params[pre] = gp

        item = self.scene.field_item(kind)
        if item is None:
            item = self.scene.add_item(FieldItem(kind, FIELD_LABELS[kind]))
            self._resort_items()
            self.preview.sync_items()
        self.preview.refresh_thumbnails(immediate=True)

        jobs = [(kind, gp, vals["isovalue"])]
        # density and spin density share their settings, so recompute the
        # sibling too if it is on display (cached volumes make this fast)
        if pre == "dens":
            other = "spin" if kind == "density" else "density"
            if self.scene.field_item(other) is not None:
                jobs.append((other, gp, vals["isovalue"]))
        self._compute_fields(jobs)

    def _compute_fields(self, jobs):
        def on_progress(frac, msg):
            GLib.idle_add(self.show_progress, frac, msg)

        def on_result(kind, payload):
            GLib.idle_add(self._field_ready, kind, payload)

        def on_done(err):
            GLib.idle_add(self._fields_done, err)

        self.show_progress(0.0, "Starting field computation...")
        self._compute_cancelled = False
        self._show_cancel()
        self.pipeline.compute_fields(jobs, on_progress, on_result,
                                     on_done)

    def _field_ready(self, kind, payload):
        item = self.scene.field_item(kind)
        if item is None:
            return False
        item.mesh_pos = payload.get("mesh_pos")
        item.mesh_neg = payload.get("mesh_neg")
        item.esp_colors = payload.get("esp_colors")
        item.esp_range = payload.get("esp_range", 0.0)
        cell = self.preview.cell_for(item)
        if cell is not None:
            cell.refresh_label()
        self.preview.refresh_one(item)
        if self.active_item is item:
            self.glview.queue_draw()
        self.update_scratch_label()
        return False

    def _fields_done(self, err):
        if self._compute_cancelled:
            self._compute_cancelled = False
            self._pending_field_jobs = []
            self.hide_progress("Calculation cancelled")
        elif err is not None:
            self._pending_field_jobs = []
            self.hide_progress("Calculation failed")
            self.show_error("Field calculation failed", str(err))
        else:
            self.hide_progress("Calculation finished")
        self.update_scratch_label()
        return False

    def _changed_field_jobs(self):
        """Jobs for on-display field items whose isovalue setting no
        longer matches the surfaces (volumes are cached, so these only
        re-run marching cubes and the ESP mapping)."""
        jobs = []
        for pre, kinds in (("dens", ("density", "spin")),
                           ("esp", ("esp",))):
            iso = self.settings[f"{pre}_isovalue"]
            if abs(iso - self._current_field_iso[pre]) <= 1e-12:
                continue
            self._current_field_iso[pre] = iso
            for kind in kinds:
                if self.scene.field_item(kind) is not None:
                    jobs.append((kind, self._field_grid_params[pre], iso))
        return jobs

    def _sync_panel_toggles(self):
        """Reflect the settings in the main-panel visibility toggles
        without re-triggering their handlers (used after the options
        dialog changes or resets the settings)."""
        self._syncing_h = True
        self.h_check.set_active(self.settings["show_hydrogens"])
        self._syncing_h = False
        self._syncing_multi = True
        self.multi_check.set_active(self.settings["multiple_bonds"])
        self._syncing_multi = False

    def apply_options(self):
        """Called by the options dialog / properties panel on any change."""
        self._sync_panel_toggles()
        self.props_panel.sync_surface()
        self._recompute_changed_surfaces()
        self.settings.save()
        self._redraw_all()

    def _recompute_changed_surfaces(self):
        """Re-extract every surface whose isovalue setting changed.  The
        pipeline runs one job at a time, so field jobs are deferred
        until a triggered orbital re-extraction finishes."""
        iso = self.settings["isovalue"]
        orb_changed = abs(iso - self._current_isovalue) > 1e-12
        if orb_changed:
            self._current_isovalue = iso
        field_jobs = self._changed_field_jobs()
        if orb_changed and self.scene.orbital_items():
            self._pending_field_jobs = field_jobs
            self._reextract_isosurfaces()
        elif field_jobs:
            self._compute_fields(field_jobs)

    def _about_clicked(self, *_a):
        show_about(self)

    def _options_clicked(self, *_a):
        if self._options_dialog is not None:
            self._options_dialog.present()
            return
        dlg = OptionsDialog(self, self.settings, self.scene.molecule)
        dlg.on_apply = self.apply_options

        def on_response(d, resp):
            if resp == Gtk.ResponseType.REJECT:
                self.settings.reset()
                d.destroy()
                self._options_dialog = None
                self._sync_panel_toggles()
                self.props_panel.sync_surface()
                self._redraw_all()
                self._recompute_changed_surfaces()
            else:
                self.settings.save()
                d.destroy()
                self._options_dialog = None

        dlg.connect("response", on_response)
        self._options_dialog = dlg

    # ---------------------------------------------------------- selection

    def _atom_clicked(self, atom):
        if atom is None:
            return
        sel = self.scene.selected_atoms
        if atom in sel:
            sel.remove(atom)
        else:
            sel.append(atom)
        self.panel.update_selection(sel, self.scene.molecule)
        self._redraw_all()

    def _clear_selection(self):
        if self.scene.selected_atoms:
            self.scene.selected_atoms = []
            self.panel.update_selection([], self.scene.molecule)
            self._redraw_all()

    # ------------------------------------------------------------ geometry

    def _add_item_command(self, item, label, activate=True):
        scene = self.scene

        def do():
            if item not in scene.items:
                scene.add_item(item)
            self._resort_items()
            self.preview.sync_items()
            self.preview.refresh_one(item)
            if activate:
                self._item_activated(item)
            else:
                self.preview.set_active_item(self.active_item)

        def undo():
            scene.remove_item(item)
            self.glview.drop_item_surfaces(item.id)
            if self.active_item is item:
                self._item_activated(scene.items[0])
            self.preview.sync_items()

        self.undo_stack.push(FuncCommand(label, do, undo))

    def _make_vector(self):
        sel = self.scene.selected_atoms
        mol = self.scene.molecule
        if len(sel) == 2:
            p0, p1 = mol.coords[sel[0]], mol.coords[sel[1]]
            dlg = LengthDialog(
                self, "Vector length", "Vector length (bohr):",
                VectorItem.default_length(p0, p1))
            resp = dlg.run()
            length = dlg.value()
            dlg.destroy()
            if resp != Gtk.ResponseType.OK:
                return
            item = VectorItem(p0, p1, f"Vector {len(self.scene.items)}",
                              length=length)
        elif len(sel) == 1:
            dlg = VectorDialog(self)
            resp = dlg.run()
            vec = np.array(dlg.vector())
            length = dlg.length()
            dlg.destroy()
            if resp != Gtk.ResponseType.OK or np.linalg.norm(vec) < 1e-9:
                return
            # no length given -> use the magnitude of the components;
            # otherwise normalise the direction to the given length
            if length is None:
                length = float(np.linalg.norm(vec))
            p0 = mol.coords[sel[0]]
            item = VectorItem(p0 - vec / 2, p0 + vec / 2,
                              f"Vector {len(self.scene.items)}",
                              length=length)
        else:
            return
        self._add_item_command(item, "Add vector")
        self._clear_selection()

    def _make_plane(self):
        sel = self.scene.selected_atoms
        if len(sel) < 3:
            return
        pts = self.scene.molecule.coords[sel]
        item = PlaneItem.fit(pts, f"Plane {len(self.scene.items)}")
        dlg = LengthDialog(self, "Plane size", "Plane width (bohr):",
                           2.0 * item.radius)
        resp = dlg.run()
        width = dlg.value()
        dlg.destroy()
        if resp != Gtk.ResponseType.OK:
            return
        item.radius = 0.5 * width
        self._add_item_command(item, "Add plane")
        self._clear_selection()

    def _make_polyhedron(self):
        sel = self.scene.selected_atoms
        if len(sel) < 4:
            return
        pts = self.scene.molecule.coords[sel]
        try:
            item = PolyhedronItem(pts, f"Polyhedron "
                                       f"{len(self.scene.items)}")
        except Exception as exc:  # degenerate geometry
            self.show_error("Cannot build polyhedron", str(exc))
            return
        self._add_item_command(item, "Add polyhedron")
        self._clear_selection()

    def _set_bond(self, order):
        sel = self.scene.selected_atoms
        if len(sel) != 2:
            return
        mol = self.scene.molecule
        i, j = sel

        state = {}

        def do():
            state["prev"] = mol.set_bond_override(i, j, order)
            self._redraw_all()

        def undo():
            mol.restore_bond_override(i, j, state["prev"])
            self._redraw_all()

        self.undo_stack.push(FuncCommand("Set bond type", do, undo))

    # ------------------------------------------------------ delete / undo

    def _delete_selected(self):
        sel = list(self.scene.selected_atoms)
        if sel:
            mol = self.scene.molecule

            def do():
                mol.hide_atoms(sel)
                self.scene.selected_atoms = []
                self.panel.update_selection([], mol)
                self._redraw_all()

            def undo():
                mol.show_atoms(sel)
                self.scene.selected_atoms = list(sel)
                self.panel.update_selection(sel, mol)
                self._redraw_all()

            self.undo_stack.push(FuncCommand("Delete atoms", do, undo))
            return
        item = self.active_item
        if item is not None and item.kind in ("vector", "plane",
                                              "polyhedron"):
            self._delete_item_command(item)

    def _delete_item_command(self, item):
        scene = self.scene
        pos = scene.items.index(item)

        def do():
            scene.remove_item(item)
            self.glview.drop_item_surfaces(item.id)
            if self.active_item is item:
                self._item_activated(scene.items[0])
            self.preview.sync_items()

        def undo():
            scene.items.insert(pos, item)
            self.preview.sync_items()
            self.preview.refresh_one(item)

        self.undo_stack.push(FuncCommand("Delete item", do, undo))

    def _undo(self):
        self.undo_stack.undo()

    def _redo(self):
        self.undo_stack.redo()

    def _update_undo_buttons(self):
        self.undo_btn.set_sensitive(self.undo_stack.can_undo)
        self.redo_btn.set_sensitive(self.undo_stack.can_redo)

    def _on_key(self, widget, event):
        key = event.keyval
        ctrl = event.state & Gdk.ModifierType.CONTROL_MASK
        if ctrl and key in (Gdk.KEY_z, Gdk.KEY_Z):
            self._undo()
            return True
        if ctrl and key in (Gdk.KEY_y, Gdk.KEY_Y):
            self._redo()
            return True
        if key == Gdk.KEY_Escape:
            self._clear_selection()
            return True
        if key == Gdk.KEY_Delete:
            self._delete_selected()
            return True
        if ctrl and key in (Gdk.KEY_o, Gdk.KEY_O):
            self._open_clicked()
            return True
        if ctrl and key in (Gdk.KEY_e, Gdk.KEY_E):
            self._export_clicked()
            return True
        return False

    # -------------------------------------------------------------- export

    def _cas_clicked(self, *_a):
        if self.data is None or not self.data.has_orbitals:
            self.show_info(
                "No orbitals loaded",
                "Load a file with molecular orbitals before building an "
                "ORCA active space.")
            return
        items = [i for i in self.scene.items if i.export_selected]
        if not items:
            self.show_info(
                "No orbitals selected",
                "Tick the 'select' box of the orbitals in the preview grid "
                "that should form the CASSCF active space, then press "
                "'Orca CAS' again.")
            return
        if any(i.kind != "orbital" for i in items):
            self.show_error(
                "Only orbitals allowed",
                "The ORCA CAS tool works on molecular orbitals only. "
                "Deselect any densities, ESP maps, the structure and "
                "geometric objects in the preview grid.")
            return
        orbitals = self.data.orbitals
        orb_indices = [i.orb_index for i in items]
        if orbitals.unrestricted and any(
                orbitals.spins[i] == 1 for i in orb_indices):
            self.show_error(
                "Alpha orbitals only",
                "ORCA builds the CASSCF initial guess from the alpha "
                "orbitals only, so beta orbitals cannot be used to define "
                "the active space. Select alpha orbitals only.")
            return
        dlg = CasDialog(self, orbitals, orb_indices)
        dlg.run()
        dlg.destroy()

    def _export_clicked(self, *_a):
        if self.export_job is not None:
            self.show_error("Export in progress",
                            "Wait for the current export to finish.")
            return
        items = [i for i in self.scene.items if i.export_selected]
        if not items:
            self.show_error(
                "Nothing selected for export",
                "Tick the 'export' box of the preview images you want "
                "to export first.")
            return
        alloc = self.glview.get_allocation()
        dlg = export_mod.ExportDialog(self, self.settings,
                                      (alloc.width, alloc.height),
                                      len(items))
        resp = dlg.run()
        opts = dlg.values()
        dlg.destroy()
        if resp != Gtk.ResponseType.OK:
            return
        self.settings["export"] = opts
        self.settings.save()

        orbitals = self.data.orbitals if self.data else None
        filenames = export_mod.make_filenames(items, orbitals, opts)
        skip = export_mod.confirm_overwrites(self, opts["folder"],
                                             filenames)
        if skip is None:
            return
        items = [i for i in items if i.id not in skip]
        if not items:
            return
        self.export_job = export_mod.ExportJob(self, items, filenames, opts)
        self.export_job.start()
        self._show_cancel()

    def export_finished(self, result):
        self.export_job = None
        if isinstance(result, Exception):
            self.hide_progress("Export failed")
            self.show_error("Export failed", str(result))
        elif result:  # cancelled
            self.hide_progress("Export cancelled")
        else:
            self.hide_progress("Export finished")
        self.update_scratch_label()
        return False

    # ---------------------------------------------------------------- exit

    def _on_destroy(self, *_a):
        self.pipeline.cancel()
        self.settings.save()
        self.scratch.cleanup()


class KaijoApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.kaijo.Kaijo",
                         flags=0)
        self.win = None
        self._open_on_start = None
        self._axis_on_start = None

    def set_start_file(self, path, axis_path=None):
        self._open_on_start = path
        self._axis_on_start = axis_path

    def do_activate(self):
        if self.win is None:
            self.win = KaijoWindow(self)
            self.win.show_all()
            if self._open_on_start:
                self.win.open_path(self._open_on_start,
                                   axis_path=self._axis_on_start)
        self.win.present()
