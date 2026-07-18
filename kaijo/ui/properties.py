# SPDX-License-Identifier: GPL-3.0-or-later
"""Embedded properties panel: geometric objects and isosurfaces.

Lives in the options bar below the selection/geometry tools.  With a
geometric object active it edits that object; with an orbital active it
edits the (shared) isosurface settings — isovalue, lobe colors, opacity —
and every isosurface updates automatically.  Otherwise a placeholder.
"""

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

from .dialogs import _color_button, _get_rgb, _spin

# Per-kind surface-settings map: which settings keys the controls edit,
# and whether the colors are user-editable (the ESP gradient is fixed).
_DENSITY_CFG = dict(
    title="Density isosurfaces",
    iso="dens_isovalue", pos="dens_color_pos", neg="dens_color_neg",
    alpha="dens_alpha", iso_lo=0.0001,
    note="Shared by the electron density and the spin density.")
_SURFACE_CFG = {
    "orbital": dict(
        title="Isosurfaces (all orbitals)",
        iso="isovalue", pos="iso_color_pos", neg="iso_color_neg",
        alpha="iso_alpha", iso_lo=0.0005, note=None),
    "density": _DENSITY_CFG,
    "spin": _DENSITY_CFG,
    "esp": dict(
        title="ESP surface",
        iso="esp_isovalue", pos=None, neg=None,
        alpha="esp_alpha", iso_lo=0.0001,
        note="The isovalue is the electron-density isovalue of the "
             "surface the potential is mapped onto.  The color scale "
             "is fixed: red = negative, white = zero, blue = positive "
             "potential."),
}


class PropertiesPanel(Gtk.Frame):
    def __init__(self, settings):
        super().__init__(label="Object / surface properties")
        self.settings = settings
        self.item = None
        self.on_change = None          # callback(item): geometry edited
        self.on_surface_change = None  # callback(): isosurface settings
        self._building = False
        self._box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,
                            spacing=4)
        self._box.set_border_width(6)
        self.add(self._box)
        self._show_placeholder()

    def _clear(self):
        for child in self._box.get_children():
            self._box.remove(child)

    def _show_placeholder(self):
        self._clear()
        lab = Gtk.Label(label="Select an orbital or geometric object")
        lab.get_style_context().add_class("dim-label")
        lab.set_xalign(0.0)
        lab.set_line_wrap(True)
        self._box.pack_start(lab, False, False, 8)
        self._box.show_all()

    def set_item(self, item):
        """item: any scene item or None; orbitals and computed fields
        show their (shared) isosurface settings, vectors/planes/
        polyhedra their own properties."""
        if item is not None and item.kind not in (
                "vector", "plane", "polyhedron") \
                and item.kind not in _SURFACE_CFG:
            item = None
        self.item = item
        if item is None:
            self._show_placeholder()
            return
        if item.kind in _SURFACE_CFG:
            self._build_surface_controls(item.kind)
            return
        self._clear()
        title = Gtk.Label()
        title.set_markup(f"<b>{item.describe()}</b>")
        title.set_xalign(0.0)
        self._box.pack_start(title, False, False, 0)

        grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        row = 0
        grid.attach(Gtk.Label(label="Color:", xalign=1.0), 0, row, 1, 1)
        self._color = _color_button(item.color)
        self._color.connect("color-set", self._apply)
        grid.attach(self._color, 1, row, 1, 1)

        row += 1
        grid.attach(Gtk.Label(label="Opacity:", xalign=1.0), 0, row, 1, 1)
        self._alpha = _spin(item.alpha, 0.05, 1.0, 0.05, 2)
        self._alpha.connect("value-changed", self._apply)
        grid.attach(self._alpha, 1, row, 1, 1)

        self._length = self._radius = self._width = None
        if item.kind == "vector":
            row += 1
            grid.attach(Gtk.Label(label="Length (bohr):", xalign=1.0),
                        0, row, 1, 1)
            self._length = _spin(item.length, 0.1, 500.0, 0.5, 2)
            self._length.connect("value-changed", self._apply)
            grid.attach(self._length, 1, row, 1, 1)
            row += 1
            grid.attach(Gtk.Label(label="Radius (bohr):", xalign=1.0),
                        0, row, 1, 1)
            self._radius = _spin(item.radius, 0.02, 1.0, 0.02)
            self._radius.connect("value-changed", self._apply)
            grid.attach(self._radius, 1, row, 1, 1)
        elif item.kind == "plane":
            row += 1
            grid.attach(Gtk.Label(label="Width (bohr):", xalign=1.0),
                        0, row, 1, 1)
            self._width = _spin(2.0 * item.radius * item.size_scale,
                                0.5, 500.0, 0.5, 2)
            self._width.connect("value-changed", self._apply)
            grid.attach(self._width, 1, row, 1, 1)

        self._box.pack_start(grid, False, False, 0)
        self._box.show_all()

    # ------------------------------------------------- isosurface controls

    def _build_surface_controls(self, kind):
        cfg = self._surface_cfg = _SURFACE_CFG[kind]
        self._clear()
        self._building = True
        title = Gtk.Label()
        title.set_markup(f"<b>{cfg['title']}</b>")
        title.set_xalign(0.0)
        self._box.pack_start(title, False, False, 0)

        grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        s = self.settings
        row = 0
        grid.attach(Gtk.Label(label="Isovalue:", xalign=1.0), 0, row, 1, 1)
        self._iso = _spin(s[cfg["iso"]], cfg["iso_lo"], 1.0, 0.005, 4)
        self._iso.connect("value-changed", self._apply_surface)
        grid.attach(self._iso, 1, row, 1, 1)
        self._pos = self._neg = None
        if cfg["pos"]:
            row += 1
            grid.attach(Gtk.Label(label="Positive lobe:", xalign=1.0),
                        0, row, 1, 1)
            self._pos = _color_button(s[cfg["pos"]])
            self._pos.connect("color-set", self._apply_surface)
            grid.attach(self._pos, 1, row, 1, 1)
            row += 1
            grid.attach(Gtk.Label(label="Negative lobe:", xalign=1.0),
                        0, row, 1, 1)
            self._neg = _color_button(s[cfg["neg"]])
            self._neg.connect("color-set", self._apply_surface)
            grid.attach(self._neg, 1, row, 1, 1)
        row += 1
        grid.attach(Gtk.Label(label="Opacity:", xalign=1.0), 0, row, 1, 1)
        self._surf_alpha = _spin(s[cfg["alpha"]], 0.05, 1.0, 0.05, 2)
        self._surf_alpha.connect("value-changed", self._apply_surface)
        grid.attach(self._surf_alpha, 1, row, 1, 1)
        self._box.pack_start(grid, False, False, 0)

        if cfg["note"]:
            note = Gtk.Label(label=cfg["note"])
            note.get_style_context().add_class("dim-label")
            note.set_xalign(0.0)
            note.set_line_wrap(True)
            note.set_max_width_chars(32)
            self._box.pack_start(note, False, False, 2)
        self._box.show_all()
        self._building = False

    def _apply_surface(self, *_a):
        if self._building:
            return
        cfg = self._surface_cfg
        s = self.settings
        s[cfg["iso"]] = self._iso.get_value()
        if self._pos is not None:
            s[cfg["pos"]] = _get_rgb(self._pos)
            s[cfg["neg"]] = _get_rgb(self._neg)
        s[cfg["alpha"]] = self._surf_alpha.get_value()
        if self.on_surface_change:
            self.on_surface_change()

    def sync_surface(self):
        """Refresh the surface widgets after settings changed elsewhere
        (e.g. the options dialog or the visualize/calculate prompts)."""
        if self.item is None or self.item.kind not in _SURFACE_CFG:
            return
        cfg = self._surface_cfg
        self._building = True
        s = self.settings
        self._iso.set_value(s[cfg["iso"]])
        self._surf_alpha.set_value(s[cfg["alpha"]])
        if self._pos is not None:
            from gi.repository import Gdk
            self._pos.set_rgba(Gdk.RGBA(*s[cfg["pos"]], 1.0))
            self._neg.set_rgba(Gdk.RGBA(*s[cfg["neg"]], 1.0))
        self._building = False

    # --------------------------------------------------- geometry controls

    def _apply(self, *_a):
        item = self.item
        if item is None or item.kind in _SURFACE_CFG:
            return
        item.color = _get_rgb(self._color)
        item.alpha = self._alpha.get_value()
        if item.kind == "vector":
            item.length = self._length.get_value()
            item.radius = self._radius.get_value()
        elif item.kind == "plane":
            item.radius = 0.5 * self._width.get_value()
            item.size_scale = 1.0
        if self.on_change:
            self.on_change(item)
