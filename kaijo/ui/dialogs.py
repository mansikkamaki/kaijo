# SPDX-License-Identifier: GPL-3.0-or-later
"""Dialogs: grid/isovalue prompt, visualization options, vector components,
geometry item properties."""

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gdk, Gtk

from ..core import elements
from ..core.grid import GRID_TYPES


def _spin(value, lo, hi, step, digits=3):
    adj = Gtk.Adjustment(value=value, lower=lo, upper=hi,
                         step_increment=step, page_increment=step * 10)
    sp = Gtk.SpinButton(adjustment=adj, digits=digits)
    sp.set_numeric(True)
    return sp


def _color_button(rgb):
    c = Gdk.RGBA(*rgb, 1.0)
    btn = Gtk.ColorButton()
    btn.set_rgba(c)
    return btn


def _get_rgb(btn):
    c = btn.get_rgba()
    return [c.red, c.green, c.blue]


class GridDialog(Gtk.Dialog):
    """Prompt for grid specification and isovalue before visualization.

    `prefix` selects which settings family the defaults come from:
    "" for the orbital isosurfaces, "dens_" for the densities and
    "esp_" for the ESP (keys like "dens_grid_spacing", "dens_isovalue").
    """

    def __init__(self, parent, settings, npoints_hint=None, prefix="",
                 title="Isosurface grid", iso_lo=0.0005, iso_tip=None):
        super().__init__(title=title, transient_for=parent,
                         modal=True)
        self.add_buttons("Cancel", Gtk.ResponseType.CANCEL,
                         "Compute", Gtk.ResponseType.OK)
        self.set_default_response(Gtk.ResponseType.OK)
        grid = Gtk.Grid(column_spacing=8, row_spacing=6, border_width=12)

        row = 0
        grid.attach(Gtk.Label(label="Grid type:", xalign=1.0), 0, row, 1, 1)
        self.kind_combo = Gtk.ComboBoxText()
        for kind in GRID_TYPES:
            self.kind_combo.append_text(kind)
        self.kind_combo.set_active(0)
        grid.attach(self.kind_combo, 1, row, 1, 1)

        row += 1
        grid.attach(Gtk.Label(label="Point spacing (bohr):", xalign=1.0),
                    0, row, 1, 1)
        self.spacing = _spin(settings[prefix + "grid_spacing"],
                             0.05, 2.0, 0.05)
        self.spacing.set_tooltip_text(
            "Distance between grid points; smaller is finer but slower. "
            "0.3-0.4 is good for previews.")
        grid.attach(self.spacing, 1, row, 1, 1)

        row += 1
        grid.attach(Gtk.Label(label="Box margin (bohr):", xalign=1.0),
                    0, row, 1, 1)
        self.margin = _spin(settings[prefix + "grid_margin"],
                            1.0, 12.0, 0.5, 1)
        self.margin.set_tooltip_text(
            "Extra space around the molecule included in the grid")
        grid.attach(self.margin, 1, row, 1, 1)

        row += 1
        grid.attach(Gtk.Label(label="Isovalue:", xalign=1.0), 0, row, 1, 1)
        self.isovalue = _spin(settings[prefix + "isovalue"],
                              iso_lo, 1.0, 0.005, 4)
        if iso_tip:
            self.isovalue.set_tooltip_text(iso_tip)
        grid.attach(self.isovalue, 1, row, 1, 1)

        if npoints_hint:
            row += 1
            self.hint = Gtk.Label()
            self.hint.set_markup(f"<small>{npoints_hint}</small>")
            grid.attach(self.hint, 0, row, 2, 1)

        self.get_content_area().add(grid)
        self.show_all()

    def values(self):
        return {
            "grid_type": self.kind_combo.get_active_text(),
            "grid_spacing": self.spacing.get_value(),
            "grid_margin": self.margin.get_value(),
            "isovalue": self.isovalue.get_value(),
        }


class OptionsDialog(Gtk.Dialog):
    """Visualization options window (applies live, persists on close)."""

    def __init__(self, parent, settings, molecule=None):
        super().__init__(title="Visualization options", transient_for=parent,
                         modal=False)
        self.settings = settings
        self.on_apply = None
        self.on_reset = None
        self.add_buttons("Reset to defaults", Gtk.ResponseType.REJECT,
                         "Close", Gtk.ResponseType.CLOSE)
        nb = Gtk.Notebook()
        nb.set_border_width(8)

        # ---- structure page ------------------------------------------
        g = Gtk.Grid(column_spacing=8, row_spacing=6, border_width=10)
        row = 0
        g.attach(Gtk.Label(label="Representation:", xalign=1.0), 0, row, 1, 1)
        self.rep_combo = Gtk.ComboBoxText()
        for rep in ("ball-and-stick", "sticks", "wireframe"):
            self.rep_combo.append_text(rep)
        self.rep_combo.set_active(
            ("ball-and-stick", "sticks", "wireframe").index(
                settings["representation"]))
        g.attach(self.rep_combo, 1, row, 1, 1)
        row += 1
        g.attach(Gtk.Label(label="Atom size scale:", xalign=1.0), 0, row, 1, 1)
        self.atom_scale = _spin(settings["atom_scale"], 0.2, 3.0, 0.1, 2)
        g.attach(self.atom_scale, 1, row, 1, 1)
        row += 1
        g.attach(Gtk.Label(label="Bond radius (Å):", xalign=1.0), 0, row, 1, 1)
        self.bond_radius = _spin(settings["bond_radius"], 0.02, 0.5, 0.01)
        g.attach(self.bond_radius, 1, row, 1, 1)
        row += 1
        g.attach(Gtk.Label(label="Background:", xalign=1.0), 0, row, 1, 1)
        self.bg_btn = _color_button(settings["background"])
        g.attach(self.bg_btn, 1, row, 1, 1)
        nb.append_page(g, Gtk.Label(label="Structure"))

        # ---- per-element page ----------------------------------------
        g2 = Gtk.Grid(column_spacing=8, row_spacing=4, border_width=10)
        self.elem_widgets = {}
        zs = sorted(set(molecule.numbers.tolist())) if molecule is not None \
            else []
        g2.attach(Gtk.Label(label="Element"), 0, 0, 1, 1)
        g2.attach(Gtk.Label(label="Color"), 1, 0, 1, 1)
        g2.attach(Gtk.Label(label="Radius (Å)"), 2, 0, 1, 1)
        for row, z in enumerate(zs, start=1):
            g2.attach(Gtk.Label(label=elements.z_to_symbol(z), xalign=0.0),
                      0, row, 1, 1)
            over_c = settings["atom_colors"].get(str(z))
            cbtn = _color_button(over_c if over_c
                                 else elements.element_color(z))
            g2.attach(cbtn, 1, row, 1, 1)
            over_r = settings["atom_radii"].get(str(z))
            rsp = _spin(over_r if over_r else elements.display_radius(z),
                        0.05, 3.0, 0.05)
            g2.attach(rsp, 2, row, 1, 1)
            self.elem_widgets[z] = (cbtn, rsp)
        sw = Gtk.ScrolledWindow()
        sw.set_min_content_height(240)
        sw.add(g2)
        nb.append_page(sw, Gtk.Label(label="Elements"))

        self.get_content_area().add(nb)
        self.rep_combo.connect("changed", self._apply)
        for w in (self.atom_scale, self.bond_radius):
            w.connect("value-changed", self._apply)
        self.bg_btn.connect("color-set", self._apply)
        for cbtn, rsp in self.elem_widgets.values():
            cbtn.connect("color-set", self._apply)
            rsp.connect("value-changed", self._apply)
        self.show_all()

    def _apply(self, *_a):
        s = self.settings
        s["representation"] = self.rep_combo.get_active_text()
        s["atom_scale"] = self.atom_scale.get_value()
        s["bond_radius"] = self.bond_radius.get_value()
        s["background"] = _get_rgb(self.bg_btn)
        for z, (cbtn, rsp) in self.elem_widgets.items():
            c = _get_rgb(cbtn)
            if [round(x, 3) for x in c] != \
                    [round(x, 3) for x in elements.element_color(z)]:
                s["atom_colors"][str(z)] = c
            r = rsp.get_value()
            if abs(r - elements.display_radius(z)) > 1e-3:
                s["atom_radii"][str(z)] = r
        if self.on_apply:
            self.on_apply()


class LengthDialog(Gtk.Dialog):
    """Generic prompt for a single length/size value (bohr)."""

    def __init__(self, parent, title, label, default, lo=0.1, hi=500.0):
        super().__init__(title=title, transient_for=parent, modal=True)
        self.add_buttons("Cancel", Gtk.ResponseType.CANCEL,
                         "OK", Gtk.ResponseType.OK)
        self.set_default_response(Gtk.ResponseType.OK)
        g = Gtk.Grid(column_spacing=8, row_spacing=6, border_width=12)
        g.attach(Gtk.Label(label=label, xalign=1.0), 0, 0, 1, 1)
        self.value_spin = _spin(default, lo, hi, 0.5, 2)
        self.value_spin.set_activates_default(True)
        g.attach(self.value_spin, 1, 0, 1, 1)
        self.get_content_area().add(g)
        self.show_all()

    def value(self):
        return self.value_spin.get_value()


def show_about(parent):
    """Open the About window: program name, version, description,
    developer (with the AI-construction note), license and a close
    button.  The text is kept in sync with the top of README.md."""
    from .. import (AI_NOTE, APP_NAME, COPYRIGHT, DESCRIPTION,
                    LICENSE_NAME, __author__, __version__)
    dlg = Gtk.AboutDialog(transient_for=parent, modal=True)
    dlg.set_program_name(APP_NAME)
    dlg.set_version(__version__)
    # description + AI note share the main face, next to the copyright
    dlg.set_comments(f"{DESCRIPTION}\n\n{AI_NOTE}")
    dlg.set_copyright(COPYRIGHT)
    # GPL_3_0 is GTK's "version 3 or later" license type
    dlg.set_license_type(Gtk.License.GPL_3_0)
    dlg.set_wrap_license(True)
    # the developer credit must always carry the AI-construction note
    dlg.set_authors([__author__, AI_NOTE])
    dlg.set_tooltip_text(LICENSE_NAME)
    dlg.run()
    dlg.destroy()


class VectorDialog(Gtk.Dialog):
    """Prompt for vector components (and optionally a length) when only
    one atom is selected."""

    def __init__(self, parent):
        super().__init__(title="Vector components", transient_for=parent,
                         modal=True)
        self.add_buttons("Cancel", Gtk.ResponseType.CANCEL,
                         "OK", Gtk.ResponseType.OK)
        self.set_default_response(Gtk.ResponseType.OK)
        g = Gtk.Grid(column_spacing=8, row_spacing=6, border_width=12)
        g.attach(Gtk.Label(label="Vector through the selected atom "
                                 "(bohr components):"), 0, 0, 2, 1)
        self.comps = []
        for i, lab in enumerate(("x:", "y:", "z:")):
            g.attach(Gtk.Label(label=lab, xalign=1.0), 0, i + 1, 1, 1)
            sp = _spin(1.0 if i == 2 else 0.0, -100, 100, 0.1)
            sp.set_activates_default(True)
            self.comps.append(sp)
            g.attach(sp, 1, i + 1, 1, 1)
        g.attach(Gtk.Label(label="Length (bohr):", xalign=1.0), 0, 4, 1, 1)
        self.length_spin = _spin(0.0, 0.0, 500.0, 0.5, 2)
        self.length_spin.set_tooltip_text(
            "Total drawn length; leave 0 to use the magnitude of the "
            "components")
        self.length_spin.set_activates_default(True)
        g.attach(self.length_spin, 1, 4, 1, 1)
        self.get_content_area().add(g)
        self.show_all()

    def vector(self):
        return [sp.get_value() for sp in self.comps]

    def length(self):
        """Requested length, or None to use the component magnitude."""
        v = self.length_spin.get_value()
        return v if v > 1e-9 else None


