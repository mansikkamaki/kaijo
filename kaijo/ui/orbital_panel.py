# SPDX-License-Identifier: GPL-3.0-or-later
"""Options bar: orbital listing, visualization controls, geometry tools.

The orbital list is sorted by occupation (descending), then energy
(ascending), with alpha and beta orbitals interleaved, as per the spec.
"""

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GObject, Gtk

_SPIN_TIP = ("Compute the spin density (alpha minus beta) and add its "
             "isosurface to the preview grid")


class OrbitalPanel(Gtk.Box):
    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_border_width(6)
        self.orbitals = None
        self.on_visualize = None       # callback(list of orbital indices)
        self.on_calculate = None       # callback("density"|"spin"|"esp")
        self.on_make_vector = None
        self.on_make_plane = None
        self.on_make_polyhedron = None
        self.on_set_bond = None        # callback(order or None)
        self.on_clear_selection = None

        # --- orbital list ---------------------------------------------
        frame = Gtk.Frame(label="Molecular orbitals")
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        vbox.set_border_width(4)
        frame.add(vbox)

        self.store = Gtk.ListStore(int, str, str, str, str)
        # columns: orbital index (hidden), name, energy, occ, spin
        self.view = Gtk.TreeView(model=self.store)
        self.view.set_rules_hint(True)
        for i, (title, w) in enumerate(
                (("MO", 52), ("E / Eh", 86), ("occ", 56), ("spin", 44))):
            cell = Gtk.CellRendererText()
            if i in (1, 2):
                cell.set_property("xalign", 1.0)
            col = Gtk.TreeViewColumn(title, cell, text=i + 1)
            col.set_min_width(w)
            col.set_resizable(True)
            self.view.append_column(col)
        self.view.get_selection().set_mode(Gtk.SelectionMode.MULTIPLE)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        sw.set_min_content_height(260)
        sw.add(self.view)
        vbox.pack_start(sw, True, True, 0)

        # quick range selection
        rbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.range_entry = Gtk.Entry()
        self.range_entry.set_placeholder_text("e.g. 120-130 or 42,45a")
        self.range_entry.set_tooltip_text(
            "Select orbitals by number: ranges (120-130), single orbitals "
            "(42), alpha/beta (45a, 45b); comma-separated")
        self.range_entry.connect("activate", self._select_range)
        rbox.pack_start(self.range_entry, True, True, 0)
        btn = Gtk.Button(label="Select")
        btn.connect("clicked", self._select_range)
        rbox.pack_start(btn, False, False, 0)
        vbox.pack_start(rbox, False, False, 0)

        self.viz_button = Gtk.Button(label="Visualize selected orbitals")
        self.viz_button.get_style_context().add_class("suggested-action")
        self.viz_button.connect("clicked", self._visualize)
        vbox.pack_start(self.viz_button, False, False, 0)
        self.pack_start(frame, True, True, 0)

        # --- calculate (density / spin density / ESP) -----------------
        css = Gtk.CssProvider()
        css.load_from_data(b"""
            button.kaijo-calc {
                background-image: none;
                background-color: #2b8a3e;
                color: #ffffff;
                text-shadow: none;
            }
            button.kaijo-calc:hover { background-color: #37b24d; }
            button.kaijo-calc:disabled {
                background-color: #b9d8c1;
                color: #f0f4f0;
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        cframe = Gtk.Frame(label="Calculate")
        cgrid = Gtk.Grid(column_spacing=4, row_spacing=4, border_width=4)
        cgrid.set_column_homogeneous(True)
        self.dens_btn = Gtk.Button(label="Density")
        self.dens_btn.set_tooltip_text(
            "Compute the electron density and add its isosurface to the "
            "preview grid")
        self.spin_btn = Gtk.Button(label="Spin density")
        self.spin_btn.set_tooltip_text(_SPIN_TIP)
        self.esp_btn = Gtk.Button(label="ESP")
        self.esp_btn.set_tooltip_text(
            "Compute the electrostatic potential mapped onto an electron "
            "density isosurface and add it to the preview grid")
        for n, (btn, kind) in enumerate(
                ((self.dens_btn, "density"), (self.spin_btn, "spin"),
                 (self.esp_btn, "esp"))):
            btn.get_style_context().add_class("kaijo-calc")
            btn.connect("clicked",
                        lambda b, k=kind: self.on_calculate
                        and self.on_calculate(k))
            cgrid.attach(btn, n, 0, 1, 1)
        cframe.add(cgrid)
        self.pack_start(cframe, False, False, 0)

        # --- selection / geometry tools -------------------------------
        tframe = Gtk.Frame(label="Selection / geometry")
        tbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        tbox.set_border_width(4)
        tframe.add(tbox)

        self.sel_label = Gtk.Label(label="No atoms selected")
        self.sel_label.set_xalign(0.0)
        self.sel_label.set_line_wrap(True)
        tbox.pack_start(self.sel_label, False, False, 0)

        grid = Gtk.Grid()
        grid.set_column_spacing(4)
        grid.set_row_spacing(4)
        self.vec_btn = Gtk.Button(label="Vector")
        self.vec_btn.set_tooltip_text(
            "Vector through two selected atoms, or prompted components "
            "for one selected atom")
        self.vec_btn.connect("clicked",
                             lambda b: self.on_make_vector
                             and self.on_make_vector())
        self.plane_btn = Gtk.Button(label="Plane")
        self.plane_btn.set_tooltip_text(
            "Best-fit plane through three or more selected atoms")
        self.plane_btn.connect("clicked",
                               lambda b: self.on_make_plane
                               and self.on_make_plane())
        self.poly_btn = Gtk.Button(label="Polyhedron")
        self.poly_btn.set_tooltip_text(
            "Coordination polyhedron around the selected atoms")
        self.poly_btn.connect("clicked",
                              lambda b: self.on_make_polyhedron
                              and self.on_make_polyhedron())
        clear_btn = Gtk.Button(label="Clear (Esc)")
        clear_btn.connect("clicked",
                          lambda b: self.on_clear_selection
                          and self.on_clear_selection())
        grid.attach(self.vec_btn, 0, 0, 1, 1)
        grid.attach(self.plane_btn, 1, 0, 1, 1)
        grid.attach(self.poly_btn, 0, 1, 1, 1)
        grid.attach(clear_btn, 1, 1, 1, 1)
        tbox.pack_start(grid, False, False, 0)

        bbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        bbox.pack_start(Gtk.Label(label="Bond:"), False, False, 0)
        self.bond_combo = Gtk.ComboBoxText()
        for label in ("auto", "none", "single", "double", "triple"):
            self.bond_combo.append_text(label)
        self.bond_combo.set_active(0)
        self.bond_combo.set_tooltip_text(
            "Set the bond type between the two selected atoms")
        bbox.pack_start(self.bond_combo, True, True, 0)
        self.bond_btn = Gtk.Button(label="Apply")
        self.bond_btn.connect("clicked", self._apply_bond)
        bbox.pack_start(self.bond_btn, False, False, 0)
        tbox.pack_start(bbox, False, False, 0)
        self.pack_start(tframe, False, False, 0)
        self.update_selection([])

    # ----------------------------------------------------------- orbitals

    def set_orbitals(self, orbitals):
        self.orbitals = orbitals
        self.store.clear()
        have = orbitals is not None
        self.dens_btn.set_sensitive(have)
        self.esp_btn.set_sensitive(have)
        unres = have and orbitals.unrestricted
        self.spin_btn.set_sensitive(unres)
        self.spin_btn.set_tooltip_text(
            _SPIN_TIP if unres or not have else
            "Spin density needs unrestricted (alpha/beta) orbitals; "
            "this file is restricted")
        if orbitals is None:
            self.viz_button.set_sensitive(False)
            return
        self.viz_button.set_sensitive(True)
        homo = orbitals.homo_index()
        homo_row = None
        for row, i in enumerate(orbitals.display_order()):
            spin = ("α" if orbitals.spins[i] == 0 else "β") \
                if orbitals.unrestricted else ""
            name = orbitals.name(i)
            if i == homo:
                name += " (HOMO)"
            self.store.append([
                int(i), name, f"{orbitals.energies[i]:.4f}",
                f"{orbitals.occupations[i]:.3f}", spin])
            if i == homo:
                homo_row = row
        if homo_row is not None:
            path = Gtk.TreePath.new_from_indices([homo_row])
            self.view.scroll_to_cell(path, None, True, 0.5, 0.0)
            self.view.get_selection().select_path(path)

    def selected_orbitals(self):
        model, paths = self.view.get_selection().get_selected_rows()
        return [model[p][0] for p in paths]

    def unselect_orbital(self, orb_index):
        sel = self.view.get_selection()
        for row, treerow in enumerate(self.store):
            if treerow[0] == orb_index:
                sel.unselect_path(Gtk.TreePath.new_from_indices([row]))
                break

    def _visualize(self, *_a):
        if self.on_visualize:
            self.on_visualize(self.selected_orbitals())

    def _select_range(self, *_a):
        if self.orbitals is None:
            return
        text = self.range_entry.get_text().strip()
        if not text:
            return
        wanted = set()
        for part in text.replace(" ", "").split(","):
            if not part:
                continue
            spin = None
            if part[-1] in "abAB" and not part[-1].isdigit():
                spin = 0 if part[-1].lower() == "a" else 1
                part = part[:-1]
            try:
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    rng = range(int(lo), int(hi) + 1)
                else:
                    rng = [int(part)]
            except ValueError:
                continue
            for i in range(self.orbitals.nmo):
                if int(self.orbitals.channel_index[i]) in rng and \
                        (spin is None or self.orbitals.spins[i] == spin):
                    wanted.add(i)
        sel = self.view.get_selection()
        sel.unselect_all()
        for row, treerow in enumerate(self.store):
            if treerow[0] in wanted:
                sel.select_path(Gtk.TreePath.new_from_indices([row]))

    # ---------------------------------------------------------- selection

    def update_selection(self, selected_atoms, molecule=None):
        n = len(selected_atoms)
        if n == 0:
            text = "No atoms selected (click atoms in the 3D view)"
        else:
            names = []
            if molecule is not None:
                from ..core import elements
                names = [f"{elements.z_to_symbol(molecule.numbers[i])}"
                         f"{i + 1}" for i in selected_atoms]
            text = f"{n} atom(s): " + ", ".join(names[:12]) + \
                ("..." if n > 12 else "")
        self.sel_label.set_text(text)
        self.vec_btn.set_sensitive(n in (1, 2))
        self.plane_btn.set_sensitive(n >= 3)
        self.poly_btn.set_sensitive(n >= 4)
        self.bond_btn.set_sensitive(n == 2)
        self.bond_combo.set_sensitive(n == 2)

    def _apply_bond(self, *_a):
        if self.on_set_bond:
            txt = self.bond_combo.get_active_text()
            order = {"auto": None, "none": 0, "single": 1,
                     "double": 2, "triple": 3}[txt]
            self.on_set_bond(order)
