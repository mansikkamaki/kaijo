# SPDX-License-Identifier: GPL-3.0-or-later
"""ORCA CASSCF active-space helper.

Given the orbitals selected in the preview grid, work out the orbital
rotations that bring them into the active space of an ORCA CASSCF
calculation and present them as a ready-to-copy ``%scf Rotate`` block.

ORCA splits the orbital space into three blocks (in molden order):

  * internal -- doubly occupied, ORCA indices ``0 .. n_internal-1``;
  * active   -- the correlated orbitals, the next ``n_active`` indices;
  * external -- unoccupied, everything after.

The number of internal orbitals is fixed by the electron count::

    n_internal = (n_electrons - n_active_electrons) / 2

``n_electrons`` is taken from the orbital occupations rather than the
nuclear charges, so effective-core-potential (ECP) core electrons -- which
are not represented by any orbital -- are correctly excluded.

IMPORTANT: ORCA numbers orbitals from 0, whereas Kaijo's per-spin MO index
starts at 1, so every ORCA index here is ``spin_index - 1``.
"""

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gtk


def total_electrons(orbitals):
    """Electrons represented by the orbitals (ECP core excluded).

    Uses the sum of the occupations, which equals the number of electrons
    actually in the orbital space; the nuclear charges would over-count
    whenever an ECP is in use."""
    return int(round(float(orbitals.occupations.sum())))


def active_electron_bounds(n_electrons, n_active):
    """(lo, hi) inclusive range of allowed active-electron counts, both of
    the same parity as ``n_electrons`` (an odd-electron system needs an odd
    active count, an even-electron system an even one).  Active orbitals hold
    at most two electrons, and the internal block cannot be negative."""
    parity = n_electrons % 2
    lo = parity
    hi = min(2 * n_active, n_electrons)
    if hi % 2 != parity:
        hi -= 1
    return lo, max(lo, hi)


def cas_rotations(selected_orca, n_internal):
    """Swaps that move the selected orbitals into the active window.

    ``selected_orca`` is the list of 0-based ORCA indices that should form
    the active space.  The active window is ``[n_internal, n_internal +
    n_active)``.  Every selected orbital lying outside the window is paired
    with a window slot currently holding a non-selected orbital, and the two
    are swapped.  The two pools are disjoint and each index is used once, so
    no orbital appears in more than one rotation.  The order within the
    active space is irrelevant, so the pairing is arbitrary (ascending)."""
    n_active = len(selected_orca)
    window = set(range(n_internal, n_internal + n_active))
    sel = set(selected_orca)
    move_in = sorted(sel - window)     # selected, still outside the window
    move_out = sorted(window - sel)    # window slots to vacate
    return list(zip(move_in, move_out))


def format_rotation_block(rotations):
    """ORCA ``%scf Rotate`` block for the given (a, b) swaps."""
    lines = ["%scf", "  Rotate"]
    if rotations:
        for a, b in rotations:
            lines.append(f"    {{{a},{b},90}}")
    else:
        lines.append("    # no rotations needed: the selected orbitals")
        lines.append("    # already occupy the active space")
    lines.append("  end")
    lines.append("end")
    return "\n".join(lines)


class CasDialog(Gtk.Dialog):
    """Window showing the active-space rotations for the selected orbitals.

    ``orb_indices`` are Kaijo orbital array indices, already validated to be
    alpha orbitals (ORCA builds the CASSCF guess from the alpha set only)."""

    def __init__(self, parent, orbitals, orb_indices):
        super().__init__(title="ORCA CASSCF active space",
                         transient_for=parent, modal=True)
        self.add_buttons("Close", Gtk.ResponseType.CLOSE)
        self.orbitals = orbitals

        self.selected_orca = sorted(
            int(orbitals.spin_index[i]) - 1 for i in orb_indices)
        self.n_active = len(self.selected_orca)
        self.n_electrons = total_electrons(orbitals)
        self.n_orbitals = int((orbitals.spins == 0).sum())
        sel_occ = float(sum(orbitals.occupations[i] for i in orb_indices))

        grid = Gtk.Grid(column_spacing=8, row_spacing=8, border_width=12)
        row = 0

        grid.attach(Gtk.Label(
            label=f"Orbitals selected for the active space: {self.n_active}",
            xalign=0.0), 0, row, 2, 1)

        row += 1
        grid.attach(Gtk.Label(label="Active electrons:", xalign=1.0),
                    0, row, 1, 1)
        lo, hi = active_electron_bounds(self.n_electrons, self.n_active)
        default = self._default_electrons(sel_occ, lo, hi)
        adj = Gtk.Adjustment(value=default, lower=lo, upper=hi,
                             step_increment=2, page_increment=2)
        self.elec_spin = Gtk.SpinButton(adjustment=adj, digits=0)
        self.elec_spin.set_numeric(True)
        parity = "odd" if self.n_electrons % 2 else "even"
        self.elec_spin.set_tooltip_text(
            f"Number of electrons in the active space. The system has "
            f"{self.n_electrons} electrons (an {parity} number), so this "
            f"must be {parity} too; the step is 2.")
        self.elec_spin.connect("value-changed", lambda *_: self._recompute())
        grid.attach(self.elec_spin, 1, row, 1, 1)

        row += 1
        self.info = Gtk.Label(xalign=0.0)
        self.info.set_line_wrap(True)
        grid.attach(self.info, 0, row, 2, 1)

        row += 1
        grid.attach(Gtk.Label(
            label="Rotations (paste into the ORCA input):", xalign=0.0),
            0, row, 2, 1)

        row += 1
        self.buffer = Gtk.TextBuffer()
        view = Gtk.TextView(buffer=self.buffer)
        view.set_editable(False)
        view.set_monospace(True)
        view.set_left_margin(6)
        view.set_right_margin(6)
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        sw.set_min_content_height(180)
        sw.set_min_content_width(320)
        sw.set_hexpand(True)
        sw.set_vexpand(True)
        sw.add(view)
        grid.attach(sw, 0, row, 2, 1)

        row += 1
        copy_btn = Gtk.Button(label="Copy to clipboard")
        copy_btn.set_halign(Gtk.Align.START)
        copy_btn.connect("clicked", self._copy)
        grid.attach(copy_btn, 0, row, 1, 1)

        row += 1
        note = Gtk.Label(xalign=0.0)
        note.set_line_wrap(True)
        note.set_markup(
            "<small><b>Note:</b> ORCA numbers orbitals starting from 0, so "
            "these indices are one less than the MO indices shown in Kaijo "
            "(which start at 1). The electron count is taken from the "
            "orbital occupations, so ECP core electrons are excluded.</small>")
        grid.attach(note, 0, row, 2, 1)

        self.get_content_area().add(grid)
        self._recompute()
        self.show_all()

    def _default_electrons(self, sel_occ, lo, hi):
        """Nearest allowed count to the electrons currently occupying the
        selected orbitals."""
        raw = int(round(sel_occ))
        # snap to the parity of lo, then clamp into [lo, hi]
        raw -= (raw - lo) % 2
        return min(max(raw, lo), hi)

    def _recompute(self, *_a):
        n_elec = self.n_electrons
        n_active_elec = int(self.elec_spin.get_value())
        n_internal = (n_elec - n_active_elec) // 2
        last = n_internal + self.n_active - 1
        overflow = n_internal + self.n_active > self.n_orbitals

        internal_range = (f"(ORCA 0–{n_internal - 1})" if n_internal
                          else "(none)")
        info = (f"System electrons (from occupations): {n_elec}\n"
                f"Internal (doubly occupied) orbitals: {n_internal}  "
                f"{internal_range}\n"
                f"Active space: {self.n_active} orbitals / {n_active_elec} "
                f"electrons  (ORCA {n_internal}–{last})")
        if overflow:
            info += ("\n⚠ The active window runs past the "
                     f"{self.n_orbitals} available orbitals; reduce the "
                     "active electrons.")
        self.info.set_text(info)

        if overflow:
            self.buffer.set_text(
                "# Cannot build the rotation block: too few orbitals for "
                "this\n# internal-space size. Lower the active electron "
                "count.")
            return
        rotations = cas_rotations(self.selected_orca, n_internal)
        self.buffer.set_text(format_rotation_block(rotations))

    def _copy(self, *_a):
        start, end = self.buffer.get_bounds()
        text = self.buffer.get_text(start, end, False)
        clip = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clip.set_text(text, -1)
