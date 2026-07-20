# SPDX-License-Identifier: GPL-3.0-or-later
"""Axis-file feature: parser unit tests plus the GUI path that turns
``kaijo-run mol.molden mol.axis`` into a vector in the preview grid."""

import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import GLib  # noqa: E402

import numpy as np  # noqa: E402

from kaijo.core.elements import BOHR  # noqa: E402
from kaijo.formats.axis import (AxisError, parse_axis_file,  # noqa: E402
                                parse_axis_text)
from kaijo.ui import mainwindow  # noqa: E402

import testdata  # noqa: E402

FILE = testdata.path(testdata.GEOMETRY_FILE)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def test_parser():
    s = parse_axis_text("Dy 0 0 1")
    check(s.atom_symbol == "Dy" and s.atom_index is None, "symbol parse")
    check(np.allclose(s.vector, [0, 0, 1]), "components parse")
    check(s.length is None, "length should be absent")
    check(abs(s.length_bohr() - 1.0 / BOHR) < 1e-9, "fallback length")

    # free format: line breaks, commas, comments, trailing junk
    s = parse_axis_text("# comment\n 3 ,\n0.1 0.2\n0.3  5.0 ignored !x")
    check(s.atom_index == 2, "1-based index -> 0-based")
    check(np.allclose(s.vector, [0.1, 0.2, 0.3]), "multi-line components")
    check(abs(s.length_bohr() - 5.0 / BOHR) < 1e-9, "length in angstrom")

    check(parse_axis_text("dy 1 0 0").atom_symbol == "Dy",
          "symbols are case-insensitive")

    for text, what in (("", "empty file"),
                       ("Xx 0 0 1", "unknown element"),
                       ("Dy 0 0", "too few components"),
                       ("Dy 0 0 x", "non-numeric components"),
                       ("Dy 0 0 0", "zero vector"),
                       ("0 0 0 1", "index below one"),
                       ("Dy 0 0 1 -2", "non-positive length"),
                       ("Dy 0 0 1 x", "non-numeric length")):
        try:
            parse_axis_text(text)
        except AxisError:
            continue
        failures.append(f"{what} should have raised")

    try:
        parse_axis_file(os.path.join(tempfile.gettempdir(), "no-such.axis"))
        failures.append("missing file should have raised")
    except AxisError:
        pass


class Driver:
    """Loads the structure with an axis file and inspects the result."""

    def __init__(self, app, axis_path, expect_warning, element_z):
        self.app = app
        self.element_z = element_z
        self.axis_path = axis_path
        self.expect_warning = expect_warning
        self.warnings = []
        self.errors = []

    def start(self):
        win = self.app.win
        # the popups are modal; record them instead of blocking the test
        win.show_warning = lambda t, m: self.warnings.append((t, m))
        win.show_error = lambda t, m: self.errors.append((t, m))
        win.open_path(FILE, axis_path=self.axis_path)
        GLib.timeout_add(400, self.poll)

    def poll(self):
        win = self.app.win
        if win.data is None:
            return True
        self.check(win)
        self.app.quit()
        return False

    def check(self, win):
        vecs = [it for it in win.scene.items if it.kind == "vector"]
        check(len(vecs) == 1, f"expected one vector, got {len(vecs)}")
        check(not self.errors, f"unexpected error popup: {self.errors}")
        check(bool(self.warnings) == self.expect_warning,
              f"warning popup: {self.warnings} (expected "
              f"{self.expect_warning})")
        if not vecs:
            return
        item = vecs[0]
        mol = win.scene.molecule
        z = next(i for i, zi in enumerate(mol.numbers)
                 if zi == self.element_z)
        mid = 0.5 * (item.p0 + item.p1)
        check(np.allclose(mid, mol.coords[z], atol=1e-9),
              "vector is not centred on the first matching atom")
        d = item.p1 - item.p0
        check(np.allclose(d / np.linalg.norm(d), [0, 0, 1], atol=1e-9),
              "vector direction wrong")
        check(abs(item.length - 4.0 / BOHR) < 1e-9,
              f"length {item.length} bohr, expected 4 A")
        p0, p1 = item.endpoints()
        check(abs(np.linalg.norm(p1 - p0) - item.length) < 1e-9,
              "drawn endpoints do not match the requested length")
        # the structure stays selected; the vector only joins the grid
        check(win.active_item.kind == "structure",
              "the axis vector should not steal the main view")
        check(win.preview.cell_for(item) is not None,
              "no preview cell for the axis vector")


def run_gui(axis_text, expect_warning, element_z):
    with tempfile.NamedTemporaryFile("w", suffix=".axis",
                                     delete=False) as fh:
        fh.write(axis_text)
        axis_path = fh.name
    try:
        app = mainwindow.KaijoApp()
        driver = Driver(app, axis_path, expect_warning, element_z)
        app.connect("activate", lambda a: GLib.timeout_add(300,
                                                           driver.start))
        app.run(None)
    finally:
        os.unlink(axis_path)


def main():
    test_parser()
    # single Gd in the test complex -> no ambiguity warning
    run_gui("Gd 0.0 0.0 1.0 4.0\n", expect_warning=False, element_z=64)
    # many carbons -> first one is used and a warning popup is raised
    run_gui("# axis\nc\n0 0 1\n4.0\n", expect_warning=True, element_z=6)
    print("FAILURES:" if failures else "AXIS OK", "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
