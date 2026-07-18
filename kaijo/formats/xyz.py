# SPDX-License-Identifier: GPL-3.0-or-later
"""XYZ geometry files.

Accepts both the standard form (atom count + comment line) and a bare
element/coordinate listing.  Coordinates are Å and converted to bohr.
"""

import os

import numpy as np

from . import FormatHandler, LoadedData
from ..core import elements
from ..core.elements import BOHR
from ..core.molecule import Molecule


def _parse_atom_line(line):
    parts = line.split()
    if len(parts) < 4:
        return None
    sym = parts[0]
    if sym.isdigit():
        z = int(sym)
    else:
        z = elements.symbol_to_z(sym)
    if z <= 0:
        return None
    try:
        x, y, zc = float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None
    return z, (x, y, zc)


class XYZHandler(FormatHandler):
    name = "xyz"
    extensions = (".xyz",)

    def sniff(self, path, head):
        if path.lower().endswith(".xyz"):
            return True
        lines = [l for l in head.splitlines() if l.strip()]
        return bool(lines) and _parse_atom_line(lines[0]) is not None

    def load(self, path, progress=None):
        with open(path) as fh:
            lines = fh.readlines()
        idx = 0
        # optional count line + comment line
        if lines and lines[0].split() and lines[0].split()[0].isdigit() \
                and _parse_atom_line(lines[0]) is None:
            idx = 2
        numbers, coords = [], []
        for line in lines[idx:]:
            if not line.strip():
                continue
            parsed = _parse_atom_line(line)
            if parsed is None:
                break
            numbers.append(parsed[0])
            coords.append(parsed[1])
        if not numbers:
            raise ValueError("No atoms found in xyz file")
        mol = Molecule(numbers, np.array(coords) / BOHR,
                       name=os.path.basename(path))
        return LoadedData(mol, path=path)
