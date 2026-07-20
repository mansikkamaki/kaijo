# SPDX-License-Identifier: GPL-3.0-or-later
# Kaijo -- molecular orbital and molecular structure visualization.
# Copyright (C) 2026  Akseli Mansikkamäki
#
# Developed by Akseli Mansikkamäki and constructed with AI using Claude
# Code (models: Claude Opus 4.8 and Claude Fable 5.0).  Licensed under the
# GNU General Public License, version 3 or later; see the LICENSE file.
"""Axis (vector) files given on the command line alongside a structure.

The file is free-format: comments (`#`, `!`) are stripped and everything
else is read as a flat token stream, so line breaks and separators do not
matter.  The tokens are

    <atom>  <x> <y> <z>  [length]

where `<atom>` is either a case-insensitive element symbol (the first atom
of that element is used) or a 1-based index into the coordinate listing,
`<x> <y> <z>` are the vector components and the optional `length` is the
total drawn length in Ångström.  Without it the magnitude of the
components is used as the length (also in Ångström).  Trailing tokens are
ignored.
"""

import numpy as np

from ..core.elements import BOHR, symbol_to_z


class AxisError(Exception):
    """The axis file could not be read -- non-critical, reported to the
    user while the program keeps running."""


class AxisSpec:
    """Parsed contents of an axis file."""

    def __init__(self, atom_index=None, atom_symbol=None,
                 vector=(0.0, 0.0, 1.0), length=None):
        self.atom_index = atom_index      # 0-based, or None
        self.atom_symbol = atom_symbol    # element symbol, or None
        self.vector = np.asarray(vector, dtype=np.float64)
        self.length = length              # Å, or None

    def resolve_atom(self, molecule):
        """Return (index, warning) for `molecule`; warning may be None."""
        if self.atom_index is not None:
            if not 0 <= self.atom_index < molecule.natoms:
                raise AxisError(
                    f"Atom index {self.atom_index + 1} is outside the "
                    f"structure ({molecule.natoms} atoms).")
            return self.atom_index, None
        z = symbol_to_z(self.atom_symbol)
        hits = [i for i, zi in enumerate(molecule.numbers) if zi == z]
        if not hits:
            raise AxisError(f"The structure contains no {self.atom_symbol} "
                            "atom.")
        warning = None
        if len(hits) > 1:
            warning = (f"The structure contains {len(hits)} "
                       f"{self.atom_symbol} atoms; the vector was centred "
                       f"on the first one (atom {hits[0] + 1}).")
        return hits[0], warning

    def length_bohr(self):
        """Total drawn length in bohr."""
        length = self.length
        if length is None:
            length = float(np.linalg.norm(self.vector))
        return length / BOHR


def _tokens(text):
    out = []
    for line in text.splitlines():
        for stop in ("#", "!"):
            pos = line.find(stop)
            if pos >= 0:
                line = line[:pos]
        out.extend(line.replace(",", " ").split())
    return out


def parse_axis_text(text):
    tokens = _tokens(text)
    if not tokens:
        raise AxisError("The file is empty.")
    head = tokens[0]
    index = symbol = None
    try:
        index = int(head)
    except ValueError:
        if symbol_to_z(head) == 0:
            raise AxisError(f"'{head}' is neither an element symbol nor an "
                            "atom index.") from None
        symbol = head.strip().capitalize()
    else:
        if index < 1:
            raise AxisError(f"Atom index {index} is not a positive integer.")
        index -= 1
    if len(tokens) < 4:
        raise AxisError("Expected an atom followed by three vector "
                        f"components, found {len(tokens)} value(s).")
    try:
        vector = [float(t) for t in tokens[1:4]]
    except ValueError:
        raise AxisError("The vector components "
                        f"'{' '.join(tokens[1:4])}' are not numbers.") \
            from None
    if float(np.linalg.norm(vector)) < 1e-12:
        raise AxisError("The vector components are all zero.")
    length = None
    if len(tokens) > 4:
        try:
            length = float(tokens[4])
        except ValueError:
            raise AxisError(f"The vector length '{tokens[4]}' is not a "
                            "number.") from None
        if length <= 0.0:
            raise AxisError(f"The vector length {length} is not positive.")
    return AxisSpec(index, symbol, vector, length)


def parse_axis_file(path):
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        raise AxisError(str(exc)) from None
    return parse_axis_text(text)
