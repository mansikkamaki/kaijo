# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian formatted checkpoint (.fchk) format: geometry, GTO basis and MOs.

The formatted checkpoint is a flat, self-describing text file.  Every field
is introduced by a 40-character label, a one-character type (I/R/C/L/H) and
either a scalar value or ``N=<count>`` followed by the data on the next
lines (six integers or five reals per line).  We index the field headers in
one pass and slice the (possibly millions of) numeric values straight out of
the raw text with numpy, so parsing is as fast as the molden path.

Only spherical-harmonic (pure) d/f/g shells are supported, matching the
molden loader and the grid evaluator; Cartesian d/f/g shells are rejected
with a clear message.  Gaussian's pure-spherical component order
(m = 0, +1, -1, +2, -2, ...) is exactly the molden order used by
``core.basis.solid_harmonics``, so no reordering or sign flips are needed.
"""

import os
import re

import numpy as np

from . import FormatHandler, LoadedData
from ..core.basis import BasisSet, MolecularOrbitals, Shell
from ..core.molecule import Molecule

# A field header: 40-char label, three spaces (Fortran 3X), a type letter,
# then either the scalar value or "N=<count>".  Data lines never reproduce
# this shape (three spaces + a type letter at column 43), so a single
# multiline scan separates headers from data cleanly.
_HEADER_RE = re.compile(r"^(.{40})   ([IRCLH])   (.*)$", re.MULTILINE)


class _Fchk:
    """Lazy index of an fchk file: field name -> header, data sliced on
    demand straight from the raw text."""

    def __init__(self, text):
        self.text = text
        self._fields = {}
        matches = list(_HEADER_RE.finditer(text))
        for i, m in enumerate(matches):
            name = m.group(1).strip()
            tail = m.group(3).strip()
            data_end = matches[i + 1].start() if i + 1 < len(matches) \
                else len(text)
            if tail.startswith("N="):
                self._fields[name] = (True, m.end(), data_end, None)
            else:
                self._fields[name] = (False, m.end(), data_end, tail)

    def has(self, name):
        return name in self._fields

    def scalar_int(self, name, default=None):
        f = self._fields.get(name)
        if f is None:
            if default is not None:
                return default
            raise KeyError(name)
        return int(f[3])

    def scalar_float(self, name, default=None):
        f = self._fields.get(name)
        if f is None:
            if default is not None:
                return default
            raise KeyError(name)
        return float(f[3])

    def array(self, name, dtype=float, optional=False):
        f = self._fields.get(name)
        if f is None:
            if optional:
                return None
            raise KeyError(f"fchk file has no '{name}' field")
        is_array, start, end, _ = f
        if not is_array:
            raise ValueError(f"fchk field '{name}' is not an array")
        vals = np.fromstring(self.text[start:end], sep=" ")
        if dtype is int:
            return np.rint(vals).astype(np.int64)
        return vals


# fchk shell-type codes: 0 = s, 1 = p, -1 = sp (combined s+p), and for the
# higher shells a negative code is spherical/pure while a positive code is
# Cartesian.  |code| is the angular momentum for the pure shells.
def _shell_l(code):
    if code in (0, 1):
        return code, False
    if code == -1:
        return -1, False          # SP marker
    if code <= -2:
        return -code, False       # pure d/f/g/...
    return code, True             # Cartesian d/f/g/... (unsupported)


def _build_basis(fchk, coords):
    shell_types = fchk.array("Shell types", dtype=int)
    nprims = fchk.array("Number of primitives per shell", dtype=int)
    shell_atom = fchk.array("Shell to atom map", dtype=int)
    exps = fchk.array("Primitive exponents")
    coeffs = fchk.array("Contraction coefficients")
    sp_coeffs = fchk.array("P(S=P) Contraction coefficients", optional=True)

    shells = []
    p = 0
    for code, nprim, iat in zip(shell_types, nprims, shell_atom):
        e = exps[p:p + nprim]
        c = coeffs[p:p + nprim]
        center = coords[iat - 1]
        l, cartesian = _shell_l(int(code))
        if cartesian:
            label = {2: "d", 3: "f", 4: "g"}.get(int(code), f"l={code}")
            raise ValueError(
                f"This fchk file uses Cartesian {label} functions, which "
                "are not supported (pure/spherical shells expected; set "
                "the calculation to use 5d/7f spherical harmonics)")
        if l == -1:  # SP: s coefficients here, p coefficients in P(S=P)
            shells.append(Shell(iat - 1, 0, center, e, c))
            cp = sp_coeffs[p:p + nprim]
            shells.append(Shell(iat - 1, 1, center, e, cp))
        else:
            if l > 4:
                raise ValueError(
                    f"Angular momentum l={l} in the fchk file exceeds the "
                    "supported maximum (g)")
            shells.append(Shell(iat - 1, l, center, e, c))
        p += nprim
    return BasisSet(shells)


def _build_orbitals(fchk, nbf):
    na = fchk.scalar_int("Number of alpha electrons")
    nb = fchk.scalar_int("Number of beta electrons")
    ea = fchk.array("Alpha Orbital Energies")
    ca = fchk.array("Alpha MO coefficients").reshape(len(ea), nbf)
    unrestricted = fchk.has("Beta MO coefficients")

    if unrestricted:
        eb = fchk.array("Beta Orbital Energies")
        cb = fchk.array("Beta MO coefficients").reshape(len(eb), nbf)
        coeffs = np.vstack([ca, cb])
        energies = np.concatenate([ea, eb])
        occs = np.concatenate([
            (np.arange(len(ea)) < na).astype(float),
            (np.arange(len(eb)) < nb).astype(float)])
        spins = np.concatenate([
            np.zeros(len(ea), dtype=np.int8),
            np.ones(len(eb), dtype=np.int8)])
    else:
        # One MO set.  Closed shell (na == nb) gives occupations of 2;
        # restricted open shell (na > nb) gives 2 in the doubly occupied
        # orbitals and 1 in the singly occupied ones.
        coeffs = ca
        energies = ea
        idx = np.arange(len(ea))
        occs = (idx < na).astype(float) + (idx < nb).astype(float)
        spins = np.zeros(len(ea), dtype=np.int8)
    return MolecularOrbitals(coeffs, energies, occs, spins)


class FchkHandler(FormatHandler):
    name = "fchk"
    extensions = (".fchk", ".fch", ".fck")

    def sniff(self, path, head):
        if super().sniff(path, head):
            return True
        # The third line of an fchk file is always the atom count; the basis
        # count follows a few lines later.
        return "Number of atoms" in head and \
            "Number of basis functions" in head

    def load(self, path, progress=None):
        if progress:
            progress(0.0, "Reading file...")
        with open(path, errors="replace") as fh:
            text = fh.read()
        fchk = _Fchk(text)

        numbers = fchk.array("Atomic numbers", dtype=int)
        coords = fchk.array("Current cartesian coordinates").reshape(-1, 3)
        mol = Molecule([int(z) for z in numbers], coords,
                       name=os.path.basename(path))

        basis = orbitals = None
        if fchk.has("Shell types") and fchk.has("Alpha MO coefficients"):
            if progress:
                progress(0.2, "Parsing basis set...")
            basis = _build_basis(fchk, coords)
            if progress:
                progress(0.5, "Parsing molecular orbitals...")
            orbitals = _build_orbitals(fchk, basis.nbf)
        if progress:
            progress(1.0, "Done")
        return LoadedData(mol, basis, orbitals, path=path)
