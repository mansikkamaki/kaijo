# SPDX-License-Identifier: GPL-3.0-or-later
"""Molden format: geometry, GTO basis, and molecular orbitals.

Spherical-harmonic bases ([5D]/[7F]/[9G]) are supported; Cartesian d/f/g
shells are rejected with a clear error.  The MO coefficient block (which can
be millions of lines) is parsed with numpy for speed.
"""

import os
import re

import numpy as np

from . import FormatHandler, LoadedData
from ..core import elements
from ..core.basis import BasisSet, MolecularOrbitals, Shell, L_LABELS
from ..core.elements import BOHR
from ..core.molecule import Molecule

_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")


def _split_sections(text):
    """Split the molden file into {section_name_lower: (args, body_lines)}."""
    sections = {}
    name, args, body = None, "", []
    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            if name is not None:
                sections[name] = (args, body)
            name = m.group(1).strip().lower()
            args = m.group(2).strip()
            body = []
        elif name is not None:
            body.append(line)
    if name is not None:
        sections[name] = (args, body)
    return sections


def _parse_atoms(args, body):
    unit = args.strip().lower()
    to_bohr = 1.0 if unit.startswith("au") else 1.0 / BOHR
    numbers, coords = [], []
    for line in body:
        parts = line.split()
        if len(parts) < 6:
            continue
        # Column 1 is the element symbol, column 3 the atomic number.
        # They normally agree, but Gaussian-derived molden files that use
        # an effective core potential write the ECP-reduced charge in the
        # number column (e.g. Gd -> 36 for a 28e core, or 11 for a
        # 4f-in-core ECP), which would misidentify the element as Kr/Na.
        # The symbol is the reliable identity, so prefer it and fall back
        # to the numeric column only when the symbol is unknown.
        z = elements.symbol_to_z(parts[0])
        if z == 0:
            z = int(float(parts[2]))
        xyz = [float(parts[3]), float(parts[4]), float(parts[5])]
        numbers.append(z)
        coords.append([c * to_bohr for c in xyz])
    return numbers, np.array(coords)


def _parse_gto(body, coords, coeffs_are_raw=False):
    """Parse the [GTO] section into a list of Shell objects."""
    shells = []
    i = 0
    n = len(body)
    while i < n:
        parts = body[i].split()
        i += 1
        if not parts:
            continue
        atom_idx = int(parts[0]) - 1
        # shells for this atom until a blank line
        while i < n:
            line = body[i].strip()
            if not line:
                i += 1
                break
            parts = line.split()
            label = parts[0].lower()
            if label not in L_LABELS and label != "sp":
                raise ValueError(f"Unsupported shell type '{parts[0]}' "
                                 "in molden [GTO] section")
            nprim = int(parts[1])
            i += 1
            exps, c1, c2 = [], [], []
            for _ in range(nprim):
                pp = body[i].replace("D", "E").replace("d", "e").split()
                exps.append(float(pp[0]))
                c1.append(float(pp[1]))
                if label == "sp":
                    c2.append(float(pp[2]))
                i += 1
            if label == "sp":
                shells.append(Shell(atom_idx, 0, coords[atom_idx], exps, c1,
                                    coeffs_are_raw))
                shells.append(Shell(atom_idx, 1, coords[atom_idx], exps, c2,
                                    coeffs_are_raw))
            else:
                shells.append(Shell(atom_idx, L_LABELS[label],
                                    coords[atom_idx], exps, c1,
                                    coeffs_are_raw))
    return shells


# ORCA writes some spherical components with inverted sign relative to the
# molden standard.  Component indices in molden order (0,+1,-1,+2,-2,...).
_ORCA_SIGN_FLIP = {3: (5, 6), 4: (5, 6, 7, 8)}  # f(+/-3), g(+/-3), g(+/-4)


def _orca_sign_vector(basis):
    import numpy as _np
    signs = _np.ones(basis.nbf)
    for sh in basis.shells:
        for comp in _ORCA_SIGN_FLIP.get(sh.l, ()):
            signs[sh.offset + comp] = -1.0
    return signs


_MO_HEAD_RE = re.compile(
    r"^[ \t]*(Sym|Ene|Spin|Occup)\s*=\s*(\S+)[ \t]*$",
    re.MULTILINE | re.IGNORECASE)


def _parse_mos(body_text, nbf, progress=None):
    """Parse the [MO] section body (single string) with numpy."""
    heads = list(_MO_HEAD_RE.finditer(body_text))
    if not heads:
        raise ValueError("Empty [MO] section in molden file")
    # Group consecutive header lines into orbital records; the coefficient
    # block of a record spans from its last header to the next record's
    # first header.
    records = []  # (meta dict, coef_start, coef_end)
    cur = {}
    cur_end = None
    starts = []
    for m in heads:
        key = m.group(1).lower()
        if key in cur:  # new orbital record begins
            records.append((cur, cur_end))
            starts.append(m.start())
            cur = {}
        elif not cur:
            starts.append(m.start())
        cur[key] = m.group(2)
        cur_end = m.end()
    records.append((cur, cur_end))

    nmo = len(records)
    coeffs = np.zeros((nmo, nbf))
    energies = np.zeros(nmo)
    occs = np.zeros(nmo)
    spins = np.zeros(nmo, dtype=np.int8)
    labels = []
    for irec, (meta, coef_start) in enumerate(records):
        coef_end = starts[irec + 1] if irec + 1 < nmo else len(body_text)
        block = body_text[coef_start:coef_end]
        data = np.fromstring(block, sep=" ")
        if data.size % 2:
            raise ValueError("Malformed MO coefficient block")
        pairs = data.reshape(-1, 2)
        idx = pairs[:, 0].astype(int) - 1
        if idx.size and (idx.min() < 0 or idx.max() >= nbf):
            raise ValueError("MO coefficient index out of range "
                             f"(nbf={nbf})")
        coeffs[irec, idx] = pairs[:, 1]
        energies[irec] = float(meta.get("ene", "0").replace("D", "E"))
        occs[irec] = float(meta.get("occup", "0").replace("D", "E"))
        spins[irec] = 1 if meta.get("spin", "Alpha").lower().startswith("b") \
            else 0
        labels.append(meta.get("sym", ""))
        if progress and irec % 64 == 0:
            progress(irec / nmo)
    return MolecularOrbitals(coeffs, energies, occs, spins, labels)


class MoldenHandler(FormatHandler):
    name = "molden"
    extensions = (".molden", ".molden.input", ".input")

    def sniff(self, path, head):
        return "[molden format]" in head.lower() or \
            super().sniff(path, head)

    def load(self, path, progress=None):
        if progress:
            progress(0.0, "Reading file...")
        with open(path, errors="replace") as fh:
            text = fh.read()
        # Split off the [MO] body as raw text (fast numpy parse); everything
        # before it goes through the line-based section splitter.
        mo_match = re.search(r"^\s*\[MO\]\s*$", text,
                             re.MULTILINE | re.IGNORECASE)
        head_text = text[:mo_match.start()] if mo_match else text
        mo_text = text[mo_match.end():] if mo_match else ""
        sections = _split_sections(head_text)

        if "atoms" not in sections:
            raise ValueError("Molden file has no [Atoms] section")
        numbers, coords = _parse_atoms(*sections["atoms"])
        mol = Molecule(numbers, coords, name=os.path.basename(path))

        # ORCA molden files deviate from the standard in two documented
        # ways: contraction coefficients include the primitive norms, and
        # the signs of the f(+/-3), g(+/-3) and g(+/-4) components are
        # inverted.
        is_orca = "orca" in text[:4096].lower()

        basis = orbitals = None
        if "gto" in sections and mo_text:
            if progress:
                progress(0.1, "Parsing basis set...")
            shells = _parse_gto(sections["gto"][1], coords,
                                coeffs_are_raw=is_orca)
            spherical = any(k in sections for k in ("5d", "5d7f", "5d10f",
                                                    "7f", "9g"))
            max_l = max((sh.l for sh in shells), default=0)
            if max_l > 1 and not spherical:
                raise ValueError(
                    "This molden file uses Cartesian d/f/g functions, which "
                    "are not supported (spherical [5D]/[7F]/[9G] expected)")
            basis = BasisSet(shells)
            if progress:
                progress(0.15, "Parsing molecular orbitals...")
            orbitals = _parse_mos(
                mo_text, basis.nbf,
                progress=(lambda f: progress(0.15 + 0.85 * f,
                                             "Parsing molecular orbitals..."))
                if progress else None)
            if is_orca and basis.max_l() >= 3:
                orbitals.coeffs *= _orca_sign_vector(basis)
        return LoadedData(mol, basis, orbitals, path=path)
