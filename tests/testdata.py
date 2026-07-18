# SPDX-License-Identifier: GPL-3.0-or-later
"""Curated, self-contained test data.

The files in ``tests/data/`` are copied from ``example_data/`` so the test
suite runs even after ``example_data/`` is removed from a published build.
The set is deliberately small but spans the parsing paths that matter:

  * software    : ORCA 6 and Gaussian 16 (via ccwrite)
  * method      : restricted (rks), unrestricted (uks), restricted
                  open-shell (roks)
  * basis       : Pople [5D10F], def2 and x2c families
  * angular mom.: up to f (ORCA sign flip on f+/-3) and g (g+/-3, g+/-4)
  * ECP         : all-electron, ORCA small-core (full nuclear charge in
                  the file) and Gaussian 4f-in-core (ECP-reduced charge in
                  the file -- the element must still be recognised by its
                  symbol, not the numeric column)

Each entry lists the properties the loader must reproduce, so the tests
double as a regression guard on the parser, not merely a "does it run".
"""

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(name):
    return os.path.join(DATA_DIR, name)


# name -> expected properties.  ``norm_offsets`` are orbital indices
# relative to the HOMO whose grid norm <psi|psi> must come out ~1 on a
# uniform grid; they are chosen among diffuse valence orbitals (compact
# 4f frontier orbitals of the small-core lanthanide file are excluded
# because a uniform grid cannot integrate them, which is a quadrature
# limitation, not a parsing error).  ``heavy_z`` asserts correct element
# identification (important for the ECP files).
MOLDEN_FILES = [
    dict(name="water-orca-rks-x2c-none.molden",
         software="orca", natoms=3, nbf=43, max_l=3,
         unrestricted=False, nelec=10, sum_z=10, heavy_z=8,
         norm_offsets=(0, -1, -2)),
    dict(name="water-g16-rks-def2-none.molden",
         software="g16", natoms=3, nbf=43, max_l=3,
         unrestricted=False, nelec=10, sum_z=10, heavy_z=8,
         norm_offsets=(0, -1, -2)),
    dict(name="benzene-g16-rks-pople-none.molden",
         software="g16", natoms=12, nbf=144, max_l=2,
         unrestricted=False, nelec=42, sum_z=42, heavy_z=6,
         norm_offsets=(0, -1, -2)),
    dict(name="xylylene-orca-uks-x2c-none.molden",
         software="orca", natoms=16, nbf=296, max_l=3,
         unrestricted=True, nelec=56, sum_z=56, heavy_z=6,
         norm_offsets=(0, -1, -2)),
    dict(name="xylylene-g16-uks-pople-none.molden",
         software="g16", natoms=16, nbf=192, max_l=2,
         unrestricted=True, nelec=56, sum_z=56, heavy_z=6,
         norm_offsets=(0, -1, -2)),
    # ORCA small-core ECP: the file carries the FULL nuclear charge
    # (Gd = 64), so sum_z != nelec and the ESP path must subtract the
    # 28-electron core.  max_l = 4 exercises the g-component sign flip.
    dict(name="GdCp3-orca-roks-def2-small.molden",
         software="orca", natoms=31, nbf=673, max_l=4,
         unrestricted=False, nelec=141, sum_z=169, heavy_z=64,
         ecp=True, norm_offsets=(-1, -2, -3)),
    # Gaussian 4f-in-core ECP: the file writes the ECP-reduced charge
    # (Gd -> 11) in the atomic-number column; the loader must still
    # identify Gd from the symbol, giving sum_z = 169.
    dict(name="GdCp3-g16-rks-pople-4fincore.molden",
         software="g16", natoms=31, nbf=392, max_l=2,
         unrestricted=False, nelec=116, sum_z=169, heavy_z=64,
         ecp=True, norm_offsets=(0, -1, -2)),
]

# Gaussian formatted-checkpoint files.  Each has a molden twin in the same
# curated set (same calculation), so the loader can be cross-validated by
# comparing electron densities, which are independent of the orbital
# representation and thus catch coefficient, ordering and normalisation
# errors.  ``sp`` flags files whose Pople basis uses combined SP shells
# (fchk shell type -1, split into separate s and p shells on load).  For
# ECP files the fchk stores the TRUE atomic number in "Atomic numbers"
# (used for element identity, so sum_z counts the full nuclear charge)
# and the ECP-reduced charge separately in "Nuclear charges".
FCHK_FILES = [
    dict(name="water-g16-rks-def2-none.fchk",
         molden_twin="water-g16-rks-def2-none.molden",
         natoms=3, nbf=43, max_l=3, unrestricted=False, nelec=10,
         sum_z=10, heavy_z=8, sp=False, norm_offsets=(0, -1, -2)),
    dict(name="water-g16-rks-pople-none.fchk",
         molden_twin="water-g16-rks-pople-none.molden",
         natoms=3, nbf=30, max_l=2, unrestricted=False, nelec=10,
         sum_z=10, heavy_z=8, sp=True, norm_offsets=(0, -1, -2)),
    dict(name="xylylene-g16-uks-pople-none.fchk",
         molden_twin="xylylene-g16-uks-pople-none.molden",
         natoms=16, nbf=192, max_l=2, unrestricted=True, nelec=56,
         sum_z=56, heavy_z=6, sp=True, norm_offsets=(0, -1, -2)),
    # 4f-in-core ECP: "Atomic numbers" gives Gd = 64 (sum_z = 169) while
    # "Nuclear charges" gives the reduced 11; the element must be Gd.
    dict(name="GdCp3-g16-rks-pople-4fincore.fchk",
         molden_twin="GdCp3-g16-rks-pople-4fincore.molden",
         natoms=31, nbf=392, max_l=2, unrestricted=False, nelec=116,
         sum_z=169, heavy_z=64, ecp=True, sp=True,
         norm_offsets=(0, -1, -2)),
]

# Geometry-only files: both xyz variants (with and without the leading
# atom-count line) plus a heavy-element case.
XYZ_FILES = [
    dict(name="water.xyz", natoms=3, has_count_line=False),
    dict(name="benzene.xyz", natoms=12, has_count_line=True),
    dict(name="DyCp3.xyz", natoms=31, has_count_line=False, heavy_z=66),
]

# Convenient sub-selections for the GUI/render tests.
SMALL_RESTRICTED = "benzene-g16-rks-pople-none.molden"
UNRESTRICTED = "xylylene-orca-uks-x2c-none.molden"
# The render and GUI tests build a polyhedron from a cluster of atoms,
# which needs a genuinely three-dimensional geometry (a planar molecule
# such as benzene degenerates the convex hull).  This restricted,
# three-dimensional complex serves that purpose and keeps the spin-density
# button disabled for the restricted-file assertions.
GEOMETRY_FILE = "GdCp3-g16-rks-pople-4fincore.molden"
