# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of the math core: solid harmonics against scipy, and molden
orbital norms on a grid for the curated real-file set (see testdata.py).

Run without arguments to sweep the whole bundled set; pass explicit molden
paths to check those instead.
"""

import os
import sys
import time
import warnings

import numpy as np
from scipy.special import sph_harm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from kaijo.core.basis import solid_harmonics, evaluate_orbitals
from kaijo.core.grid import RectilinearGrid
from kaijo.formats import load_file

import testdata


def real_ylm_scipy(l, m, theta, phi):
    """Real spherical harmonics from scipy (Condon-Shortley included)."""
    if m == 0:
        return sph_harm(0, l, phi, theta).real
    y = sph_harm(abs(m), l, phi, theta)
    if m > 0:
        return np.sqrt(2.0) * (-1) ** m * y.real
    return np.sqrt(2.0) * (-1) ** m * y.imag


def test_harmonics():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(200, 3))
    x, y, z = pts.T
    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    # molden order m sequence per l
    order = {0: [0], 1: [1, -1, 0], 2: [0, 1, -1, 2, -2],
             3: [0, 1, -1, 2, -2, 3, -3],
             4: [0, 1, -1, 2, -2, 3, -3, 4, -4]}
    ok = True
    for l in range(5):
        mine = solid_harmonics(l, x, y, z)
        for comp, m in enumerate(order[l]):
            ref = real_ylm_scipy(l, m, theta, phi) * r ** l
            err = np.abs(mine[comp] - ref).max() / max(np.abs(ref).max(), 1e-300)
            status = "OK" if err < 1e-10 else "FAIL"
            if err >= 1e-10:
                ok = False
            print(f"  l={l} m={m:+d}: rel err {err:.2e} {status}")
    return ok


def test_molden(path, spec=None):
    """Load a molden file, verify its header-derived properties, and check
    the grid norm of a few diffuse valence orbitals."""
    t0 = time.time()
    data = load_file(path)
    t1 = time.time()
    mol, basis, mos = data.molecule, data.basis, data.orbitals
    print(f"  loaded in {t1-t0:.2f} s: {mol.natoms} atoms, "
          f"{basis.nbf} basis fns, {mos.nmo} MOs, max_l={basis.max_l()}, "
          f"unrestricted={mos.unrestricted}")
    ok = True

    if spec is not None:
        checks = [
            ("natoms", mol.natoms, spec["natoms"]),
            ("nbf", basis.nbf, spec["nbf"]),
            ("max_l", basis.max_l(), spec["max_l"]),
            ("unrestricted", mos.unrestricted, spec["unrestricted"]),
            ("sum_z", int(mol.numbers.sum()), spec["sum_z"]),
            ("nelec", round(float(mos.occupations.sum())), spec["nelec"]),
            ("heavy_z", int(mol.numbers.max()), spec["heavy_z"]),
        ]
        for label, got, want in checks:
            good = got == want
            ok &= good
            print(f"    {label}: {got} (expect {want}) "
                  f"{'OK' if good else 'FAIL'}")
        offsets = spec["norm_offsets"]
    else:
        offsets = (0, -1, -2)

    homo = mos.homo_index()
    pick = [homo + o for o in offsets]
    grid = RectilinearGrid.for_molecule(mol, spacing=0.2, margin=6.0)
    print(f"  grid: {grid.describe()} ({grid.npoints/1e6:.1f} M points)")
    t0 = time.time()
    vols = evaluate_orbitals(basis, mos.coeffs[pick], grid)
    t1 = time.time()
    dv = grid.spacing.prod()
    print(f"  evaluated {len(pick)} orbitals in {t1-t0:.2f} s")
    for p, v in zip(pick, vols):
        norm = float((v.astype(np.float64) ** 2).sum() * dv)
        good = abs(norm - 1) < 0.02
        ok &= good
        print(f"  MO {mos.name(p)} (E={mos.energies[p]:.4f}, "
              f"occ={mos.occupations[p]:.3f}): <psi|psi> = {norm:.4f} "
              f"{'OK' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("Solid harmonics vs scipy:")
    ok = test_harmonics()
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            print(f"\nMolden file {f}:")
            ok &= test_molden(f)
    else:
        for spec in testdata.MOLDEN_FILES:
            print(f"\n{spec['name']} [{spec['software']}]:")
            ok &= test_molden(testdata.path(spec["name"]), spec)
    print("\nALL OK" if ok else "\nFAILURES PRESENT")
    sys.exit(0 if ok else 1)
