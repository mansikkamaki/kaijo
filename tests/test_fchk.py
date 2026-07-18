# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of the Gaussian formatted-checkpoint (.fchk) loader.

For every bundled fchk file this checks:

  * the header-derived properties (atom/basis counts, angular momentum,
    restricted/unrestricted, electron count and -- importantly for the ECP
    files -- the element identity via the true atomic number);
  * the grid norm <psi|psi> ~ 1 of a few diffuse valence orbitals;
  * that the electron density matches the molden file of the *same*
    calculation to grid precision.  The density is independent of the
    orbital representation, so a match certifies that the basis, the
    contraction normalisation, the spherical-harmonic component order and
    the occupation assignment are all correct.

It also checks that Cartesian d/f/g shells are rejected with a clear error.

Run without arguments to sweep the bundled set; pass explicit fchk paths to
check those instead (the cross-check is skipped for unknown files).
"""

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from kaijo.core.basis import evaluate_orbitals
from kaijo.core.fields import evaluate_density
from kaijo.core.grid import RectilinearGrid
from kaijo.formats import load_file
from kaijo.formats.fchk import _Fchk, _build_basis

import testdata


def _check(label, got, want):
    good = got == want
    print(f"    {label}: {got} (expect {want}) {'OK' if good else 'FAIL'}")
    return good


def test_properties(data, spec):
    mol, basis, mos = data.molecule, data.basis, data.orbitals
    ok = True
    ok &= _check("natoms", mol.natoms, spec["natoms"])
    ok &= _check("nbf", basis.nbf, spec["nbf"])
    ok &= _check("max_l", basis.max_l(), spec["max_l"])
    ok &= _check("unrestricted", mos.unrestricted, spec["unrestricted"])
    ok &= _check("sum_z", int(mol.numbers.sum()), spec["sum_z"])
    ok &= _check("nelec", round(float(mos.occupations.sum())), spec["nelec"])
    ok &= _check("heavy_z", int(mol.numbers.max()), spec["heavy_z"])
    return ok


def test_norms(data, offsets):
    mol, basis, mos = data.molecule, data.basis, data.orbitals
    homo = mos.homo_index()
    pick = [homo + o for o in offsets]
    grid = RectilinearGrid.for_molecule(mol, spacing=0.2, margin=6.0)
    vols = evaluate_orbitals(basis, mos.coeffs[pick], grid)
    dv = grid.spacing.prod()
    ok = True
    for p, v in zip(pick, vols):
        norm = float((v.astype(np.float64) ** 2).sum() * dv)
        good = abs(norm - 1) < 0.02
        ok &= good
        print(f"    MO {mos.name(p)} <psi|psi> = {norm:.4f} "
              f"{'OK' if good else 'FAIL'}")
    return ok


def test_density_vs_molden(data, twin_path):
    """The fchk and its molden twin describe the same calculation, so their
    densities must agree to grid precision."""
    twin = load_file(twin_path)
    grid = RectilinearGrid.for_molecule(data.molecule, spacing=0.3,
                                        margin=4.0)
    rho_f = evaluate_density(data.basis, data.orbitals, grid).astype(
        np.float64)
    rho_m = evaluate_density(twin.basis, twin.orbitals, grid).astype(
        np.float64)
    rel = float(np.abs(rho_f - rho_m).max() / max(rho_m.max(), 1e-12))
    good = rel < 1e-3
    print(f"    density vs molden twin: rel max diff {rel:.2e} "
          f"{'OK' if good else 'FAIL'}")
    return good


def test_cartesian_rejected():
    """A doctored fchk with a Cartesian d shell (type +2) must be rejected."""
    path = testdata.path("water-g16-rks-pople-none.fchk")
    with open(path, errors="replace") as fh:
        text = fh.read()
    doctored = text.replace("          -2", "           2", 1)
    fchk = _Fchk(doctored)
    coords = fchk.array("Current cartesian coordinates").reshape(-1, 3)
    try:
        _build_basis(fchk, coords)
    except ValueError as e:
        print(f"  Cartesian shell rejected: OK ({str(e)[:48]}...)")
        return True
    print("  Cartesian shell NOT rejected: FAIL")
    return False


def run_file(path, spec=None):
    t0 = time.time()
    data = load_file(path)
    mol, basis, mos = data.molecule, data.basis, data.orbitals
    print(f"  loaded in {time.time()-t0:.2f} s: {mol.natoms} atoms, "
          f"{basis.nbf} basis fns, {mos.nmo} MOs, max_l={basis.max_l()}, "
          f"unrestricted={mos.unrestricted}")
    ok = True
    if spec is not None:
        ok &= test_properties(data, spec)
        ok &= test_norms(data, spec["norm_offsets"])
        ok &= test_density_vs_molden(
            data, testdata.path(spec["molden_twin"]))
    else:
        ok &= test_norms(data, (0, -1, -2))
    return ok


if __name__ == "__main__":
    print("Cartesian-shell guard:")
    ok = test_cartesian_rejected()
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            print(f"\nfchk file {f}:")
            ok &= run_file(f)
    else:
        for spec in testdata.FCHK_FILES:
            print(f"\n{spec['name']}:")
            ok &= run_file(testdata.path(spec["name"]), spec)
    print("\nALL OK" if ok else "\nFAILURES PRESENT")
    sys.exit(0 if ok else 1)
