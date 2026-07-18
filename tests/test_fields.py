# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of the field core: density normalisation, spin density on a
synthetic unrestricted system, ESP against the analytic Gaussian result,
and the ECP effective-charge heuristic."""

import os
import sys
import time
import warnings

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=DeprecationWarning)

from kaijo.core.basis import BasisSet, MolecularOrbitals, Shell
from kaijo.core.fields import (effective_nuclear_charges, evaluate_density,
                               evaluate_esp, esp_vertex_colors,
                               grid_cusp_charges, sample_volume)
from kaijo.core.grid import RectilinearGrid
from kaijo.core.molecule import Molecule
from kaijo.formats import load_file

import testdata


def _check(label, value, ok):
    print(f"  {label}: {value} {'OK' if ok else 'FAIL'}")
    return ok


def test_effective_charges():
    """All-electron files pass through; ECP core holes are recovered."""
    ok = True
    # neutral all-electron molecule (H2O): unchanged
    z = effective_nuclear_charges([8, 1, 1], 10.0)
    ok &= _check("all-electron H2O", z.tolist(), z.tolist() == [8, 1, 1])
    # +2 cation, all-electron: still unchanged (plausible charge)
    z = effective_nuclear_charges([26, 8, 8], 40.0)
    ok &= _check("all-electron ion", z.tolist(), z.tolist() == [26, 8, 8])
    # Gd with a 28-electron small-core ECP in a neutral complex; this is
    # the convention of GdCp3-orca-roks-def2-small.molden, where the file
    # carries the full nuclear charge Gd = 64.
    z = effective_nuclear_charges([64, 6, 6, 1, 1], 64 + 14 - 28)
    ok &= _check("Gd ECP28 (def2-small)", z.tolist(),
                 z.tolist() == [36, 6, 6, 1, 1])
    # Dy with a 28-electron ECP core in a neutral complex
    z = effective_nuclear_charges([66, 6, 6, 1, 1], 66 + 14 - 28)
    ok &= _check("Dy ECP28", z.tolist(), z.tolist() == [38, 6, 6, 1, 1])
    # Bi(ECP60) + Dy(ECP28) together
    z = effective_nuclear_charges([83, 66, 6], 83 + 66 + 6 - 60 - 28)
    ok &= _check("Bi ECP60 + Dy ECP28", z.tolist(),
                 z.tolist() == [23, 38, 6])
    return ok


def _s_basis(alpha, center=(0.0, 0.0, 0.0)):
    return Shell(0, 0, center, [alpha], [1.0])


def test_spin_density_synthetic():
    """Two s-type 'orbitals' with alpha/beta spins: the density is the
    sum and the spin density the difference of the two |psi|^2."""
    shells = [_s_basis(0.8, (-1.0, 0, 0)), _s_basis(1.2, (1.0, 0, 0))]
    basis = BasisSet(shells)
    coeffs = np.eye(2)
    mos = MolecularOrbitals(coeffs, [-1.0, -0.9], [1.0, 1.0], [0, 1])
    grid = RectilinearGrid((-6.0, -5.0, -5.0), (0.2, 0.2, 0.2),
                           (61, 51, 51))
    from kaijo.core.basis import evaluate_orbitals
    psi = evaluate_orbitals(basis, coeffs, grid)
    rho = evaluate_density(basis, mos, grid)
    srho = evaluate_density(basis, mos, grid, spin=True)
    ref_rho = psi[0] ** 2 + psi[1] ** 2
    ref_spin = psi[0] ** 2 - psi[1] ** 2
    e1 = float(np.abs(rho - ref_rho).max())
    e2 = float(np.abs(srho - ref_spin).max())
    ok = _check("density = sum |psi|^2", f"max err {e1:.2e}", e1 < 1e-6)
    ok &= _check("spin density = alpha - beta", f"max err {e2:.2e}",
                 e2 < 1e-6)
    dv = float(grid.spacing.prod())
    n = float(rho.astype(np.float64).sum() * dv)
    s = float(srho.astype(np.float64).sum() * dv)
    ok &= _check("integral of density", f"{n:.4f} (expect 2)",
                 abs(n - 2.0) < 0.01)
    ok &= _check("integral of spin density", f"{s:.4f} (expect 0)",
                 abs(s) < 0.01)
    return ok


def test_esp_gaussian():
    """ESP of a normalised Gaussian charge against the analytic
    erf(sqrt(a) r) / r result (electron part only, no nuclei)."""
    a = 1.0
    grid = RectilinearGrid((-8.0, -8.0, -8.0), (0.25, 0.25, 0.25),
                           (65, 65, 65))
    x = grid.axis(0)[:, None, None]
    y = grid.axis(1)[None, :, None]
    z = grid.axis(2)[None, None, :]
    r2 = x * x + y * y + z * z
    rho = ((a / np.pi) ** 1.5 * np.exp(-a * r2)).astype(np.float32)
    mol = Molecule(np.zeros(0, dtype=int), np.zeros((0, 3)))
    v = evaluate_esp(mol, rho, grid)
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.5, 1.5, 1.5],
                    [0.0, 0.0, 4.0]])
    got = sample_volume(v, grid, pts)
    r = np.linalg.norm(pts, axis=1)
    ref = -erf(np.sqrt(a) * r) / r
    err = float(np.abs(got - ref).max())
    ok = _check("Gaussian ESP vs erf(r)/r", f"max abs err {err:.2e} Eh",
                err < 0.01)
    return ok


def test_file_fields(path):
    """Real molden file: density normalisation and ESP sanity."""
    t0 = time.time()
    data = load_file(path)
    print(f"  loaded in {time.time()-t0:.2f} s: "
          f"{data.molecule.natoms} atoms, {data.basis.nbf} basis fns")
    nelec = float(data.orbitals.occupations.sum())
    grid = RectilinearGrid.for_molecule(data.molecule, spacing=0.4,
                                        margin=5.0)
    print(f"  grid: {grid.describe()} ({grid.npoints/1e6:.1f} M points)")
    t0 = time.time()
    rho = evaluate_density(data.basis, data.orbitals, grid)
    print(f"  density evaluated in {time.time()-t0:.2f} s")
    dv = float(grid.spacing.prod())
    n = float(rho.astype(np.float64).sum() * dv)
    # uniform-grid quadrature cannot resolve the nuclear cusps of the
    # all-electron heavy atoms (a sub-0.01-bohr feature), so the
    # integral carries a grid-alignment-dependent error of several
    # percent; evaluate_esp rescales it away, so a loose check suffices.
    ok = _check("density integral", f"{n:.2f} (expect {nelec:.1f})",
                abs(n - nelec) / nelec < 0.15)
    ok &= _check("density non-negative", f"min {rho.min():.2e}",
                 rho.min() > -1e-4)
    # the analytic cusp correction must recover the exact electron count
    t0 = time.time()
    delta = grid_cusp_charges(data.basis, data.orbitals, grid,
                              natoms=data.molecule.natoms)
    nc = n + float(delta.sum())
    ok &= _check("cusp-corrected electron count",
                 f"{nc:.3f} (expect {nelec:.1f}, "
                 f"{time.time()-t0:.2f} s)",
                 abs(nc - nelec) / nelec < 0.005)
    t0 = time.time()
    v = evaluate_esp(data.molecule, rho, grid, nelec=nelec,
                     cusp_charges=delta)
    print(f"  ESP solved in {time.time()-t0:.2f} s")
    ok &= _check("ESP finite", f"range [{v.min():.2f}, {v.max():.2f}] Eh",
                 np.isfinite(v).all())
    # close to a nucleus the (positive) nuclear term must dominate
    iat = int(np.argmax(data.molecule.numbers))
    vn = sample_volume(v, grid, data.molecule.coords[[iat]] + 0.21)[0]
    ok &= _check("ESP positive next to the heaviest nucleus",
                 f"{vn:.2f} Eh", vn > 0)
    # colors: finite, in range, alpha 1
    verts = grid.origin + np.random.default_rng(0).uniform(
        0, 1, (100, 3)) * (np.array(grid.shape) - 1) * grid.spacing
    colors, vmax = esp_vertex_colors(sample_volume(v, grid, verts))
    ok &= _check("ESP colors in [0, 1]",
                 f"range [{colors.min():.2f}, {colors.max():.2f}], "
                 f"scale ±{vmax:.3f} Eh",
                 colors.min() >= 0.0 and colors.max() <= 1.0)
    return ok


if __name__ == "__main__":
    print("Effective nuclear charges (ECP heuristic):")
    ok = test_effective_charges()
    print("Synthetic unrestricted system:")
    ok &= test_spin_density_synthetic()
    print("Gaussian ESP vs analytic:")
    ok &= test_esp_gaussian()
    # Real files: a light all-electron system (clean cusp correction) and
    # an ORCA small-core ECP system (the ESP must balance the reduced
    # electron count against the ECP-corrected nuclear charges).
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [testdata.path("water-g16-rks-def2-none.molden"),
                 testdata.path("GdCp3-orca-roks-def2-small.molden")]
    for f in files:
        print(f"\nMolden file {os.path.basename(f)}:")
        ok &= test_file_fields(f)
    print("\nALL OK" if ok else "\nFAILURES PRESENT")
    sys.exit(0 if ok else 1)
