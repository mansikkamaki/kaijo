# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian-type orbital basis and molecular-orbital evaluation on grids.

Supports spherical-harmonic (pure) basis functions up to l = 4 (g), which
covers the [5D]/[7F]/[9G] molden convention.  Evaluation is organised for
speed:

  * shells are evaluated only inside the radius where they are non-negligible
    (sub-box screening on the rectangular grid);
  * the radial part is evaluated once per shell and reused for all 2l+1
    angular components;
  * a whole batch of orbitals is evaluated in one sweep so the (expensive)
    basis evaluation is amortised over all requested orbitals;
  * shells whose coefficients are all negligible in the batch are skipped.
"""

import re

import numpy as np

_SQRT_PI = np.sqrt(np.pi)

# Strip the leading orbital number a program may pack into a molden Sym=
# string ("1a1" -> "a1"); the remainder is the symmetry-species label.
_SYM_LABEL_RE = re.compile(r"^\s*\d*\s*(.*?)\s*$")


def _symmetry_species(label):
    """Symmetry-species label from a raw molden Sym= string.

    Molden writes the species prefixed by a running number ("1a", "23b1u");
    only the species part is kept.  When no symmetry information is present
    the label defaults to "a", the totally symmetric irrep of the C1 group.
    """
    if label:
        m = _SYM_LABEL_RE.match(label)
        species = (m.group(1) if m else label).strip()
        if species:
            return species
    return "a"

L_LABELS = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

_DFACT = {0: 1.0, 1: 1.0, 2: 3.0, 3: 15.0, 4: 105.0, 5: 945.0}  # (2n-1)!!


def _primitive_norm(alpha, l):
    """Norm of r^l exp(-alpha r^2) against an orthonormal real Y_lm."""
    df = _DFACT.get(l + 1)  # (2l+1)!!
    return np.sqrt(2.0 ** (l + 2) * (2.0 * alpha) ** (l + 1.5)
                   / (df * _SQRT_PI))


class Shell:
    """One contracted shell of 2l+1 spherical components."""

    __slots__ = ("atom", "l", "center", "exps", "coeffs", "offset", "rcut")

    def __init__(self, atom, l, center, exps, coeffs, coeffs_are_raw=False):
        """coeffs_are_raw: True if the contraction coefficients already
        include the primitive normalisation factors (ORCA molden files);
        False for the molden standard (coefficients w.r.t. normalised
        primitives).  Either way the contracted shell is renormalised."""
        self.atom = atom
        self.l = l
        self.center = np.asarray(center, dtype=np.float64)
        self.exps = np.asarray(exps, dtype=np.float64)
        raw = np.asarray(coeffs, dtype=np.float64)
        if coeffs_are_raw:
            c = raw.copy()
        else:
            c = raw * _primitive_norm(self.exps, l)
        dfp = _DFACT[l + 1]
        aij = self.exps[:, None] + self.exps[None, :]
        radint = dfp * _SQRT_PI / (2.0 ** (l + 2) * aij ** (l + 1.5))
        s = float(c @ radint @ c)
        if s > 0:
            c /= np.sqrt(s)
        self.coeffs = c
        self.offset = 0  # first basis-function index; set by BasisSet
        self.rcut = self._cutoff_radius()

    @property
    def nfunc(self):
        return 2 * self.l + 1

    def _cutoff_radius(self, eps=1e-8):
        """Radius beyond which |radial part| < eps (found numerically)."""
        r = np.linspace(0.0, 40.0, 400)
        rad = np.abs(self.radial(r * r)) * np.maximum(r, 1.0) ** self.l
        nz = np.nonzero(rad > eps)[0]
        if len(nz) == 0:
            return 1.0
        return float(r[min(nz[-1] + 1, len(r) - 1)]) + 0.25

    def radial(self, r2):
        """Contracted radial factor (without the r^l, which lives in the
        solid-harmonic polynomials) on an array of squared distances."""
        out = self.coeffs[0] * np.exp(-self.exps[0] * r2)
        for a, c in zip(self.exps[1:], self.coeffs[1:]):
            out += c * np.exp(-a * r2)
        return out


class BasisSet:
    def __init__(self, shells):
        self.shells = shells
        off = 0
        for sh in shells:
            sh.offset = off
            off += sh.nfunc
        self.nbf = off

    def max_l(self):
        return max((sh.l for sh in self.shells), default=0)


class MolecularOrbitals:
    """MO coefficient matrix plus per-orbital metadata.

    Orbitals are stored in file order; `display_order()` gives the order
    required by the UI: occupation descending, then energy ascending, with
    alpha and beta interleaved.

    Two 1-based indices are assigned to every orbital, both counted in file
    order:

      * ``spin_index`` -- position within the orbital's own spin channel
        (the order the file lists them in);
      * ``sym_index``  -- position within the orbital's own (spin, symmetry
        species) group.

    ``sym_species`` holds the parsed symmetry label ("a" when the file
    carries no symmetry information).
    """

    def __init__(self, coeffs, energies, occupations, spins, labels=None):
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.energies = np.asarray(energies, dtype=np.float64)
        self.occupations = np.asarray(occupations, dtype=np.float64)
        self.spins = np.asarray(spins, dtype=np.int8)  # 0 alpha, 1 beta
        self.labels = labels or [""] * len(self.energies)
        self.sym_species = [_symmetry_species(l) for l in self.labels]
        n = len(self.energies)
        self.spin_index = np.zeros(n, dtype=np.int32)
        self.sym_index = np.zeros(n, dtype=np.int32)
        spin_counts = {0: 0, 1: 0}
        sym_counts = {}
        for i in range(n):
            s = int(self.spins[i])
            spin_counts[s] += 1
            self.spin_index[i] = spin_counts[s]
            key = (s, self.sym_species[i])
            sym_counts[key] = sym_counts.get(key, 0) + 1
            self.sym_index[i] = sym_counts[key]

    @property
    def nmo(self):
        return len(self.energies)

    @property
    def unrestricted(self):
        return bool((self.spins == 1).any())

    def display_order(self):
        keys = np.lexsort((self.energies, -self.occupations))
        return keys

    def name(self, i):
        """Per-spin file-order name, e.g. '42a' / '42b' / '42'."""
        n = str(self.spin_index[i])
        if self.unrestricted:
            n += "ab"[self.spins[i]]
        return n

    def sym_name(self, i):
        """Symmetry species with its per-(spin, species) index, e.g. '3a'."""
        return f"{self.sym_index[i]}{self.sym_species[i]}"

    def homo_index(self):
        """Highest-energy (at least half-)occupied orbital.  The 0.5
        threshold keeps the label sensible for natural-orbital files with
        small fractional occupations."""
        occ = np.nonzero(self.occupations >= 0.5)[0]
        if len(occ) == 0:
            occ = np.nonzero(self.occupations > 1e-6)[0]
        if len(occ) == 0:
            return None
        return int(occ[np.argmax(self.energies[occ])])


# --------------------------------------------------------------------------
# Real solid harmonics (orthonormal real Y_lm times r^l) in molden order.
# Molden component order: p: x, y, z;  d/f/g: m = 0, +1, -1, +2, -2, ...
# --------------------------------------------------------------------------

def solid_harmonics(l, x, y, z):
    """Return the 2l+1 solid-harmonic polynomials in molden order."""
    if l == 0:
        one = np.broadcast_to(np.float64(0.28209479177387814), x.shape)
        return [one]
    if l == 1:
        c = 0.4886025119029199
        return [c * x, c * y, c * z]
    x2, y2, z2 = x * x, y * y, z * z
    if l == 2:
        r2 = x2 + y2 + z2
        return [
            0.31539156525252005 * (3.0 * z2 - r2),
            1.0925484305920792 * x * z,
            1.0925484305920792 * y * z,
            0.5462742152960396 * (x2 - y2),
            1.0925484305920792 * x * y,
        ]
    if l == 3:
        return [
            0.3731763325901154 * z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2),
            0.4570457994644658 * x * (4.0 * z2 - x2 - y2),
            0.4570457994644658 * y * (4.0 * z2 - x2 - y2),
            1.4453057213202771 * z * (x2 - y2),
            2.8906114426405543 * x * y * z,
            0.5900435899266435 * x * (x2 - 3.0 * y2),
            0.5900435899266435 * y * (3.0 * x2 - y2),
        ]
    if l == 4:
        r2 = x2 + y2 + z2
        r4 = r2 * r2
        return [
            0.10578554691520431 * (35.0 * z2 * z2 - 30.0 * z2 * r2 + 3.0 * r4),
            0.6690465435572892 * x * z * (7.0 * z2 - 3.0 * r2),
            0.6690465435572892 * y * z * (7.0 * z2 - 3.0 * r2),
            0.47308734787878004 * (x2 - y2) * (7.0 * z2 - r2),
            0.9461746957575601 * x * y * (7.0 * z2 - r2),
            1.7701307697799304 * x * z * (x2 - 3.0 * y2),
            1.7701307697799304 * y * z * (3.0 * x2 - y2),
            0.6258357354491761 * (x2 * (x2 - 3.0 * y2)
                                  - y2 * (3.0 * x2 - y2)),
            2.5033429417967046 * x * y * (x2 - y2),
        ]
    raise ValueError(f"Angular momentum l={l} not supported (max is g)")


# --------------------------------------------------------------------------
# Batch evaluation of orbitals on a rectangular grid
# --------------------------------------------------------------------------

def evaluate_orbitals(basis, mo_coeffs, grid, progress=None, cancel=None,
                      coef_eps=1e-10):
    """Evaluate a batch of molecular orbitals on a rectangular grid.

    basis      : BasisSet
    mo_coeffs  : (k, nbf) coefficient rows
    grid       : RectilinearGrid (must expose axes x, y, z)
    progress   : optional callable(fraction) called from this thread
    cancel     : optional callable() -> bool; when True, abort and return None

    Returns (k, nx, ny, nz) float32 array of orbital amplitudes.
    """
    mo = np.atleast_2d(np.asarray(mo_coeffs, dtype=np.float64))
    k = mo.shape[0]
    ax, ay, az = grid.axis(0), grid.axis(1), grid.axis(2)
    vol = np.zeros((k, len(ax), len(ay), len(az)), dtype=np.float32)

    nsh = len(basis.shells)
    for ish, sh in enumerate(basis.shells):
        if cancel is not None and cancel():
            return None
        block = mo[:, sh.offset:sh.offset + sh.nfunc]
        cmax = np.abs(block).max() if block.size else 0.0
        if cmax < coef_eps:
            if progress:
                progress((ish + 1) / nsh)
            continue
        cx, cy, cz = sh.center
        rc = sh.rcut
        i0, i1 = np.searchsorted(ax, (cx - rc, cx + rc))
        j0, j1 = np.searchsorted(ay, (cy - rc, cy + rc))
        k0, k1 = np.searchsorted(az, (cz - rc, cz + rc))
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            if progress:
                progress((ish + 1) / nsh)
            continue
        dx = (ax[i0:i1] - cx)[:, None, None]
        dy = (ay[j0:j1] - cy)[None, :, None]
        dz = (az[k0:k1] - cz)[None, None, :]
        r2 = dx * dx + dy * dy + dz * dz
        rad = sh.radial(r2)
        polys = solid_harmonics(sh.l, dx, dy, dz)
        shape = r2.shape
        npts = r2.size
        chi = np.empty((sh.nfunc, npts))
        for m, poly in enumerate(polys):
            chi[m] = (np.broadcast_to(poly, shape).reshape(npts)
                      * rad.reshape(npts))
        contrib = block @ chi  # (k, npts)
        vol[:, i0:i1, j0:j1, k0:k1] += contrib.reshape(
            (k,) + shape).astype(np.float32)
        if progress:
            progress((ish + 1) / nsh)
    return vol
