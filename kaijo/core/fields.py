# SPDX-License-Identifier: GPL-3.0-or-later
"""Scalar fields derived from the wavefunction: electron density, spin
density and the electrostatic potential (ESP).

Density
    rho(r) = sum_i n_i |psi_i(r)|^2.  The batched orbital-grid machinery
    in basis.py is re-used: occupied orbitals are evaluated in
    memory-bounded batches and accumulated in place, so the peak memory
    stays bounded no matter how many orbitals are (fractionally)
    occupied, and the expensive basis evaluation is amortised over each
    batch.  This form also covers natural-orbital files directly.

Spin density
    Same sum with weights +n_i for alpha and -n_i for beta orbitals.
    Only meaningful for unrestricted files; the UI guards against
    restricted input.

ESP
    V(r) = sum_A Z_A / |r - R_A| - int rho(r') / |r - r'| d3r'.
    The electronic part is obtained by convolving the gridded density
    with the 1/r kernel using zero-padded FFTs (a free-space Poisson
    solve).  This is O(N log N) in the number of grid points versus
    O(N * nbf^2) for analytic Gaussian integrals.  Two corrections make
    the result quantitative at visualization grid spacings: the
    per-nucleus quadrature defect of the unresolvable density cusps is
    computed analytically and compensated with point charges (see
    grid_cusp_charges), and the density is rescaled so the total model
    charge is exact (exact monopole / far field).  For files from ECP
    calculations the effective nuclear charges are inferred so nuclei
    and electrons balance.  Because the ESP is smooth, it never needs
    the finest grids; callers may solve it on a coarser grid and
    interpolate (see sample_volume).
"""

import numpy as np

from .basis import evaluate_orbitals

_SQRT_PI = np.sqrt(np.pi)

# Cap the in-memory batch size for the orbital volumes used to build
# densities (float32), matching the pipeline's batching policy.
_BATCH_BYTES = 256 * 1024 * 1024

#: Fixed ESP color scale: red = negative, white = zero, blue = positive
#: potential (the common quantum-chemistry convention).
ESP_COLOR_NEG = np.array([0.85, 0.15, 0.10])
ESP_COLOR_POS = np.array([0.10, 0.25, 0.85])


def evaluate_density(basis, orbitals, grid, spin=False, progress=None,
                     cancel=None, occ_eps=1e-6):
    """Electron (or spin) density on a rectangular grid.

    Returns an (nx, ny, nz) float32 array, or None when cancelled.
    """
    occ = orbitals.occupations
    if spin:
        weights = np.where(orbitals.spins == 0, occ, -occ)
    else:
        weights = occ
    sel = np.nonzero(np.abs(weights) > occ_eps)[0]
    rho = np.zeros(grid.shape, dtype=np.float32)
    if len(sel) == 0:
        return rho
    batch = max(1, int(_BATCH_BYTES // max(grid.nbytes(), 1)))
    nbatch = (len(sel) + batch - 1) // batch
    for bi, b0 in enumerate(range(0, len(sel), batch)):
        chunk = sel[b0:b0 + batch]

        def prog(f, _bi=bi):
            if progress:
                progress((_bi + f) / nbatch)

        vols = evaluate_orbitals(basis, orbitals.coeffs[chunk], grid,
                                 progress=prog, cancel=cancel)
        if vols is None:  # cancelled
            return None
        vols *= vols  # in place: |psi_i|^2
        rho += np.tensordot(weights[chunk].astype(np.float32), vols,
                            axes=1)
    return rho


# --------------------------------------------------------------------------
# Electrostatic potential
# --------------------------------------------------------------------------

#: Plausible ECP core sizes (closed-shell "magic numbers" used by the
#: standard effective-core-potential families).
_ECP_CORES = (2, 10, 18, 28, 36, 46, 54, 60, 68, 78, 92)


def _default_ecp_core(z):
    """Core size of the standard (Stuttgart/def2-style) ECP for Z."""
    if z < 19:
        return 0
    if z <= 36:
        return 10
    if z <= 54:
        return 28
    if z <= 56:
        return 46
    if z <= 71:
        return 28   # lanthanides
    if z <= 86:
        return 60
    return 78


def _core_assignment(z, missing, candidates):
    """Subset-sum over per-atom core choices: at most one entry of
    candidates(z_i) per atom, total as close to `missing` as possible.
    Returns ([(atom, core), ...], deviation)."""
    reachable = {0: []}  # core-electron total -> [(atom, core), ...]
    for i, zi in enumerate(z):
        cores = [c for c in candidates(int(zi)) if 0 < c < zi]
        if not cores:
            continue
        new = dict(reachable)
        for s, assign in reachable.items():
            for c in cores:
                t = s + c
                if t <= missing + 4 and t not in new:
                    new[t] = assign + [(i, c)]
        reachable = new
    s = min(reachable, key=lambda t: abs(missing - t))
    return reachable[s], abs(missing - s)


def effective_nuclear_charges(numbers, nelec):
    """Best-guess effective nuclear charges for the ESP.

    Molden files list the full atomic number even when the calculation
    used effective core potentials, in which case the core electrons are
    absent from the density and the full Z would wreck the ESP.  When
    sum(Z) differs from the electron count by more than a plausible
    molecular charge (|q| <= 4), ECP core sizes are assigned to the
    heavy atoms by a subset-sum search until nuclei and electrons
    balance again -- first trying each element's standard default core,
    then any magic-number core.  All-electron files pass unchanged.
    """
    z = np.asarray(numbers, dtype=np.int64)
    missing = int(round(float(z.sum() - nelec)))
    if abs(missing) <= 4:
        return z.astype(np.float64)
    assign, dev = _core_assignment(
        z, missing, lambda zi: (_default_ecp_core(zi),))
    if dev > 4:
        assign, dev = _core_assignment(z, missing, lambda zi: _ECP_CORES)
    if dev > 4:
        return z.astype(np.float64)  # cannot balance; keep file charges
    zeff = z.astype(np.float64)
    for i, c in assign:
        zeff[i] -= c
    return zeff


# --------------------------------------------------------------------------
# Nuclear-cusp quadrature correction
#
# A uniform grid cannot resolve the density cusp at the nuclei (a
# sub-0.01-bohr feature for heavy atoms), so the grid quadrature of the
# density is off by up to several electrons per heavy atom, which would
# wreck the ESP.  The cusp is built from products of tight one-center
# primitives; for those the difference between the analytic Gaussian
# integral and the actual grid sum separates into three 1-D sums and can
# be computed exactly and cheaply.  The defect is localised well inside
# the core, so it is compensated by point charges at the nuclei.
# --------------------------------------------------------------------------

_MONOMIALS = None
_PAIR_MONO = {}

# Gamma((p+1)/2) for even p = 0, 2, 4, 6, 8: the analytic 1-D moments
# int x^p exp(-a x^2) dx = Gamma((p+1)/2) / a^((p+1)/2).
_GAMMA_HALF = {0: _SQRT_PI, 2: _SQRT_PI / 2.0, 4: 3.0 * _SQRT_PI / 4.0,
               6: 15.0 * _SQRT_PI / 8.0, 8: 105.0 * _SQRT_PI / 16.0}


def _solid_harmonic_monomials():
    """Monomial expansions of the real solid harmonics up to l = 4:
    [l][component] -> [((i, j, k), coeff), ...].  Solid harmonics are
    homogeneous polynomials, so the expansion is exact (fitted once to
    machine precision)."""
    global _MONOMIALS
    if _MONOMIALS is not None:
        return _MONOMIALS
    from .basis import solid_harmonics
    rng = np.random.default_rng(7)
    out = []
    for l in range(5):
        monos = [(i, j, l - i - j) for i in range(l + 1)
                 for j in range(l + 1 - i)]
        pts = rng.normal(size=(2 * len(monos) + 4, 3))
        design = np.stack(
            [pts[:, 0] ** i * pts[:, 1] ** j * pts[:, 2] ** k
             for i, j, k in monos], axis=1)
        comps = []
        for vals in solid_harmonics(l, pts[:, 0], pts[:, 1], pts[:, 2]):
            coef = np.linalg.lstsq(design, np.asarray(vals),
                                   rcond=None)[0]
            comps.append([(m, float(c)) for m, c in zip(monos, coef)
                          if abs(c) > 1e-10])
        out.append(comps)
    _MONOMIALS = out
    return out


def _pair_monomials(l1, l2):
    """Product polynomials of every solid-harmonic component pair:
    [m1][m2] -> [((i, j, k), coeff), ...] with degree l1 + l2."""
    key = (l1, l2)
    if key not in _PAIR_MONO:
        mono = _solid_harmonic_monomials()
        table = []
        for p1 in mono[l1]:
            row = []
            for p2 in mono[l2]:
                d = {}
                for (e1, c1) in p1:
                    for (e2, c2) in p2:
                        k = (e1[0] + e2[0], e1[1] + e2[1], e1[2] + e2[2])
                        d[k] = d.get(k, 0.0) + c1 * c2
                row.append([(k, c) for k, c in d.items()
                            if abs(c) > 1e-12])
            table.append(row)
        _PAIR_MONO[key] = table
    return _PAIR_MONO[key]


def grid_cusp_charges(basis, orbitals, grid, natoms=None, occ_eps=1e-6):
    """Electrons missed (positive) or over-counted (negative) by the
    grid quadrature of the density near each nucleus.

    Only same-center primitive products stiffer than the grid can
    resolve are considered; two-center products are smooth on the grid
    scale.  Returns a (natoms,) float array of the per-atom defects.
    """
    occ = orbitals.occupations
    sel = np.nonzero(np.abs(occ) > occ_eps)[0]
    nat = natoms if natoms is not None else \
        (max((sh.atom for sh in basis.shells), default=-1) + 1)
    delta = np.zeros(nat)
    if len(sel) == 0:
        return delta
    coefs = orbitals.coeffs[sel]
    dmat = (coefs * occ[sel][:, None]).T @ coefs  # AO density matrix
    axes = [grid.axis(i) for i in range(3)]
    h = np.asarray(grid.spacing, dtype=np.float64)
    # products softer than this are integrated exactly by the grid
    # (Poisson summation: the leading error term is ~ exp(-pi^2/(a h^2)))
    a_cut = 0.5 / float(h.max()) ** 2

    by_atom = {}
    for sh in basis.shells:
        by_atom.setdefault(sh.atom, []).append(sh)
    for atom, shells in by_atom.items():
        d = [ax - c for ax, c in zip(axes, shells[0].center)]
        dpow = [np.stack([dd ** p for p in range(9)], axis=1)
                for dd in d]  # (n, 9) per axis
        for i1, sh1 in enumerate(shells):
            for sh2 in shells[i1:]:
                if sh1.exps.max() + sh2.exps.max() < a_cut:
                    continue
                sym = 1.0 if sh2 is sh1 else 2.0
                aa = (sh1.exps[:, None] + sh2.exps[None, :]).ravel()
                cc = (sh1.coeffs[:, None] * sh2.coeffs[None, :]).ravel()
                keep = aa >= a_cut
                if not keep.any():
                    continue
                aa, cc = aa[keep], cc[keep]
                # 1-D grid sums S[axis][pair, power] and moments M
                S = [np.exp(-np.outer(aa, dd * dd)) @ (dp * hx)
                     for dd, dp, hx in zip(d, dpow, h)]
                M = np.zeros((len(aa), 9))
                for p, g in _GAMMA_HALF.items():
                    M[:, p] = g / aa ** ((p + 1) / 2.0)
                block = dmat[sh1.offset:sh1.offset + sh1.nfunc,
                             sh2.offset:sh2.offset + sh2.nfunc]
                pairs = _pair_monomials(sh1.l, sh2.l)
                poly = {}
                for m1 in range(sh1.nfunc):
                    for m2 in range(sh2.nfunc):
                        w = block[m1, m2]
                        if abs(w) < 1e-12:
                            continue
                        for k, c in pairs[m1][m2]:
                            poly[k] = poly.get(k, 0.0) + w * c
                dsum = 0.0
                for (i, j, k), coef in poly.items():
                    ana = M[:, i] * M[:, j] * M[:, k]
                    num = S[0][:, i] * S[1][:, j] * S[2][:, k]
                    dsum += coef * float(cc @ (ana - num))
                delta[atom] += sym * dsum
    return delta


def _voxel_self_potential(spacing):
    """Average of 1/r over one voxel centred at the origin (the r = 0
    element of the discrete Coulomb kernel), by midpoint quadrature."""
    m = 20  # even, so no sample hits r = 0
    t = (np.arange(m) + 0.5) / m - 0.5
    x = (t * spacing[0])[:, None, None]
    y = (t * spacing[1])[None, :, None]
    z = (t * spacing[2])[None, None, :]
    return float((1.0 / np.sqrt(x * x + y * y + z * z)).mean())


def evaluate_esp(molecule, rho, grid, nelec=None, cusp_charges=None,
                 progress=None, cancel=None):
    """Electrostatic potential (Hartree/e) on the grid carrying `rho`.

    rho          : electron density volume on `grid`
    nelec        : exact electron count; when given, rho is rescaled so
                   the total electron charge of the model is exact
    cusp_charges : per-atom grid-quadrature defects of the density (see
                   grid_cusp_charges), compensated as point charges at
                   the nuclei

    Returns an (nx, ny, nz) float32 array, or None when cancelled.
    """
    import scipy.fft as sfft

    sp = np.asarray(grid.spacing, dtype=np.float64)
    voxel = float(np.prod(sp))
    rho = np.asarray(rho, dtype=np.float32)
    tot = float(rho.astype(np.float64).sum()) * voxel
    cusp_total = float(cusp_charges.sum()) if cusp_charges is not None \
        else 0.0
    if nelec is not None and tot > 1e-9:
        # the gridded density carries (nelec - cusp charges) electrons;
        # after the cusp correction this rescale is a tiny touch-up
        rho = rho * np.float32((nelec - cusp_total) / tot)
    else:
        nelec = tot + cusp_total

    if cancel is not None and cancel():
        return None
    if progress:
        progress(0.05)

    # Zero-padded circular convolution with the symmetric 1/r kernel is
    # exactly the free-space (non-periodic) convolution when the padded
    # size is >= 2n - 1 in every dimension.
    n = rho.shape
    pad = [sfft.next_fast_len(2 * s) for s in n]
    d0 = (np.minimum(np.arange(pad[0]), pad[0] - np.arange(pad[0]))
          * sp[0]).astype(np.float32)
    d1 = (np.minimum(np.arange(pad[1]), pad[1] - np.arange(pad[1]))
          * sp[1]).astype(np.float32)
    d2 = (np.minimum(np.arange(pad[2]), pad[2] - np.arange(pad[2]))
          * sp[2]).astype(np.float32)
    r2 = (d0[:, None, None] ** 2 + d1[None, :, None] ** 2
          + d2[None, None, :] ** 2)
    with np.errstate(divide="ignore"):
        kernel = 1.0 / np.sqrt(r2)
    del r2
    kernel[0, 0, 0] = _voxel_self_potential(sp)

    if cancel is not None and cancel():
        return None
    fk = sfft.rfftn(kernel, s=pad)
    del kernel
    if progress:
        progress(0.35)
    if cancel is not None and cancel():
        return None
    fr = sfft.rfftn(rho, s=pad)
    fk *= fr
    del fr
    if progress:
        progress(0.55)
    if cancel is not None and cancel():
        return None
    v = sfft.irfftn(fk, s=pad)[:n[0], :n[1], :n[2]]
    del fk
    v = -voxel * v.astype(np.float64)  # electrons carry negative charge
    if progress:
        progress(0.75)

    # Nuclear part: direct sum over (effective) point charges, folding
    # in the cusp-defect compensation.  Distances are clipped at half a
    # grid spacing; the surface being colored is far from the nuclei.
    zeff = effective_nuclear_charges(molecule.numbers, nelec)
    if cusp_charges is not None:
        zeff = zeff - cusp_charges
    ax, ay, az = grid.axis(0), grid.axis(1), grid.axis(2)
    rmin2 = (0.5 * float(sp.min())) ** 2
    for ia, (zi, ri) in enumerate(zip(zeff, molecule.coords)):
        if zi == 0.0:
            continue
        if cancel is not None and ia % 16 == 0 and cancel():
            return None
        dsq = ((ax - ri[0]) ** 2)[:, None, None] \
            + ((ay - ri[1]) ** 2)[None, :, None] \
            + ((az - ri[2]) ** 2)[None, None, :]
        v += zi / np.sqrt(np.maximum(dsq, rmin2))
    if progress:
        progress(1.0)
    return v.astype(np.float32)


# --------------------------------------------------------------------------
# Sampling and the fixed ESP color scale
# --------------------------------------------------------------------------

def sample_volume(volume, grid, points):
    """Trilinear interpolation of a gridded volume at world points."""
    from scipy.ndimage import map_coordinates
    idx = (np.asarray(points, np.float64) - grid.origin) / grid.spacing
    return map_coordinates(np.asarray(volume, np.float64), idx.T,
                           order=1, mode="nearest")


def esp_vertex_colors(values):
    """Map ESP values to the fixed red-white-blue diverging scale.

    Returns ((n, 4) float32 RGBA with alpha 1, symmetric range limit).
    The range adapts to the surface (99th percentile of |V|, robust to
    the few vertices closest to a nucleus); the colors themselves are
    fixed and not user-configurable.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return np.zeros((0, 4), np.float32), 0.0
    vmax = float(np.percentile(np.abs(v), 99.0))
    if vmax < 1e-12:
        vmax = 1e-12
    t = np.clip(v / vmax, -1.0, 1.0)
    tp = np.clip(t, 0.0, 1.0)[:, None]
    tn = np.clip(-t, 0.0, 1.0)[:, None]
    white = np.ones(3)
    rgb = white - tp * (white - ESP_COLOR_POS) - tn * (white - ESP_COLOR_NEG)
    out = np.empty((len(v), 4), dtype=np.float32)
    out[:, :3] = rgb
    out[:, 3] = 1.0
    return out, vmax
