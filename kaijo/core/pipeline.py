# SPDX-License-Identifier: GPL-3.0-or-later
"""Background compute pipeline: orbital volumes -> isosurface meshes.

Volumes are cached on the scratch disk keyed by orbital and grid, so
changing only the isovalue re-runs marching cubes (fast) without
re-evaluating the orbitals (slow).  All heavy work happens in a worker
thread; callbacks are invoked from that thread and the UI layer is
responsible for marshalling them to the main loop.
"""

import hashlib
import threading

import numpy as np

from . import fields, isosurface
from .basis import evaluate_orbitals
from .grid import make_grid

# Cap the in-memory batch size for orbital evaluation (float32 volumes).
_BATCH_BYTES = 256 * 1024 * 1024

# The ESP is smooth, so the Poisson solve never needs a grid finer than
# this; on finer carrier grids the ESP is solved on a coarsened grid and
# interpolated onto the surface vertices.
_ESP_SOURCE_SPACING = 0.25

FIELD_LABELS = {"density": "Electron density", "spin": "Spin density",
                "esp": "ESP"}


class GridParams:
    """User-facing grid options; hashable so volumes can be cache-keyed."""

    def __init__(self, kind="rectangular", spacing=0.35, margin=4.0):
        self.kind = kind
        self.spacing = float(spacing)
        self.margin = float(margin)

    def signature(self):
        s = f"{self.kind}:{self.spacing:.6f}:{self.margin:.6f}"
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def build(self, molecule):
        return make_grid(self.kind, molecule,
                         spacing=self.spacing, margin=self.margin)


def compute_field_payload(data, scratch, kind, gp, isovalue, progress=None,
                          cancel=None, key_prefix="vol"):
    """Compute the display payload for one scalar field (synchronous).

    kind : 'density' | 'spin' | 'esp'
    Returns {'mesh_pos', 'mesh_neg'} for the densities and additionally
    {'esp_colors', 'esp_range'} for the ESP (which is the fixed
    red-white-blue potential map on the density isosurface), or None
    when cancelled.  Volumes are cached in the scratch directory keyed
    by field and grid signature, so isovalue-only changes are fast.
    """
    def prog(f, msg):
        if progress:
            progress(f, msg)

    grid = gp.build(data.molecule)

    def density_volume(spin, gpx, gridx, f0, f1, msg):
        key = f"{key_prefix}_{'spin' if spin else 'density'}_" \
              f"{gpx.signature()}"
        vol = scratch.load_volume(key)
        if vol is None:
            vol = fields.evaluate_density(
                data.basis, data.orbitals, gridx, spin=spin,
                progress=lambda f: prog(f0 + f * (f1 - f0), msg),
                cancel=cancel)
            if vol is None:
                return None
            scratch.save_volume(key, vol)
        return vol

    if kind in ("density", "spin"):
        msg = ("Evaluating spin density on grid..." if kind == "spin"
               else "Evaluating electron density on grid...")
        vol = density_volume(kind == "spin", gp, grid, 0.0, 0.95, msg)
        if vol is None:
            return None
        prog(0.97, "Extracting isosurfaces...")
        pos, neg = isosurface.extract_pair(vol, isovalue, grid)
        return {"mesh_pos": pos, "mesh_neg": neg}

    # ESP: potential mapped onto the density isosurface.  The carrier
    # surface uses the requested grid; the Poisson solve runs on a
    # coarser source grid when the carrier grid is finer than needed.
    vol = density_volume(False, gp, grid, 0.0, 0.45,
                         "Evaluating electron density on grid...")
    if vol is None:
        return None
    if gp.spacing >= _ESP_SOURCE_SPACING - 1e-9:
        gp_src, grid_src, rho_src = gp, grid, vol
    else:
        gp_src = GridParams(gp.kind, _ESP_SOURCE_SPACING, gp.margin)
        grid_src = gp_src.build(data.molecule)
        rho_src = density_volume(False, gp_src, grid_src, 0.45, 0.6,
                                 "Evaluating ESP source density...")
        if rho_src is None:
            return None
    esp_key = f"{key_prefix}_esp_{gp_src.signature()}"
    vesp = scratch.load_volume(esp_key)
    if vesp is None:
        prog(0.6, "Solving electrostatic potential...")
        cusp = fields.grid_cusp_charges(data.basis, data.orbitals,
                                        grid_src,
                                        natoms=data.molecule.natoms)
        vesp = fields.evaluate_esp(
            data.molecule, rho_src, grid_src,
            nelec=float(data.orbitals.occupations.sum()),
            cusp_charges=cusp,
            progress=lambda f: prog(0.62 + f * 0.33,
                                    "Solving electrostatic potential..."),
            cancel=cancel)
        if vesp is None:
            return None
        scratch.save_volume(esp_key, vesp)
    prog(0.97, "Extracting the density surface...")
    carrier = isosurface.extract(vol, abs(isovalue), grid)
    if carrier.empty:
        return {"mesh_pos": carrier, "mesh_neg": None,
                "esp_colors": None, "esp_range": 0.0}
    values = fields.sample_volume(vesp, grid_src, carrier.vertices)
    colors, vmax = fields.esp_vertex_colors(values)
    return {"mesh_pos": carrier, "mesh_neg": None,
            "esp_colors": colors, "esp_range": vmax}


class OrbitalPipeline:
    def __init__(self, scratch):
        self.scratch = scratch
        self.data = None           # LoadedData
        self._thread = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def set_system(self, data):
        self.cancel()
        self.data = data
        self.scratch.clear_volumes()

    def _vol_key(self, orb, gp):
        return f"vol_{orb}_{gp.signature()}"

    # ------------------------------------------------------------------ jobs

    def cancel(self):
        self._cancel.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join()

    def request_cancel(self):
        """Signal the worker to stop, without blocking on it (unlike
        cancel(), which joins).  The worker still reports completion
        through its on_done callback."""
        self._cancel.set()

    def is_running(self):
        t = self._thread
        return t is not None and t.is_alive()

    def compute_surfaces(self, orb_indices, grid_params, isovalue,
                         on_progress, on_surface, on_done):
        """Compute isosurface meshes for the given orbitals (async).

        on_progress(fraction, message)
        on_surface(orb_index, mesh_pos, mesh_neg)   (per orbital, in order)
        on_done(error_or_None)
        All callbacks fire on the worker thread.
        """
        self.cancel()
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            args=(list(orb_indices), grid_params, float(isovalue),
                  on_progress, on_surface, on_done))
        self._thread.start()

    def compute_fields(self, jobs, on_progress, on_result, on_done):
        """Compute density/spin/ESP surfaces (async).

        jobs: list of (kind, grid_params, isovalue) with kind one of
        'density' | 'spin' | 'esp'; processed in order in one worker.
        on_progress(fraction, message)
        on_result(kind, payload)      (see compute_field_payload)
        on_done(error_or_None)
        All callbacks fire on the worker thread.
        """
        self.cancel()
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run_fields, daemon=True,
            args=(list(jobs), on_progress, on_result, on_done))
        self._thread.start()

    def _run_fields(self, jobs, on_progress, on_result, on_done):
        try:
            cancel = self._cancel.is_set
            ntot = len(jobs)
            for nj, (kind, gp, isovalue) in enumerate(jobs):
                if cancel():
                    on_done(None)
                    return

                def prog(f, msg, _n=nj):
                    if not cancel():
                        on_progress((_n + f) / ntot, msg)

                payload = compute_field_payload(
                    self.data, self.scratch, kind, gp, float(isovalue),
                    progress=prog, cancel=cancel)
                if payload is None:  # cancelled
                    on_done(None)
                    return
                on_result(kind, payload)
            on_done(None)
        except Exception as exc:  # noqa: BLE001 - report to UI
            on_done(exc)

    def _run(self, orbs, gp, isovalue, on_progress, on_surface, on_done):
        try:
            grid = gp.build(self.data.molecule)
            cached = [o for o in orbs
                      if self.scratch.has_volume(self._vol_key(o, gp))]
            missing = [o for o in orbs if o not in cached]

            # marching cubes for already-cached volumes first (fast feedback)
            ntot = len(orbs)
            ndone = 0
            for o in cached:
                if self._cancel.is_set():
                    on_done(None)
                    return
                vol = self.scratch.load_volume(self._vol_key(o, gp))
                pos, neg = isosurface.extract_pair(vol, isovalue, grid)
                on_surface(o, pos, neg)
                ndone += 1
                on_progress(ndone / ntot, "Extracting isosurfaces...")

            if missing:
                batch = max(1, int(_BATCH_BYTES // max(grid.nbytes(), 1)))
                mos = self.data.orbitals
                for b0 in range(0, len(missing), batch):
                    chunk = missing[b0:b0 + batch]
                    frac0 = ndone / ntot

                    def prog(f, _f0=frac0, _n=len(chunk)):
                        if not self._cancel.is_set():
                            on_progress(_f0 + f * _n / ntot * 0.9,
                                        "Evaluating orbitals on grid...")

                    vols = evaluate_orbitals(
                        self.data.basis, mos.coeffs[chunk], grid,
                        progress=prog, cancel=self._cancel.is_set)
                    if vols is None:  # cancelled
                        on_done(None)
                        return
                    for o, vol in zip(chunk, vols):
                        if self._cancel.is_set():
                            on_done(None)
                            return
                        self.scratch.save_volume(self._vol_key(o, gp), vol)
                        pos, neg = isosurface.extract_pair(vol, isovalue,
                                                           grid)
                        on_surface(o, pos, neg)
                        ndone += 1
                        on_progress(ndone / ntot,
                                    "Extracting isosurfaces...")
            on_done(None)
        except Exception as exc:  # noqa: BLE001 - report to UI
            on_done(exc)
