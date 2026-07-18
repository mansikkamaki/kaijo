# SPDX-License-Identifier: GPL-3.0-or-later
"""Image export: options dialog and the (self-running) export job.

Orbital surfaces are recomputed at export grid quality in a worker thread;
GL rendering and PNG writing happen on the main loop between computations,
so the UI stays responsive but the export needs no user intervention once
started.
"""

import os
import threading

import numpy as np

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk
from PIL import Image, ImageChops

from ..core import isosurface
from ..core.basis import evaluate_orbitals
from ..core.pipeline import GridParams, compute_field_payload

_MAX_PIXELS = 64_000_000   # memory guard for the render target
_MAX_DIM = 16384           # GL renderbuffer limit guard
_MIN_DIM = 64              # smallest sensible image dimension


class ExportDialog(Gtk.Dialog):
    def __init__(self, parent, settings, base_size, n_items):
        super().__init__(title=f"Export {n_items} image(s)",
                         transient_for=parent, modal=True)
        self.settings = settings
        self.base_w, self.base_h = base_size
        self.add_buttons("Cancel", Gtk.ResponseType.CANCEL,
                         "Export", Gtk.ResponseType.OK)
        self.set_default_response(Gtk.ResponseType.OK)
        ex = settings["export"]

        g = Gtk.Grid(column_spacing=8, row_spacing=6, border_width=12)
        row = 0
        g.attach(Gtk.Label(label="Folder:", xalign=1.0), 0, row, 1, 1)
        self.folder_btn = Gtk.FileChooserButton(
            title="Choose export folder",
            action=Gtk.FileChooserAction.SELECT_FOLDER)
        folder = ex["folder"] or os.path.expanduser("~")
        self.folder_btn.set_filename(folder)
        g.attach(self.folder_btn, 1, row, 2, 1)

        row += 1
        g.attach(Gtk.Label(label="Base name:", xalign=1.0), 0, row, 1, 1)
        self.basename = Gtk.Entry()
        self.basename.set_text(ex["basename"])
        g.attach(self.basename, 1, row, 2, 1)

        row += 1
        g.attach(Gtk.Label(label="Size multiple:", xalign=1.0), 0, row, 1, 1)
        min_scale = max(_MIN_DIM / max(min(self.base_w, self.base_h), 1),
                        0.05)
        max_scale = min(_MAX_DIM / max(self.base_w, self.base_h),
                        np.sqrt(_MAX_PIXELS / (self.base_w * self.base_h)))
        adj = Gtk.Adjustment(value=min(max(ex["scale"], min_scale),
                                       max_scale),
                             lower=min_scale, upper=max_scale,
                             step_increment=0.25, page_increment=1.0)
        self.scale = Gtk.SpinButton(adjustment=adj, digits=2)
        self.scale.set_tooltip_text(
            "Image size as a multiple of the preview (main view) size")
        g.attach(self.scale, 1, row, 1, 1)
        self.res_label = Gtk.Label()
        g.attach(self.res_label, 2, row, 1, 1)
        self.scale.connect("value-changed", self._update_res)
        self._update_res()

        row += 1
        self.transparent = Gtk.CheckButton(label="Transparent background")
        self.transparent.set_active(ex["transparent"])
        g.attach(self.transparent, 1, row, 2, 1)

        row += 1
        g.attach(Gtk.Label(label="Background color:", xalign=1.0),
                 0, row, 1, 1)
        from .dialogs import _color_button
        self.bg_btn = _color_button(ex["bg_color"])
        g.attach(self.bg_btn, 1, row, 1, 1)
        self.transparent.connect(
            "toggled", lambda b: self.bg_btn.set_sensitive(
                not b.get_active()))
        self.bg_btn.set_sensitive(not ex["transparent"])

        row += 1
        g.attach(Gtk.Label(label="Orbital grid spacing (bohr):", xalign=1.0),
                 0, row, 1, 1)
        from .dialogs import _spin
        self.spacing = _spin(ex["grid_spacing"], 0.03, 1.0, 0.02)
        self.spacing.set_tooltip_text(
            "Isosurfaces are recomputed at this (finer) grid quality "
            "for export")
        g.attach(self.spacing, 1, row, 1, 1)

        row += 1
        g.attach(Gtk.Label(label="PNG compression (0-9):", xalign=1.0),
                 0, row, 1, 1)
        self.compression = _spin(ex["compression"], 0, 9, 1, 0)
        g.attach(self.compression, 1, row, 1, 1)

        row += 1
        self.crop = Gtk.CheckButton(label="Crop images to content")
        self.crop.set_active(ex["crop"])
        self.crop.set_tooltip_text(
            "Remove empty space around the molecule after rendering; the "
            "result can be smaller than the resolution shown above")
        g.attach(self.crop, 1, row, 2, 1)

        self.get_content_area().add(g)
        self.show_all()

    def _update_res(self, *_a):
        s = self.scale.get_value()
        self.res_label.set_text(
            f"→ {int(self.base_w * s)} × {int(self.base_h * s)} px")

    def values(self):
        c = self.bg_btn.get_rgba()
        return {
            "folder": self.folder_btn.get_filename() or
            os.path.expanduser("~"),
            "basename": self.basename.get_text().strip() or "orbital",
            "scale": self.scale.get_value(),
            "transparent": self.transparent.get_active(),
            "bg_color": [c.red, c.green, c.blue],
            "compression": int(self.compression.get_value()),
            "crop": self.crop.get_active(),
            "grid_spacing": self.spacing.get_value(),
        }


def make_filenames(items, orbitals, options):
    """Filename per item following the spec's naming scheme."""
    base = options["basename"]
    orb_items = [it for it in items if it.kind == "orbital"]
    geo_items = [it for it in items if it.kind != "orbital"]
    names = {}
    if orb_items:
        width = max(len(str(int(orbitals.channel_index[it.orb_index])))
                    for it in orb_items)
        for it in orb_items:
            idx = int(orbitals.channel_index[it.orb_index])
            suffix = ""
            if orbitals.unrestricted:
                suffix = "ab"[orbitals.spins[it.orb_index]]
            names[it.id] = f"{base}{idx:0{width}d}{suffix}.png"
    if geo_items:
        width = max(len(str(len(geo_items))), 2)
        for n, it in enumerate(geo_items, start=1):
            names[it.id] = f"{base}{n:0{width}d}.png"
    return names


class ExportJob:
    """Runs the export; construct then call start()."""

    def __init__(self, window, items, filenames, options):
        self.win = window
        self.items = items
        self.filenames = filenames
        self.opt = options
        self.cancelled = False
        self._render_done = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _progress(self, frac, msg):
        GLib.idle_add(self.win.show_progress, frac, msg)

    def _run(self):
        try:
            data = self.win.data
            gp = GridParams(self.win.settings["grid_type"],
                            self.opt["grid_spacing"],
                            self.win.settings["grid_margin"])
            grid = gp.build(data.molecule) if data is not None else None
            ntot = len(self.items)
            for n, item in enumerate(self.items):
                if self.cancelled:
                    break
                payload = None
                if item.kind == "orbital":
                    self._progress(
                        n / ntot,
                        f"Export: computing surface for {item.label} "
                        f"({n + 1}/{ntot})...")
                    key = f"export_{item.orb_index}_{gp.signature()}"
                    vol = self.win.scratch.load_volume(key)
                    if vol is None:
                        vol = evaluate_orbitals(
                            data.basis,
                            data.orbitals.coeffs[[item.orb_index]],
                            grid,
                            progress=lambda f: self._progress(
                                (n + f * 0.8) / ntot,
                                f"Export: evaluating {item.label} "
                                f"({n + 1}/{ntot})..."))[0]
                        self.win.scratch.save_volume(key, vol)
                        GLib.idle_add(self.win.update_scratch_label)
                    pos, neg = isosurface.extract_pair(
                        vol, self.win.settings["isovalue"], grid)
                    payload = {"mesh_pos": pos, "mesh_neg": neg}
                elif item.kind in ("density", "spin", "esp"):
                    # fields are recomputed at export grid quality too
                    settings = self.win.settings
                    pre = "esp" if item.kind == "esp" else "dens"
                    gp_f = GridParams(settings[f"{pre}_grid_type"],
                                      self.opt["grid_spacing"],
                                      settings[f"{pre}_grid_margin"])
                    payload = compute_field_payload(
                        data, self.win.scratch, item.kind, gp_f,
                        settings[f"{pre}_isovalue"],
                        progress=lambda f, m: self._progress(
                            (n + f * 0.85) / ntot,
                            f"Export: {m} ({n + 1}/{ntot})"),
                        cancel=lambda: self.cancelled,
                        key_prefix="export")
                    if payload is None:  # cancelled
                        break
                    GLib.idle_add(self.win.update_scratch_label)
                self._progress((n + 0.9) / ntot,
                               f"Export: rendering {item.label} "
                               f"({n + 1}/{ntot})...")
                self._render_done.clear()
                GLib.idle_add(self._render_and_save, item, payload)
                self._render_done.wait()
            GLib.idle_add(self.win.export_finished, self.cancelled)
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self.win.export_finished, exc)

    def _render_and_save(self, item, payload):
        """Main-loop part: GL render + PNG write for one item.

        payload: high-quality surfaces to swap in for the render (see
        compute_field_payload), or None for items without surfaces."""
        try:
            glview = self.win.glview
            w = int(self.win.glview.get_allocation().width
                    * self.opt["scale"])
            h = int(self.win.glview.get_allocation().height
                    * self.opt["scale"])
            w, h = max(w, _MIN_DIM), max(h, _MIN_DIM)
            old = None
            if payload is not None:
                old = (item.mesh_pos, item.mesh_neg,
                       getattr(item, "esp_colors", None),
                       getattr(item, "esp_range", None))
                item.mesh_pos = payload.get("mesh_pos")
                item.mesh_neg = payload.get("mesh_neg")
                if item.kind == "esp":
                    item.esp_colors = payload.get("esp_colors")
                    item.esp_range = payload.get("esp_range", 0.0)
            # selection highlights must never appear in exported images
            scene = self.win.scene
            saved_sel = scene.selected_atoms
            if saved_sel:
                scene.selected_atoms = []
                scene.invalidate_structure()
            img = glview.render_offscreen(
                item if item.kind != "structure" else None, w, h,
                background=self.opt["bg_color"],
                transparent=self.opt["transparent"])
            if saved_sel:
                scene.selected_atoms = saved_sel
                scene.invalidate_structure()
            if old is not None:
                item.mesh_pos, item.mesh_neg = old[0], old[1]
                if item.kind == "esp":
                    item.esp_colors, item.esp_range = old[2], old[3]
            if img is None:
                raise RuntimeError("GL context unavailable for export")
            pil = Image.fromarray(img, "RGBA")
            if not self.opt["transparent"]:
                pil = pil.convert("RGB")
            if self.opt["crop"]:
                pil = self._crop(pil)
            path = os.path.join(self.opt["folder"],
                                self.filenames[item.id])
            pil.save(path, "PNG",
                     compress_level=self.opt["compression"])
        except Exception as exc:  # noqa: BLE001
            self.cancelled = True
            GLib.idle_add(self.win.show_error, "Export failed", str(exc))
        finally:
            self._render_done.set()
        return False

    def _crop(self, pil, pad=6):
        if pil.mode == "RGBA":
            bbox = pil.getchannel("A").getbbox()
        else:
            bg = Image.new("RGB", pil.size,
                           tuple(int(c * 255)
                                 for c in self.opt["bg_color"]))
            bbox = ImageChops.difference(pil, bg).getbbox()
        if bbox is None:
            return pil
        left = max(bbox[0] - pad, 0)
        top = max(bbox[1] - pad, 0)
        right = min(bbox[2] + pad, pil.width)
        bottom = min(bbox[3] + pad, pil.height)
        return pil.crop((left, top, right, bottom))


def confirm_overwrites(parent, folder, filenames):
    """Prompt per existing file with a 'yes to all' option.

    Returns the set of item ids to skip, or None if cancelled entirely.
    """
    skip = set()
    yes_all = False
    for iid, fname in filenames.items():
        path = os.path.join(folder, fname)
        if not os.path.exists(path) or yes_all:
            continue
        dlg = Gtk.MessageDialog(
            transient_for=parent, modal=True,
            message_type=Gtk.MessageType.QUESTION,
            text=f"Overwrite existing file '{fname}'?")
        dlg.add_buttons("Cancel export", Gtk.ResponseType.CANCEL,
                        "Skip", Gtk.ResponseType.NO,
                        "Yes", Gtk.ResponseType.YES,
                        "Yes to all", Gtk.ResponseType.ACCEPT)
        resp = dlg.run()
        dlg.destroy()
        if resp == Gtk.ResponseType.CANCEL:
            return None
        if resp == Gtk.ResponseType.NO:
            skip.add(iid)
        elif resp == Gtk.ResponseType.ACCEPT:
            yes_all = True
    return skip
