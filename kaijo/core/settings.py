# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent user settings (JSON in the XDG config directory)."""

import copy
import json
import os

DEFAULTS = {
    "representation": "ball-and-stick",  # ball-and-stick | sticks | wireframe
    "show_hydrogens": True,
    "multiple_bonds": True,              # render bond orders separately
    "atom_scale": 1.0,
    "bond_radius": 0.13,                 # Å
    "atom_colors": {},                   # per-element overrides {Z: [r,g,b]}
    "atom_radii": {},                    # per-element overrides {Z: r}
    "isovalue": 0.03,
    "iso_color_pos": [0.85, 0.10, 0.10],
    "iso_color_neg": [0.10, 0.25, 0.85],
    "iso_alpha": 1.0,                    # 1.0 = opaque
    "background": [1.0, 1.0, 1.0],
    "grid_spacing": 0.35,                # bohr, preview quality
    "grid_margin": 4.0,                  # bohr
    "grid_type": "rectangular",
    # densities (electron + spin density) have their own grid and
    # surface settings, kept separate from the orbital ones
    "dens_isovalue": 0.002,              # e / bohr^3
    "dens_color_pos": [0.95, 0.60, 0.10],
    "dens_color_neg": [0.10, 0.60, 0.40],
    "dens_alpha": 1.0,
    "dens_grid_spacing": 0.35,
    "dens_grid_margin": 4.0,
    "dens_grid_type": "rectangular",
    # ESP: again its own grid settings; the isovalue is the density
    # isovalue of the carrier surface the potential is mapped onto, and
    # the color gradient itself is fixed (not user-configurable)
    "esp_isovalue": 0.001,               # e / bohr^3 (carrier surface)
    "esp_alpha": 1.0,
    "esp_grid_spacing": 0.35,
    "esp_grid_margin": 5.0,
    "esp_grid_type": "rectangular",
    "preview_columns": 3,
    "export": {
        "folder": "",
        "basename": "orbital",
        "scale": 2.0,
        "transparent": True,
        "bg_color": [1.0, 1.0, 1.0],
        "compression": 9,
        "crop": False,
        "grid_spacing": 0.15,
    },
}


class Settings:
    def __init__(self):
        base = os.environ.get("XDG_CONFIG_HOME",
                              os.path.expanduser("~/.config"))
        self.path = os.path.join(base, "kaijo", "settings.json")
        self.data = copy.deepcopy(DEFAULTS)
        self.load()

    def load(self):
        try:
            with open(self.path) as fh:
                stored = json.load(fh)
        except (OSError, ValueError):
            return
        self._merge(self.data, stored)

    @staticmethod
    def _merge(dst, src):
        for k, v in src.items():
            if k in dst and isinstance(dst[k], dict) and isinstance(v, dict) \
                    and k not in ("atom_colors", "atom_radii"):
                Settings._merge(dst[k], v)
            else:
                dst[k] = v

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=1)
        os.replace(tmp, self.path)

    def reset(self):
        self.data = copy.deepcopy(DEFAULTS)
        self.save()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
