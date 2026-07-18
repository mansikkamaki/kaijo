# SPDX-License-Identifier: GPL-3.0-or-later
"""Scratch-disk management.

Computed orbital volumes are cached here as .npy files so that changing the
isovalue only re-runs marching cubes instead of re-evaluating the orbitals.
The whole directory is removed when the program exits (also on SIGINT and
SIGTERM), as required by the design spec.
"""

import atexit
import os
import shutil
import signal
import tempfile

import numpy as np


class ScratchManager:
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="kaijo-")
        atexit.register(self.cleanup)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, self._make_handler(prev))
            except (ValueError, OSError):
                pass

    def _make_handler(self, prev):
        def handler(signum, frame):
            self.cleanup()
            if callable(prev):
                prev(signum, frame)
            else:
                raise SystemExit(128 + signum)
        return handler

    def path(self, name):
        return os.path.join(self.dir, name)

    def save_volume(self, key, volume):
        np.save(self.path(key + ".npy"), volume)

    def load_volume(self, key):
        p = self.path(key + ".npy")
        if os.path.exists(p):
            return np.load(p)
        return None

    def has_volume(self, key):
        return os.path.exists(self.path(key + ".npy"))

    def delete_volume(self, key):
        p = self.path(key + ".npy")
        if os.path.exists(p):
            os.unlink(p)

    def clear_volumes(self):
        for f in os.listdir(self.dir):
            if f.endswith(".npy"):
                os.unlink(os.path.join(self.dir, f))

    def usage_bytes(self):
        total = 0
        try:
            for f in os.listdir(self.dir):
                total += os.path.getsize(os.path.join(self.dir, f))
        except OSError:
            pass
        return total

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)
