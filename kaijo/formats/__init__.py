# SPDX-License-Identifier: GPL-3.0-or-later
"""File-format registry.

Each format module registers a `FormatHandler`; `load_file` sniffs the file
and dispatches.  New formats are added by appending to HANDLERS.
"""

import os


class LoadedData:
    """Result of loading a file: geometry always, orbitals when available."""

    def __init__(self, molecule, basis=None, orbitals=None, path=""):
        self.molecule = molecule
        self.basis = basis          # core.basis.BasisSet or None
        self.orbitals = orbitals    # core.basis.MolecularOrbitals or None
        self.path = path

    @property
    def has_orbitals(self):
        return self.basis is not None and self.orbitals is not None


class FormatHandler:
    name = "abstract"
    extensions = ()

    def sniff(self, path, head):
        """head: first ~2 kB of the file as text."""
        return any(path.lower().endswith(e) for e in self.extensions)

    def load(self, path, progress=None):
        raise NotImplementedError


def _handlers():
    from . import molden, xyz, fchk
    return [molden.MoldenHandler(), fchk.FchkHandler(), xyz.XYZHandler()]


HANDLERS = None


def load_file(path, progress=None):
    """Load a structure/orbital file, auto-detecting the format."""
    global HANDLERS
    if HANDLERS is None:
        HANDLERS = _handlers()
    with open(path, "r", errors="replace") as fh:
        head = fh.read(2048)
    for h in HANDLERS:
        if h.sniff(path, head):
            return h.load(path, progress=progress)
    raise ValueError(f"Unrecognised file format: {os.path.basename(path)}")


def file_filters():
    """(name, patterns) list for the file-chooser dialog."""
    return [
        ("All supported", ["*.molden", "*.molden.input", "*.input",
                           "*.fchk", "*.fch", "*.fck", "*.xyz"]),
        ("Molden files", ["*.molden", "*.molden.input", "*.input"]),
        ("Gaussian fchk files", ["*.fchk", "*.fch", "*.fck"]),
        ("XYZ files", ["*.xyz"]),
        ("All files", ["*"]),
    ]
