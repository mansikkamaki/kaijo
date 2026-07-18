# SPDX-License-Identifier: GPL-3.0-or-later
# Kaijo -- molecular orbital and molecular structure visualization.
# Copyright (C) 2026  Akseli Mansikkamäki
#
# Developed by Akseli Mansikkamäki and constructed with AI using Claude
# Code (models: Claude Opus 4.8 and Claude Fable 5.0).  Licensed under the
# GNU General Public License, version 3 or later; see the LICENSE file.
"""Kaijo entry point."""

import argparse
import sys
import warnings


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="kaijo",
        description="Molecular orbital and structure visualization")
    parser.add_argument("file", nargs="?",
                        help="molden or xyz file to open")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")

    from .ui.mainwindow import KaijoApp

    app = KaijoApp()
    if args.file:
        app.set_start_file(args.file)
    return app.run(None)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
