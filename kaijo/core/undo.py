# SPDX-License-Identifier: GPL-3.0-or-later
"""Undo/redo command stack."""


class Command:
    """A reversible action.  Subclasses implement do() and undo()."""

    label = ""

    def do(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


class FuncCommand(Command):
    def __init__(self, label, do_func, undo_func):
        self.label = label
        self._do = do_func
        self._undo = undo_func

    def do(self):
        self._do()

    def undo(self):
        self._undo()


class UndoStack:
    def __init__(self, limit=200):
        self._undo = []
        self._redo = []
        self._limit = limit
        self.on_change = None  # callback after any stack operation

    def push(self, command):
        """Execute the command and record it."""
        command.do()
        self._undo.append(command)
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()
        self._notify()

    def undo(self):
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        self._notify()
        return cmd

    def redo(self):
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.do()
        self._undo.append(cmd)
        self._notify()
        return cmd

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def _notify(self):
        if self.on_change:
            self.on_change()
