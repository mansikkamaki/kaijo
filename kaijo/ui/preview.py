# SPDX-License-Identifier: GPL-3.0-or-later
"""Preview grid: one thumbnail per scene item, sharing the main camera.

Thumbnails are rendered offscreen through the main view's GL context, so
every cell shows exactly the orientation of the main view.  Re-rendering is
debounced while the user interacts with the main view.
"""

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk

_THUMB_W, _THUMB_H = 240, 190


class PreviewCell(Gtk.Frame):
    def __init__(self, item, on_activate, on_export_toggle, on_close=None):
        super().__init__()
        self.item = item
        self.set_shadow_type(Gtk.ShadowType.IN)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.set_border_width(3)
        self.add(box)

        ebox = Gtk.EventBox()
        self.image = Gtk.Image()
        self.image.set_size_request(_THUMB_W, _THUMB_H)
        ebox.add(self.image)
        ebox.connect("button-press-event",
                     lambda w, e: on_activate(self.item))
        box.pack_start(ebox, True, True, 0)

        # Caption can span several lines (orbital index, symmetry, energy,
        # occupation), so wrap rather than ellipsize to a single line.
        self.label = Gtk.Label(label=item.describe())
        self.label.set_line_wrap(True)
        self.label.set_justify(Gtk.Justification.LEFT)
        self.label.set_xalign(0.0)
        box.pack_start(self.label, False, False, 0)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        if on_close is not None:
            close = Gtk.Button.new_from_icon_name(
                "window-close-symbolic", Gtk.IconSize.MENU)
            close.set_relief(Gtk.ReliefStyle.NONE)
            close.set_focus_on_click(False)
            close.set_tooltip_text("Remove this window from the "
                                   "preview grid")
            close.connect("clicked", lambda b: on_close(self.item))
            hbox.pack_end(close, False, False, 0)
        self.check = Gtk.CheckButton(label="select")
        self.check.set_tooltip_text(
            "Select this window (for image export or the ORCA CAS tool)")
        self.check.set_active(item.export_selected)
        self.check.connect("toggled", lambda b: on_export_toggle(
            self.item, b.get_active()))
        hbox.pack_start(self.check, False, False, 2)
        box.pack_start(hbox, False, False, 0)
        self.show_all()

    def set_pixbuf(self, pixbuf):
        self.image.set_from_pixbuf(pixbuf)

    def set_active(self, active):
        ctx = self.get_style_context()
        if active:
            ctx.add_class("kaijo-active-cell")
        else:
            ctx.remove_class("kaijo-active-cell")

    def refresh_label(self):
        self.label.set_text(self.item.describe())


class PreviewGrid(Gtk.ScrolledWindow):
    def __init__(self, scene, glview, settings):
        super().__init__()
        self.scene = scene
        self.glview = glview
        self.settings = settings
        self.on_item_activated = None  # callback(item)
        self.on_item_closed = None     # callback(item)
        self._cells = {}               # item.id -> PreviewCell
        self._debounce = 0
        self._render_queue = []        # cells still to render this pass
        self._render_source = 0        # idle id of the incremental walk

        self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.flow = Gtk.FlowBox()
        self.flow.set_selection_mode(Gtk.SelectionMode.NONE)
        self.flow.set_homogeneous(True)
        self.flow.set_valign(Gtk.Align.START)
        self.flow.set_min_children_per_line(1)
        self.flow.set_max_children_per_line(30)  # as many as fit the width
        self.add(self.flow)

        css = Gtk.CssProvider()
        css.load_from_data(
            b".kaijo-active-cell { border: 2px solid #ff9900; }")
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), css,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    # -------------------------------------------------------------- content

    def sync_items(self):
        """Make the cells match scene.items (order preserved)."""
        wanted = {it.id for it in self.scene.items}
        for iid in [i for i in self._cells if i not in wanted]:
            cell = self._cells.pop(iid)
            parent = cell.get_parent()
            if parent is not None:
                self.flow.remove(parent)
        for pos, item in enumerate(self.scene.items):
            if item.id not in self._cells:
                cell = PreviewCell(
                    item, self._activate, self._export_toggle,
                    on_close=self._close if item.kind != "structure"
                    else None)
                self._cells[item.id] = cell
                self.flow.insert(cell, pos)
        self.flow.show_all()

    def _activate(self, item):
        if self.on_item_activated:
            self.on_item_activated(item)

    def _export_toggle(self, item, active):
        item.export_selected = active

    def _close(self, item):
        if self.on_item_closed:
            self.on_item_closed(item)

    def set_active_item(self, item):
        for iid, cell in self._cells.items():
            cell.set_active(item is not None and iid == item.id)

    def cell_for(self, item):
        return self._cells.get(item.id)

    # ------------------------------------------------------------ rendering

    def refresh_thumbnails(self, immediate=False):
        """Debounced re-render of all thumbnails with the shared camera."""
        if immediate:
            self._do_render()
            return
        if self._debounce:
            GLib.source_remove(self._debounce)
        self._debounce = GLib.timeout_add(160, self._debounced)

    def _debounced(self):
        self._debounce = 0
        self._do_render()
        return False

    def refresh_one(self, item):
        cell = self._cells.get(item.id)
        if cell is not None:
            self._render_cell(cell)

    def _do_render(self):
        """Re-render every cell, but one cell per main-loop iteration so a
        large grid never blocks input.  A new pass cancels the previous
        walk instead of stacking on top of it."""
        if self._render_source:
            GLib.source_remove(self._render_source)
        self._render_queue = list(self._cells.values())
        self._render_source = GLib.idle_add(self._render_step)

    def _render_step(self):
        if not self._render_queue:
            self._render_source = 0
            return False
        cell = self._render_queue.pop(0)
        # skip cells removed since the pass started
        if self._cells.get(cell.item.id) is cell:
            self._render_cell(cell)
        if not self._render_queue:
            self._render_source = 0
            return False
        return True

    def _render_cell(self, cell):
        if not self.glview.get_realized() or self.glview.renderer is None:
            return
        img = self.glview.render_offscreen(
            cell.item if cell.item.kind != "structure" else None,
            _THUMB_W, _THUMB_H,
            background=self.settings["background"], transparent=False,
            reuse=True)
        if img is None:
            return
        h, w, _ = img.shape
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            GLib.Bytes.new(img.tobytes()), GdkPixbuf.Colorspace.RGB,
            True, 8, w, h, w * 4)
        cell.set_pixbuf(pixbuf)
